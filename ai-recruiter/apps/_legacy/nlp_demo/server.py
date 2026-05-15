"""
FastAPI Backend — AI Recruiter Interview Simulator

Replaces Streamlit. Exposes:
  POST /api/session          — create session
  DELETE /api/session/{sid}  — reset session
  GET  /api/status           — agent/model health
  GET  /api/stream           — SSE: streams text tokens + base64 audio chunks
  POST /api/chat             — non-streaming fallback (returns full response)

SSE stream format (one JSON object per line, prefixed with "data: "):
  { "type": "token",  "text": "..." }          — text token to append
  { "type": "audio",  "data": "<base64 wav>" } — PCM audio chunk to play
  { "type": "done",   "reasoning": "...", "is_follow_up": bool }
  { "type": "error",  "message": "..." }

Run with:
  uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import asyncio
import base64
import json
import sys
import os
import uuid
from typing import Optional

# Add parent directory so agent imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="AI Recruiter API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Agent singleton (loaded once on startup)
# ---------------------------------------------------------------------------

_agent = None


def get_agent():
    global _agent
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not ready yet — try again in a moment")
    return _agent


@app.on_event("startup")
async def startup():
    global _agent
    print("[server] Loading NLP agent...")
    from agent import NLPAgent
    _agent = NLPAgent(
        model_path="/teamspace/studios/this_studio/Agentic-Ai-Recruiter/model_cache",
        enable_scorer=True,  # Enable answer analysis
        scorer_model="Qwen/Qwen2.5-1.5B-Instruct",
        enable_refiner=True,  # Enable question refinement
        refiner_model="Qwen/Qwen2.5-1.5B-Instruct",
        enable_tts=True,
        tts_voice="af_heart",
        use_compact_prompt=False,
        auto_cleanup_gpu=False,
        load_timeout_sec=300,
    )
    print("[server] Warming up models...")
    await _agent.warmup()
    print("[server] Agent ready.")

    
# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class SessionCreateRequest(BaseModel):
    cv_text: str
    job_role: str = "AI Engineering Intern"


class SessionResponse(BaseModel):
    session_id: str


class ChatRequest(BaseModel):
    session_id: str
    cv_text: str
    job_role: str
    answer: str
    history: list = []


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/session", response_model=SessionResponse)
async def create_session(req: SessionCreateRequest):
    sid = f"sess-{uuid.uuid4().hex[:12]}"
    agent = get_agent()
    # Pre-create session with CV topics
    agent._get_or_create_session(sid, req.cv_text)
    agent._get_or_create_prompt(sid, req.cv_text, req.job_role)
    return SessionResponse(session_id=sid)


@app.delete("/api/session/{session_id}")
async def reset_session(session_id: str):
    agent = get_agent()
    agent.reset_session(session_id)
    return {"ok": True}


@app.get("/api/status")
async def get_status():
    global _agent
    if _agent is None:
        return {"ready": False, "message": "Agent loading..."}
    try:
        status = _agent.get_status()
        return {
            "ready": True,
            "engine_ready": status["engine_ready"],
            "tts_ready": status["tts_ready"],
            "scorer_ready": status["scorer_ready"],
            "gpu_free_gb": round(status["gpu_free_gb"], 1),
            "gpu_total_gb": round(status["gpu_total_gb"], 1),
        }
    except Exception as e:
        return {"ready": False, "message": str(e)}

@app.get("/api/stream")
async def stream_response(
    session_id: str = Query(...),
    cv_text: str = Query(...),
    job_role: str = Query("AI Engineering Intern"),
    answer: str = Query(...),
    history: str = Query("[]"),  # JSON encoded list of [candidate, recruiter] pairs
):
    """
    SSE endpoint — streams text tokens and audio chunks as they're generated.

    The client should:
      1. Open an EventSource to this URL with the query params
      2. On "token" events: append text to display buffer
      3. On "audio" events: decode base64, create a Blob, play via AudioContext
      4. On "done": finalize display, stop orb animation
    """
    agent = get_agent()

    try:
        history_parsed = json.loads(history)
    except Exception:
        history_parsed = []

    from agent import QuestionGenerationRequest
    request = QuestionGenerationRequest(
        candidate_cv=cv_text,
        job_role=job_role,
        conversation_history=history_parsed[-6:],
        candidate_latest_answer=answer,
        session_id=session_id,
    )

    async def event_generator():
        reasoning = "unknown"
        is_follow_up = False
        try:
            async for chunk in agent.generate_question_streaming_with_audio(request):
                # Text token
                if chunk.text_delta:
                    payload = json.dumps({"type": "token", "text": chunk.text_delta})
                    yield f"data: {payload}\n\n"

                # Audio chunk (one sentence worth of PCM)
                if chunk.tts_result is not None:
                    wav_bytes = chunk.tts_result.to_wav_bytes()
                    b64 = base64.b64encode(wav_bytes).decode("utf-8")
                    payload = json.dumps({"type": "audio", "data": b64})
                    yield f"data: {payload}\n\n"

                # Final chunk carries metadata
                if chunk.is_final:
                    payload = json.dumps({
                        "type": "done",
                        "reasoning": reasoning,
                        "is_follow_up": is_follow_up,
                    })
                    yield f"data: {payload}\n\n"
                    return

        except Exception as e:
            import traceback
            traceback.print_exc()
            payload = json.dumps({"type": "error", "message": str(e)})
            yield f"data: {payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )


@app.post("/api/chat")
async def chat_non_streaming(req: ChatRequest):
    """Non-streaming fallback — returns complete response at once."""
    agent = get_agent()
    from agent import QuestionGenerationRequest
    request = QuestionGenerationRequest(
        candidate_cv=req.cv_text,
        job_role=req.job_role,
        conversation_history=req.history[-6:],
        candidate_latest_answer=req.answer,
        session_id=req.session_id,
    )
    response = await agent.generate_question(request)
    audio_b64 = None
    if response.audio:
        try:
            audio_b64 = base64.b64encode(response.audio.to_wav_bytes()).decode("utf-8")
        except Exception:
            pass

    return {
        "question": response.question,
        "reasoning": response.reasoning,
        "is_follow_up": response.is_follow_up,
        "audio_b64": audio_b64,
        "metrics": {
            "latency_sec": response.metrics.latency_sec if response.metrics else None,
            "num_tokens": response.metrics.num_tokens if response.metrics else None,
        },
    }


class TTSRequest(BaseModel):
    text: str


@app.post("/api/tts")
async def text_to_speech(req: TTSRequest):
    """Convert text to speech and return base64-encoded WAV audio."""
    agent = get_agent()

    print(f"[TTS] Received request for {len(req.text)} characters")

    if not agent._tts or not agent._tts.is_ready:
        print("[TTS] ERROR: TTS not enabled or not ready")
        raise HTTPException(status_code=503, detail="TTS not enabled")

    try:
        # Generate TTS audio using TTSEngine.synthesize()
        print("[TTS] Generating speech...")
        audio_result = await agent._tts.synthesize(req.text)
        print(f"[TTS] Generated audio")

        # Convert to WAV and encode as base64
        wav_bytes = audio_result.to_wav_bytes()
        print(f"[TTS] WAV bytes: {len(wav_bytes)}")
        audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
        print(f"[TTS] Base64 length: {len(audio_b64)}")

        return {"audio_b64": audio_b64}
    except Exception as e:
        import traceback
        print("[TTS] ERROR:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")