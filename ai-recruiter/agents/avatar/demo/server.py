"""
FastAPI Backend — AI Recruiter Avatar Interview Demo

Unified server that combines the NLP interview pipeline with the
3D avatar lip-sync system. Extends the NLP demo server with avatar
endpoints for GLB model serving and viseme extraction.

Endpoints:
    --- NLP Pipeline (unchanged) ---
    POST /api/session          — Create interview session
    DELETE /api/session/{sid}  — Reset session
    GET  /api/status           — Agent/model health
    GET  /api/stream           — SSE: text tokens + audio + visemes
    POST /api/chat             — Non-streaming fallback
    POST /api/tts              — Text-to-speech synthesis

    --- Avatar ---
    GET  /avatar/model.glb     — Serve the GLB avatar file
    GET  /avatar/status        — Avatar agent health
    POST /avatar/visemes       — Extract visemes from base64 WAV audio

Run with:
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Optional

# Resolve paths so imports work regardless of cwd
DEMO_DIR = Path(__file__).resolve().parent
AVATAR_DIR = DEMO_DIR.parent
AGENTS_DIR = AVATAR_DIR.parent
NLP_DIR = AGENTS_DIR / "nlp"

sys.path.insert(0, str(NLP_DIR))
sys.path.insert(0, str(AGENTS_DIR))

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
LOG_FORMAT = (
    "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"
)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("avatar.server")

# Quiet down noisy libraries
for noisy in ("urllib3", "httpcore", "httpx", "uvicorn.access"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI Recruiter Avatar API",
    version="2.0",
    description="NLP Interview Pipeline + 3D Avatar Lip Sync",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

_nlp_agent = None
_avatar_agent = None


def get_nlp_agent():
    if _nlp_agent is None:
        raise HTTPException(status_code=503, detail="NLP agent not ready yet")
    return _nlp_agent


def get_avatar_agent():
    if _avatar_agent is None:
        raise HTTPException(status_code=503, detail="Avatar agent not ready yet")
    return _avatar_agent


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    global _nlp_agent, _avatar_agent

    logger.info("=" * 60)
    logger.info("  AI Recruiter Avatar Server — Starting")
    logger.info("=" * 60)
    start = time.monotonic()

    # --- Load NLP Agent ---
    logger.info("[1/2] Loading NLP agent...")
    try:
        from agent import NLPAgent
        _nlp_agent = NLPAgent(
            model_path="/teamspace/studios/this_studio/Agentic-Ai-Recruiter/model_cache",
            enable_scorer=True,
            scorer_model="Qwen/Qwen2.5-1.5B-Instruct",
            enable_refiner=True,
            refiner_model="Qwen/Qwen2.5-1.5B-Instruct",
            enable_tts=True,
            tts_voice="af_heart",
            use_compact_prompt=False,
            auto_cleanup_gpu=False,
            load_timeout_sec=300,
        )
        logger.info("[1/2] Warming up NLP models...")
        await _nlp_agent.warmup()
        logger.info("[1/2] NLP agent ready")
    except Exception as e:
        logger.error("[1/2] NLP agent failed to load: %s", e, exc_info=True)
        raise

    # --- Load Avatar Agent ---
    logger.info("[2/2] Loading Avatar agent...")
    try:
        from avatar.agent import AvatarAgent, AvatarConfig
        _avatar_agent = AvatarAgent(config=AvatarConfig(
            glb_path=str(AVATAR_DIR / "brunette.glb"),
        ))
        await _avatar_agent.initialize()
        logger.info("[2/2] Avatar agent ready")
    except Exception as e:
        logger.error("[2/2] Avatar agent failed: %s", e, exc_info=True)
        raise

    elapsed = time.monotonic() - start
    logger.info("=" * 60)
    logger.info("  Server ready in %.1fs", elapsed)
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown():
    logger.info("Server shutting down...")


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


class TTSRequest(BaseModel):
    text: str


class VisemeRequest(BaseModel):
    audio_b64: str


# ---------------------------------------------------------------------------
# Middleware — request logging
# ---------------------------------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    elapsed_ms = (time.monotonic() - start) * 1000
    logger.debug(
        "%s %s → %d (%.0fms)",
        request.method, request.url.path, response.status_code, elapsed_ms,
    )
    return response


# ============================================================================
# NLP ENDPOINTS (same as NLP demo server)
# ============================================================================

@app.post("/api/session", response_model=SessionResponse)
async def create_session(req: SessionCreateRequest):
    logger.info("Creating session for role=%s cv_len=%d", req.job_role, len(req.cv_text))
    sid = f"sess-{uuid.uuid4().hex[:12]}"
    agent = get_nlp_agent()
    agent._get_or_create_session(sid, req.cv_text)
    agent._get_or_create_prompt(sid, req.cv_text, req.job_role)
    logger.info("Session created: %s", sid)
    return SessionResponse(session_id=sid)


@app.delete("/api/session/{session_id}")
async def reset_session(session_id: str):
    logger.info("Resetting session: %s", session_id)
    agent = get_nlp_agent()
    agent.reset_session(session_id)
    return {"ok": True}


@app.get("/api/status")
async def get_status():
    """Combined health status for NLP + Avatar agents."""
    result = {"nlp_ready": False, "avatar_ready": False}

    if _nlp_agent:
        try:
            status = _nlp_agent.get_status()
            result.update({
                "nlp_ready": True,
                "engine_ready": status["engine_ready"],
                "tts_ready": status["tts_ready"],
                "scorer_ready": status["scorer_ready"],
                "gpu_free_gb": round(status["gpu_free_gb"], 1),
                "gpu_total_gb": round(status["gpu_total_gb"], 1),
            })
        except Exception as e:
            logger.error("NLP status check failed: %s", e)
            result["nlp_error"] = str(e)

    if _avatar_agent:
        try:
            avatar_status = _avatar_agent.get_status()
            result.update({
                "avatar_ready": avatar_status["ready"],
                "glb_loaded": avatar_status["glb_loaded"],
                "glb_size_mb": avatar_status["glb_file_size_mb"],
                "viseme_engine_ready": avatar_status["viseme_engine_ready"],
            })
        except Exception as e:
            logger.error("Avatar status check failed: %s", e)
            result["avatar_error"] = str(e)

    result["ready"] = result.get("nlp_ready", False) and result.get("avatar_ready", False)
    return result


@app.get("/api/stream")
async def stream_response(
    session_id: str = Query(...),
    cv_text: str = Query(...),
    job_role: str = Query("AI Engineering Intern"),
    answer: str = Query(...),
    history: str = Query("[]"),
):
    """
    SSE endpoint — streams text tokens, audio chunks, and viseme data.

    Enhanced over the NLP demo to include viseme timelines for each audio
    chunk, enabling real-time avatar lip sync.

    Event types:
        token   — { type: "token", text: "..." }
        audio   — { type: "audio", data: "<base64 wav>" }
        visemes — { type: "visemes", data: [...] }
        done    — { type: "done", reasoning: "...", is_follow_up: bool }
        error   — { type: "error", message: "..." }
    """
    agent = get_nlp_agent()
    avatar = _avatar_agent  # May be None if not loaded

    logger.info(
        "SSE stream | session=%s answer_len=%d",
        session_id, len(answer),
    )

    try:
        history_parsed = json.loads(history)
    except Exception:
        logger.warning("Failed to parse history JSON, using empty list")
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
        chunk_count = 0
        audio_count = 0
        stream_start = time.monotonic()

        try:
            async for chunk in agent.generate_question_streaming_with_audio(request):
                # Text token
                if chunk.text_delta:
                    chunk_count += 1
                    payload = json.dumps({"type": "token", "text": chunk.text_delta})
                    yield f"data: {payload}\n\n"

                # Audio chunk + viseme data
                if chunk.tts_result is not None:
                    audio_count += 1
                    wav_bytes = chunk.tts_result.to_wav_bytes()
                    b64 = base64.b64encode(wav_bytes).decode("utf-8")

                    # Send audio
                    payload = json.dumps({"type": "audio", "data": b64})
                    yield f"data: {payload}\n\n"

                    # Extract and send visemes for this audio chunk
                    if avatar:
                        try:
                            visemes = avatar.extract_visemes(wav_bytes)
                            payload = json.dumps({"type": "visemes", "data": visemes})
                            yield f"data: {payload}\n\n"
                        except Exception as ve:
                            logger.warning("Viseme extraction failed: %s", ve)

                # Final chunk
                if chunk.is_final:
                    elapsed = time.monotonic() - stream_start
                    logger.info(
                        "SSE complete | session=%s tokens=%d audio_chunks=%d time=%.2fs",
                        session_id, chunk_count, audio_count, elapsed,
                    )
                    payload = json.dumps({
                        "type": "done",
                        "reasoning": reasoning,
                        "is_follow_up": is_follow_up,
                    })
                    yield f"data: {payload}\n\n"
                    return

        except Exception as e:
            logger.error("SSE stream error: %s", e, exc_info=True)
            payload = json.dumps({"type": "error", "message": str(e)})
            yield f"data: {payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/chat")
async def chat_non_streaming(req: ChatRequest):
    """Non-streaming fallback with optional viseme data."""
    logger.info("Non-streaming chat | session=%s", req.session_id)
    agent = get_nlp_agent()
    avatar = _avatar_agent

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
    visemes = None
    if response.audio:
        try:
            wav_bytes = response.audio.to_wav_bytes()
            audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
            if avatar:
                visemes = avatar.extract_visemes(wav_bytes)
        except Exception as e:
            logger.error("Audio/viseme processing failed: %s", e)

    return {
        "question": response.question,
        "reasoning": response.reasoning,
        "is_follow_up": response.is_follow_up,
        "audio_b64": audio_b64,
        "visemes": visemes,
        "metrics": {
            "latency_sec": response.metrics.latency_sec if response.metrics else None,
            "num_tokens": response.metrics.num_tokens if response.metrics else None,
        },
    }


@app.post("/api/tts")
async def text_to_speech(req: TTSRequest):
    """Convert text to speech with optional viseme extraction."""
    agent = get_nlp_agent()
    avatar = _avatar_agent

    logger.info("TTS request | text_len=%d", len(req.text))

    if not agent._tts or not agent._tts.is_ready:
        logger.error("TTS not enabled or not ready")
        raise HTTPException(status_code=503, detail="TTS not enabled")

    try:
        audio_result = await agent._tts.synthesize(req.text)
        wav_bytes = audio_result.to_wav_bytes()
        audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")

        visemes = None
        if avatar:
            try:
                visemes = avatar.extract_visemes(wav_bytes)
            except Exception as ve:
                logger.warning("TTS viseme extraction failed: %s", ve)

        logger.info("TTS complete | wav_bytes=%d viseme_frames=%d",
                     len(wav_bytes), len(visemes) if visemes else 0)
        return {"audio_b64": audio_b64, "visemes": visemes}

    except Exception as e:
        logger.error("TTS generation failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"TTS failed: {e}")


# ============================================================================
# AVATAR ENDPOINTS
# ============================================================================

@app.get("/avatar/model.glb")
async def serve_avatar_model():
    """Serve the GLB avatar model file."""
    avatar = get_avatar_agent()
    glb_path = avatar.get_glb_path()
    logger.debug("Serving GLB avatar: %s", glb_path)
    return FileResponse(
        path=str(glb_path),
        media_type="model/gltf-binary",
        filename="avatar.glb",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.get("/avatar/status")
async def avatar_status():
    """Detailed avatar agent status."""
    avatar = get_avatar_agent()
    return avatar.get_status()


@app.post("/avatar/visemes")
async def extract_visemes(req: VisemeRequest):
    """
    Extract viseme timeline from base64-encoded WAV audio.

    Used when client needs server-side viseme extraction
    (e.g., for pre-computed animations or non-real-time scenarios).
    """
    avatar = get_avatar_agent()

    logger.debug("Viseme extraction request | b64_len=%d", len(req.audio_b64))

    try:
        wav_bytes = base64.b64decode(req.audio_b64)
    except Exception as e:
        logger.error("Invalid base64 audio: %s", e)
        raise HTTPException(status_code=400, detail="Invalid base64 audio data")

    try:
        visemes = avatar.extract_visemes(wav_bytes)
        logger.info("Viseme extraction complete | frames=%d", len(visemes))
        return {"visemes": visemes}
    except Exception as e:
        logger.error("Viseme extraction failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Viseme extraction failed: {e}")


# ============================================================================
# Health check
# ============================================================================

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}
