"""
FastAPI Backend — AI Recruiter Voice Interview Demo

Extends the Avatar demo server with real-time Speech-to-Text (STT)
using Moonshine Voice + Silero VAD. Supports both WebSocket-based microphone
streaming and HTTP POST for audio file transcription.

Endpoints:
    --- NLP Pipeline ---
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

    --- Voice / STT ---
    WS   /ws/transcribe        — Real-time mic streaming → live transcript
    POST /api/transcribe       — Upload audio file → transcript

Run with:
    uvicorn server:app --host 0.0.0.0 --port 8002 --reload
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import sys
import time
import uuid
import wave
from pathlib import Path
from typing import Optional

import numpy as np

# Resolve paths so imports work regardless of cwd
DEMO_DIR = Path(__file__).resolve().parent
VOICE_DIR = DEMO_DIR.parent
AGENTS_DIR = VOICE_DIR.parent
AVATAR_DIR = AGENTS_DIR / "avatar"
NLP_DIR = AGENTS_DIR / "nlp"

sys.path.insert(0, str(NLP_DIR))
sys.path.insert(0, str(AGENTS_DIR))

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
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
logger = logging.getLogger("voice.server")

# Quiet down noisy libraries
for noisy in ("urllib3", "httpcore", "httpx", "uvicorn.access"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI Recruiter Voice API",
    version="2.0",
    description="NLP Interview Pipeline + 3D Avatar + Voice STT (Moonshine + Silero VAD)",
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
_moonshine_transcriber = None
_moonshine_model_path = None
_moonshine_model_arch = None
_silero_vad_model = None


def get_nlp_agent():
    if _nlp_agent is None:
        raise HTTPException(status_code=503, detail="NLP agent not ready yet")
    return _nlp_agent


def get_avatar_agent():
    if _avatar_agent is None:
        raise HTTPException(status_code=503, detail="Avatar agent not ready yet")
    return _avatar_agent


def get_moonshine():
    if _moonshine_transcriber is None:
        raise HTTPException(status_code=503, detail="Moonshine STT not ready yet")
    return _moonshine_transcriber


def get_silero_vad():
    if _silero_vad_model is None:
        raise HTTPException(status_code=503, detail="Silero VAD not ready yet")
    return _silero_vad_model


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    global _nlp_agent, _avatar_agent, _moonshine_transcriber
    global _moonshine_model_path, _moonshine_model_arch, _silero_vad_model

    logger.info("=" * 60)
    logger.info("  AI Recruiter Voice Server — Starting")
    logger.info("=" * 60)
    start = time.monotonic()

    # --- Load NLP Agent ---
    logger.info("[1/3] Loading NLP agent...")
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
        logger.info("[1/3] Warming up NLP models...")
        await _nlp_agent.warmup()
        logger.info("[1/3] NLP agent ready")
    except Exception as e:
        logger.error("[1/3] NLP agent failed to load: %s", e, exc_info=True)
        raise

    # --- Load Avatar Agent ---
    logger.info("[2/3] Loading Avatar agent...")
    try:
        from avatar.agent import AvatarAgent, AvatarConfig
        _avatar_agent = AvatarAgent(config=AvatarConfig(
            glb_path=str(AVATAR_DIR / "brunette.glb"),
        ))
        await _avatar_agent.initialize()
        logger.info("[2/3] Avatar agent ready")
    except Exception as e:
        logger.error("[2/3] Avatar agent failed: %s", e, exc_info=True)
        raise
    # --- Load Moonshine STT + Silero VAD ---
    logger.info("[3/3] Loading Moonshine STT + Silero VAD...")
    try:
        import torch
        from moonshine_voice import (
            Transcriber,
            TranscriptEventListener,
            get_model_for_language,
        )
        # Download / locate Moonshine English model
        _moonshine_model_path, _moonshine_model_arch = get_model_for_language("en")
        logger.info(
            "[3/3] Moonshine model path=%s arch=%s",
            _moonshine_model_path, _moonshine_model_arch,
        )
        # Create global Transcriber (model loaded once, shared across streams)
        _moonshine_transcriber = Transcriber(
            model_path=_moonshine_model_path,
            model_arch=_moonshine_model_arch,
        )
        # Warmup with 1 s silence
        silence = np.zeros(16000, dtype=np.float32).tolist()
        _moonshine_transcriber.transcribe_without_streaming(silence, 16000)
        logger.info("[3/3] Moonshine STT ready")

        # Load Silero VAD (CPU, lightweight)
        logger.info("[3/3] Loading Silero VAD...")
        torch.set_num_threads(1)
        _silero_vad_model, _silero_vad_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        logger.info("[3/3] Silero VAD ready")
    except Exception as e:
        logger.error("[3/3] Moonshine/Silero failed: %s", e, exc_info=True)
        logger.warning("STT will be unavailable — voice input disabled")

    elapsed = time.monotonic() - start
    logger.info("=" * 60)
    logger.info("  Server ready in %.1fs", elapsed)
    logger.info("=" * 60)


def _pcm_int16_to_float32(pcm_bytes: bytes) -> list:
    """Convert raw PCM 16-bit signed mono bytes to a list of float32 in [-1, 1]."""
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    return (samples.astype(np.float32) / 32768.0).tolist()


def _wav_bytes_to_float32(wav_bytes: bytes) -> tuple:
    """Decode WAV bytes → (list[float], sample_rate)."""
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
        n_channels = wf.getnchannels()
        width = wf.getsampwidth()
    if width == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels == 2:
        samples = samples[::2]  # left channel
    return samples.tolist(), sr


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


class TranscribeRequest(BaseModel):
    audio_b64: str
    language: str = "en"


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
# NLP ENDPOINTS
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
    """Combined health status for NLP + Avatar + STT agents."""
    result = {"nlp_ready": False, "avatar_ready": False, "stt_ready": False}

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

    result["stt_ready"] = _moonshine_transcriber is not None
    result["vad_ready"] = _silero_vad_model is not None
    result["ready"] = (
        result.get("nlp_ready", False)
        and result.get("avatar_ready", False)
    )
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
    """
    agent = get_nlp_agent()
    avatar = _avatar_agent

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
                if chunk.text_delta:
                    chunk_count += 1
                    payload = json.dumps({"type": "token", "text": chunk.text_delta})
                    yield f"data: {payload}\n\n"

                if chunk.tts_result is not None:
                    audio_count += 1
                    wav_bytes = chunk.tts_result.to_wav_bytes()
                    b64 = base64.b64encode(wav_bytes).decode("utf-8")

                    payload = json.dumps({"type": "audio", "data": b64})
                    yield f"data: {payload}\n\n"

                    if avatar:
                        try:
                            visemes = avatar.extract_visemes(wav_bytes)
                            payload = json.dumps({"type": "visemes", "data": visemes})
                            yield f"data: {payload}\n\n"
                        except Exception as ve:
                            logger.warning("Viseme extraction failed: %s", ve)

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
    """Extract viseme timeline from base64-encoded WAV audio."""
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
# VOICE / STT ENDPOINTS — Moonshine + Silero VAD
# ============================================================================

@app.post("/api/transcribe")
async def transcribe_audio(req: TranscribeRequest):
    """
    Transcribe a base64-encoded WAV audio file to text using Moonshine.
    Used for the "record and send" flow.
    """
    transcriber = get_moonshine()

    logger.info("Transcribe request | b64_len=%d lang=%s", len(req.audio_b64), req.language)

    try:
        audio_bytes = base64.b64decode(req.audio_b64)
    except Exception as e:
        logger.error("Invalid base64 audio: %s", e)
        raise HTTPException(status_code=400, detail="Invalid base64 audio data")

    if len(audio_bytes) < 100:
        return {"text": "", "segments": [], "language": req.language}

    try:
        audio_float, sr = _wav_bytes_to_float32(audio_bytes)

        loop = asyncio.get_event_loop()
        transcript = await loop.run_in_executor(
            None,
            lambda: transcriber.transcribe_without_streaming(audio_float, sr),
        )

        segments = []
        text_parts = []
        for line in transcript.lines:
            text = line.text.strip()
            if text:
                segments.append({
                    "start": round(line.start_time, 2),
                    "end": round(line.start_time + line.duration, 2),
                    "text": text,
                })
                text_parts.append(text)

        full_text = " ".join(text_parts)
        logger.info("Transcription complete | text_len=%d segments=%d", len(full_text), len(segments))
        return {"text": full_text, "segments": segments, "language": req.language}
    except Exception as e:
        logger.error("Transcription failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")


@app.websocket("/ws/transcribe")
async def ws_transcribe(websocket: WebSocket):
    """
    WebSocket for real-time microphone streaming → live transcript.

    Uses Silero VAD for speech detection / turn-taking and Moonshine Voice
    for streaming speech-to-text.

    Protocol:
        Client → Server:
            - Binary frames: raw PCM audio (16-bit, 16kHz, mono)
            - JSON text: {"command": "stop"} to end session

        Server → Client:
            - JSON text: {"type": "partial", "text": "..."} — interim result
            - JSON text: {"type": "final", "text": "..."} — stable result
            - JSON text: {"type": "vad", "speech": true/false} — voice activity
            - JSON text: {"type": "error", "message": "..."} — error
    """
    await websocket.accept()
    logger.info("[WS] Client connected for real-time STT")

    transcriber = _moonshine_transcriber
    vad_model = _silero_vad_model
    if transcriber is None:
        await websocket.send_json({"type": "error", "message": "STT not available"})
        await websocket.close()
        return

    # ---- Per-session state ----
    import torch
    from silero_vad.utils_vad import VADIterator

    vad_iterator = VADIterator(
        vad_model,
        sampling_rate=16000,
        threshold=0.5,
        min_silence_duration_ms=800,   # end-of-turn silence
        speech_pad_ms=30,
    ) if vad_model is not None else None

    # Collect text from Moonshine events
    from moonshine_voice import TranscriptEventListener

    class _WSListener(TranscriptEventListener):
        def __init__(self):
            self.partial = ""
            self.completed_parts: list[str] = []
            self.has_new_partial = False
            self.has_new_completion = False

        def on_line_text_changed(self, event):
            self.partial = event.line.text.strip()
            self.has_new_partial = True

        def on_line_completed(self, event):
            text = event.line.text.strip()
            if text:
                self.completed_parts.append(text)
            self.partial = ""
            self.has_new_completion = True

        def full_text(self) -> str:
            parts = list(self.completed_parts)
            if self.partial:
                parts.append(self.partial)
            return " ".join(parts)

        def reset(self):
            self.partial = ""
            self.completed_parts.clear()
            self.has_new_partial = False
            self.has_new_completion = False

    listener = _WSListener()

    # Use a Moonshine stream for this connection so the global transcriber
    # stays available for the HTTP endpoint.
    stream = transcriber.create_stream()
    stream.add_listener(listener)
    stream.start()

    sample_rate = 16000
    is_speaking = False

    try:
        while True:
            try:
                message = await asyncio.wait_for(websocket.receive(), timeout=30.0)
            except asyncio.TimeoutError:
                try:
                    await websocket.send_json({"type": "keepalive"})
                except Exception:
                    break
                continue

            # Check for disconnect message from Starlette
            if message.get("type") == "websocket.disconnect":
                logger.info("[WS] Received disconnect frame")
                break

            if "bytes" in message:
                pcm_bytes = message["bytes"]
                if len(pcm_bytes) < 2:
                    continue

                audio_float = _pcm_int16_to_float32(pcm_bytes)
                audio_tensor = torch.tensor(audio_float, dtype=torch.float32)

                # --- Silero VAD: real-time speech detection ---
                if vad_iterator is not None:
                    # Process in 512-sample windows (required by Silero)
                    for i in range(0, len(audio_tensor), 512):
                        chunk = audio_tensor[i : i + 512]
                        if len(chunk) < 512:
                            break
                        speech_dict = vad_iterator(chunk, return_seconds=False)
                        if speech_dict is not None:
                            if "start" in speech_dict:
                                if not is_speaking:
                                    is_speaking = True
                                    try:
                                        await websocket.send_json({
                                            "type": "vad",
                                            "speech": True,
                                        })
                                    except Exception:
                                        pass
                            elif "end" in speech_dict:
                                if is_speaking:
                                    is_speaking = False
                                    try:
                                        await websocket.send_json({
                                            "type": "vad",
                                            "speech": False,
                                        })
                                    except Exception:
                                        pass

                # --- Moonshine: streaming STT ---
                stream.add_audio(audio_float, sample_rate)

                # Push partial text updates to client
                if listener.has_new_partial:
                    listener.has_new_partial = False
                    text = listener.full_text()
                    if text:
                        try:
                            await websocket.send_json({
                                "type": "partial",
                                "text": text,
                            })
                        except Exception:
                            pass

                # Push completed-line updates
                if listener.has_new_completion:
                    listener.has_new_completion = False
                    text = listener.full_text()
                    if text:
                        try:
                            await websocket.send_json({
                                "type": "partial",
                                "text": text,
                            })
                        except Exception:
                            pass

            elif "text" in message:
                try:
                    cmd = json.loads(message["text"])
                except (json.JSONDecodeError, TypeError):
                    continue

                if cmd.get("command") == "stop":
                    # Finalize: stop triggers LineCompleted for any active line
                    stream.stop()
                    final_text = listener.full_text()
                    try:
                        await websocket.send_json({
                            "type": "final",
                            "text": final_text,
                        })
                    except Exception:
                        pass
                    # Prepare for next utterance
                    listener.reset()
                    if vad_iterator is not None:
                        vad_iterator.reset_states()
                    is_speaking = False
                    stream.start()

                elif cmd.get("command") == "reset":
                    stream.stop()
                    listener.reset()
                    if vad_iterator is not None:
                        vad_iterator.reset_states()
                    is_speaking = False
                    stream.start()
                    try:
                        await websocket.send_json({"type": "reset", "text": ""})
                    except Exception:
                        pass

    except WebSocketDisconnect:
        logger.info("[WS] Client disconnected")
    except RuntimeError as e:
        # Starlette raises RuntimeError when receiving after disconnect
        if "disconnect" in str(e).lower():
            logger.info("[WS] Client already disconnected")
        else:
            logger.error("[WS] Runtime error: %s", e)
    except Exception as e:
        logger.error("[WS] Error: %s", e, exc_info=True)
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        try:
            stream.stop()
        except Exception:
            pass


# ============================================================================
# Health check
# ============================================================================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "timestamp": time.time(),
        "stt_available": _moonshine_transcriber is not None,
        "vad_available": _silero_vad_model is not None,
    }
