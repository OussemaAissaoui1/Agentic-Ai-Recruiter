"""
FastAPI Backend — AI Recruiter Voice Interview Demo

Extends the Avatar demo server with real-time Speech-to-Text (STT)
using Faster Whisper. Supports both WebSocket-based microphone streaming
and HTTP POST for audio file transcription.

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
import struct
import sys
import tempfile
import time
import traceback
import uuid
import wave
from pathlib import Path
from typing import Optional

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
    description="NLP Interview Pipeline + 3D Avatar + Voice STT (Faster Whisper)",
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
_whisper_model = None


def get_nlp_agent():
    if _nlp_agent is None:
        raise HTTPException(status_code=503, detail="NLP agent not ready yet")
    return _nlp_agent


def get_avatar_agent():
    if _avatar_agent is None:
        raise HTTPException(status_code=503, detail="Avatar agent not ready yet")
    return _avatar_agent


def get_whisper():
    if _whisper_model is None:
        raise HTTPException(status_code=503, detail="Whisper STT not ready yet")
    return _whisper_model


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    global _nlp_agent, _avatar_agent, _whisper_model

    logger.info("=" * 60)
    logger.info("  AI Recruiter Voice Server — Starting")
    logger.info("=" * 60)
    start = time.monotonic()

    # --- Load NLP Agent ---
    logger.info("[1/3] Loading NLP agent...")
    try:
        from agent import NLPAgent
        _nlp_agent = NLPAgent(
            model_path="oussema2021/fintuned_v3_AiRecruter",
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

    # --- Load Faster Whisper STT ---
    logger.info("[3/3] Loading Faster Whisper STT...")
    try:
        from faster_whisper import WhisperModel

        # Use small model for balance of speed/accuracy, GPU if available
        whisper_model_size = os.environ.get("WHISPER_MODEL", "small")
        device = "cuda" if _check_gpu() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"

        logger.info(
            "[3/3] Loading Whisper model=%s device=%s compute=%s",
            whisper_model_size, device, compute_type,
        )
        _whisper_model = WhisperModel(
            whisper_model_size,
            device=device,
            compute_type=compute_type,
        )
        # Warmup with a short silence
        _warmup_whisper(_whisper_model)
        logger.info("[3/3] Faster Whisper STT ready")
    except Exception as e:
        logger.error("[3/3] Faster Whisper failed: %s", e, exc_info=True)
        logger.warning("STT will be unavailable — voice input disabled")

    elapsed = time.monotonic() - start
    logger.info("=" * 60)
    logger.info("  Server ready in %.1fs", elapsed)
    logger.info("=" * 60)


def _check_gpu() -> bool:
    """Check if CUDA GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        pass
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


def _warmup_whisper(model):
    """Warmup Whisper with a short silence WAV."""
    try:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * 16000)  # 1 second silence
        buf.seek(0)
        # Write to a temp file since faster-whisper needs a file path
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            tmp.write(buf.read())
            tmp.flush()
            segments, _ = model.transcribe(tmp.name, beam_size=1)
            list(segments)  # consume iterator
        logger.info("[Whisper] Warmup complete")
    except Exception as e:
        logger.warning("[Whisper] Warmup failed (non-fatal): %s", e)


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

    result["stt_ready"] = _whisper_model is not None
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
# VOICE / STT ENDPOINTS — Faster Whisper
# ============================================================================

@app.post("/api/transcribe")
async def transcribe_audio(req: TranscribeRequest):
    """
    Transcribe a base64-encoded audio file (WAV/WebM/OGG) to text.
    Used for the "record and send" flow.
    """
    model = get_whisper()

    logger.info("Transcribe request | b64_len=%d lang=%s", len(req.audio_b64), req.language)

    try:
        audio_bytes = base64.b64decode(req.audio_b64)
    except Exception as e:
        logger.error("Invalid base64 audio: %s", e)
        raise HTTPException(status_code=400, detail="Invalid base64 audio data")

    if len(audio_bytes) < 100:
        return {"text": "", "segments": [], "language": req.language}

    try:
        text, segments = await _transcribe_bytes(model, audio_bytes, req.language)
        logger.info("Transcription complete | text_len=%d segments=%d", len(text), len(segments))
        return {"text": text, "segments": segments, "language": req.language}
    except Exception as e:
        logger.error("Transcription failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")


@app.websocket("/ws/transcribe")
async def ws_transcribe(websocket: WebSocket):
    """
    WebSocket for real-time microphone streaming → live transcript.

    Protocol:
        Client → Server:
            - Binary frames: raw PCM audio (16-bit, 16kHz, mono)
            - JSON text: {"command": "stop"} to end session

        Server → Client:
            - JSON text: {"type": "partial", "text": "..."} — interim result
            - JSON text: {"type": "final", "text": "..."} — stable result
            - JSON text: {"type": "error", "message": "..."} — error
    """
    await websocket.accept()
    logger.info("[WS] Client connected for real-time STT")

    model = _whisper_model
    if model is None:
        await websocket.send_json({"type": "error", "message": "STT not available"})
        await websocket.close()
        return

    audio_buffer = bytearray()
    sample_rate = 16000
    # Process audio every ~2.5 seconds of NEW audio (not cumulative)
    chunk_duration_sec = 2.5
    chunk_threshold = int(sample_rate * 2 * chunk_duration_sec)  # 16-bit = 2 bytes/sample
    accumulated_text_parts = []  # finalized transcript segments
    pending_buffer = bytearray()  # only NEW audio since last transcription
    is_transcribing = False  # guard against overlapping transcriptions

    disconnected = False

    try:
        while not disconnected:
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
                pending_buffer.extend(message["bytes"])

                # Only transcribe NEW audio when enough accumulates
                if len(pending_buffer) >= chunk_threshold and not is_transcribing:
                    is_transcribing = True
                    try:
                        chunk_text = await _transcribe_pcm_buffer(
                            model, bytes(pending_buffer), sample_rate
                        )
                        if chunk_text:
                            accumulated_text_parts.append(chunk_text)
                        pending_buffer.clear()
                        full_text = " ".join(accumulated_text_parts)
                        if full_text:
                            await websocket.send_json({
                                "type": "partial",
                                "text": full_text,
                            })
                    except Exception as te:
                        logger.warning("[WS] Transcription error: %s", te)
                    finally:
                        is_transcribing = False

            elif "text" in message:
                try:
                    cmd = json.loads(message["text"])
                except (json.JSONDecodeError, TypeError):
                    continue

                if cmd.get("command") == "stop":
                    # Final transcription of remaining pending audio
                    if len(pending_buffer) > 1000:
                        chunk_text = await _transcribe_pcm_buffer(
                            model, bytes(pending_buffer), sample_rate
                        )
                        if chunk_text:
                            accumulated_text_parts.append(chunk_text)
                    pending_buffer.clear()
                    full_text = " ".join(accumulated_text_parts)
                    try:
                        await websocket.send_json({
                            "type": "final",
                            "text": full_text,
                        })
                    except Exception:
                        pass
                    accumulated_text_parts.clear()

                elif cmd.get("command") == "reset":
                    pending_buffer.clear()
                    accumulated_text_parts.clear()
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


async def _transcribe_bytes(model, audio_bytes: bytes, language: str = "en"):
    """Transcribe audio bytes (any format) using Faster Whisper."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()

        loop = asyncio.get_event_loop()
        segments_raw, info = await loop.run_in_executor(
            None,
            lambda: model.transcribe(
                tmp.name,
                beam_size=5,
                language=language,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            ),
        )
        segments = []
        full_text_parts = []
        for seg in segments_raw:
            segments.append({
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text.strip(),
            })
            full_text_parts.append(seg.text.strip())

        return " ".join(full_text_parts), segments


async def _transcribe_pcm_buffer(model, pcm_bytes: bytes, sample_rate: int = 16000):
    """Transcribe raw PCM 16-bit mono buffer using Faster Whisper."""
    if len(pcm_bytes) < 1000:
        return ""

    # Convert PCM to WAV in memory
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    wav_buf.seek(0)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp.write(wav_buf.read())
        tmp.flush()

        loop = asyncio.get_event_loop()
        segments_raw, info = await loop.run_in_executor(
            None,
            lambda: model.transcribe(
                tmp.name,
                beam_size=1,
                language="en",
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=300),
            ),
        )
        parts = []
        for seg in segments_raw:
            text = seg.text.strip()
            if text:
                parts.append(text)

        return " ".join(parts)


# ============================================================================
# Health check
# ============================================================================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "timestamp": time.time(),
        "stt_available": _whisper_model is not None,
    }
