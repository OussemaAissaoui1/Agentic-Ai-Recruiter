"""HireFlow — Unified FastAPI Application.

Composes every BaseAgent in the platform under a single ASGI app:

    /                          HireFlow SPA (apps/hireflow/dist)
    /assets/*                  HireFlow hashed JS/CSS bundles
    /<spa-route>               served via SPA fallback to index.html

    /api/matching/*            matching agent (CV parsing + ranking)
    /api/nlp/*                 nlp agent (interview turns + TTS)
    /api/vision/*              vision agent (HTTP)
    /ws/vision                 vision agent (WebSocket)
    /api/avatar/*              avatar agent
    /api/scoring/*             scoring agent
    /api/voice/*               voice agent (stub)
    /api/orchestrator/*        orchestrator agent
    /api/recruit/*             recruit agent (jobs / applications / notifications / JD-gen)

    /api/health                aggregated per-agent readiness probe
    /.well-known/agents/<name> agent card discovery
    /docs                      OpenAPI

Heavy GPU loads stay lazy in the agents themselves; this entrypoint never
blocks startup on a model load. Use POST /api/nlp/warmup once after boot.

Run:
    cd ai-recruiter
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import asyncio
import json

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agents.avatar import AvatarAgentAdapter
from agents.matching import MatchingAgent
from agents.nlp.agent_adapter import NLPAgentAdapter
from agents.orchestrator import OrchestratorAgent
from agents.recruit import RecruitAgent
from agents.scoring import ScoringAgent
from agents.vision import VisionAgent
from agents.voice import VoiceAgent
from configs import load_runtime, resolve
from core.contracts import HealthStatus
from core.observability import configure_logging
from core.runtime import AgentRegistry, build_lifespan

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_cfg = load_runtime()
configure_logging(level=_cfg["server"]["log_level"])
log = logging.getLogger("main")

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Build registry
# ---------------------------------------------------------------------------
registry = AgentRegistry()
registry.register(MatchingAgent())
registry.register(VisionAgent())
registry.register(NLPAgentAdapter())
registry.register(AvatarAgentAdapter())
registry.register(ScoringAgent())
registry.register(VoiceAgent())
registry.register(OrchestratorAgent())
registry.register(RecruitAgent())

lifespan = build_lifespan(
    registry,
    startup_order=_cfg.get("agents", {}).get("startup_order"),
)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AI Recruiter",
    version="0.2.0",
    description="Unified pre-screening + multimodal interview + behavioral analysis platform.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cfg["server"].get("cors_origins", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Mount agent routers
# ---------------------------------------------------------------------------
for agent in registry.all():
    app.include_router(agent.router, prefix=f"/api/{agent.name}")
    log.info("mounted: /api/%s -> %s v%s", agent.name, type(agent).__name__, agent.version)

# Vision agent's WebSocket lives at /ws/vision (root-level, not /api/vision).
vision = registry.get("vision")
if hasattr(vision, "ws_router"):
    app.include_router(vision.ws_router, prefix="/ws/vision")
    log.info("mounted: /ws/vision -> vision agent WebSocket")


# ---------------------------------------------------------------------------
# Aggregate health + discovery
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def aggregate_health() -> Dict[str, Any]:
    """Aggregate per-agent readiness. Always returns 200 — clients inspect state."""
    statuses = await registry.health_all()
    overall = "ready"
    if any(s.state == "error" for s in statuses.values()):
        overall = "error"
    elif any(s.state == "loading" for s in statuses.values()):
        overall = "loading"
    elif any(s.state == "fallback" for s in statuses.values()):
        overall = "fallback"
    return {
        "version": "0.2.0",
        "overall": overall,
        "agents": {n: s.model_dump() for n, s in statuses.items()},
    }


@app.get("/.well-known/agents/{name}")
async def agent_card(name: str):
    cards = registry.cards()
    if name not in cards:
        raise HTTPException(404, f"agent not found: {name}")
    return cards[name].model_dump()


@app.get("/.well-known/agents")
async def list_agent_cards():
    return {n: c.model_dump() for n, c in registry.cards().items()}


# ---------------------------------------------------------------------------
# Legacy /api/* aliases for the prebuilt interview React bundle
#
# The static React bundle in apps/static/interview (built from
# agents/avatar/demo) calls /api/{status,session,stream,chat,tts,warmup} at
# the root. Re-mount the NLP router under /api so those paths resolve without
# rebuilding the bundle. Registered AFTER /api/health so the aggregate health
# endpoint above keeps priority over NLP's /health.
# ---------------------------------------------------------------------------
_nlp_agent = registry.get("nlp")
_avatar_agent_for_status = registry.get("avatar")  # used by combined /api/status below

# Combined /api/status — must be registered BEFORE the NLP router include below
# so it wins route resolution. The interview React bundle (built from
# agents/voice/demo — the canonical interview UI) expects {ready, nlp_ready,
# avatar_ready, stt_ready, vad_ready, engine_ready, tts_ready, gpu_free_gb,
# gpu_total_gb, glb_loaded, viseme_engine_ready} and treats `ready=false` as
# "server offline". The NLP router's /status endpoint only exposes the NLP
# half, so we wrap NLP + avatar + STT here.
if _nlp_agent is not None or _avatar_agent_for_status is not None:
    from agents.voice import stt as _stt_mod  # safe — does not trigger lazy load

    @app.get("/api/status")
    async def combined_status():
        result = {
            "nlp_ready": False,
            "avatar_ready": False,
            "stt_ready": _stt_mod.is_loaded(),
            "vad_ready": _stt_mod.is_loaded(),
        }
        if _nlp_agent is not None and _nlp_agent._inner is not None:
            try:
                s = _nlp_agent._inner.get_status()
                result.update({
                    "nlp_ready": True,
                    "engine_ready": s.get("engine_ready", False),
                    "tts_ready": s.get("tts_ready", False),
                    "scorer_ready": s.get("scorer_ready", False),
                    "gpu_free_gb": round(s.get("gpu_free_gb") or 0, 1),
                    "gpu_total_gb": round(s.get("gpu_total_gb") or 0, 1),
                    "active_sessions": s.get("active_sessions", 0),
                })
            except Exception as e:
                result["nlp_error"] = f"{type(e).__name__}: {e}"
        if _avatar_agent_for_status is not None and getattr(
            _avatar_agent_for_status, "_inner", None
        ) is not None:
            try:
                a = _avatar_agent_for_status._inner.get_status()
                result.update({
                    "avatar_ready": a.get("ready", False),
                    "glb_loaded": a.get("glb_loaded", False),
                    "glb_size_mb": a.get("glb_file_size_mb"),
                    "viseme_engine_ready": a.get("viseme_engine_ready", False),
                })
            except Exception as e:
                result["avatar_error"] = f"{type(e).__name__}: {e}"
        result["ready"] = bool(result.get("nlp_ready")) and bool(result.get("avatar_ready"))
        return result

if _nlp_agent is not None:
    app.include_router(_nlp_agent.router, prefix="/api")
    log.info("mounted legacy aliases: /api/{status,session,stream,chat,tts,warmup}")


# ---------------------------------------------------------------------------
# Avatar bundle aliases.
#
# The interview React bundle (built from agents/avatar/demo) calls:
#   GET  /avatar/model.glb
#   GET  /avatar/status
#   POST /avatar/visemes
#
# These don't exist as bare routes — the unified avatar router lives under
# /api/avatar/{glb,health,visemes}. Add thin aliases at /avatar/* so the
# bundle works as built without a rebuild.
# ---------------------------------------------------------------------------
_avatar_agent = registry.get("avatar")
if _avatar_agent is not None:
    from fastapi import Body

    @app.get("/avatar/model.glb")
    async def avatar_model_glb():
        try:
            p = _avatar_agent._inner.get_glb_path()
            return FileResponse(
                str(p),
                media_type="model/gltf-binary",
                filename="avatar.glb",
                headers={"Cache-Control": "public, max-age=86400"},
            )
        except Exception as e:
            raise HTTPException(503, f"avatar unavailable: {e}")

    @app.get("/avatar/status")
    async def avatar_status_alias():
        try:
            return _avatar_agent._inner.get_status()
        except Exception as e:
            raise HTTPException(503, f"avatar unavailable: {e}")

    @app.post("/avatar/visemes")
    async def avatar_visemes_alias(payload: dict = Body(...)):
        import base64
        b64 = payload.get("audio_b64") or payload.get("wav_b64")
        if not b64:
            raise HTTPException(400, "missing audio_b64")
        try:
            wav = base64.b64decode(b64)
            return {"visemes": _avatar_agent._inner.extract_visemes(wav)}
        except Exception as e:
            raise HTTPException(500, f"viseme extraction failed: {e}")

    log.info("mounted avatar aliases: /avatar/{model.glb,status,visemes}")


# ---------------------------------------------------------------------------
# Voice / STT — Moonshine + Silero VAD
#
# Endpoints expected by the voice/demo bundle:
#   POST /api/transcribe        — upload WAV (base64) → {text, segments, language}
#   WS   /ws/transcribe         — stream raw 16-bit/16kHz PCM, receive
#                                 {type: "partial"|"final"|"vad"|"reset"|"error"|"keepalive", ...}
#
# The Moonshine model + Silero VAD are loaded lazily on first call (≈5–10 s)
# so app startup stays fast. See agents/voice/stt.py for the loader.
# ---------------------------------------------------------------------------
class _TranscribeRequest(BaseModel):
    audio_b64: str
    language: str = "en"


@app.post("/api/transcribe")
async def transcribe_audio(req: _TranscribeRequest):
    """Whole-clip transcription. Used by record-and-send flows."""
    from agents.voice import stt as stt_mod
    try:
        backend = stt_mod.get_stt()
    except Exception as e:
        log.exception("STT init failed")
        raise HTTPException(503, f"STT unavailable: {type(e).__name__}: {e}")
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(
            None, lambda: backend.transcribe_wav(req.audio_b64, req.language)
        )
    except Exception as e:
        log.exception("Transcription failed")
        raise HTTPException(500, f"Transcription failed: {e}")


@app.websocket("/ws/transcribe")
async def ws_transcribe(websocket: WebSocket):
    """Streaming STT with VAD-based turn-taking.

    Client → Server:
        Binary frames: raw PCM (int16, 16 kHz, mono)
        JSON  text:    {"command": "stop"}  → finalize current utterance
                       {"command": "reset"} → drop in-progress text

    Server → Client (all JSON):
        {"type": "partial",   "text": "..."}
        {"type": "final",     "text": "..."}
        {"type": "vad",       "speech": true|false}
        {"type": "reset",     "text": ""}
        {"type": "keepalive"}
        {"type": "error",     "message": "..."}
    """
    await websocket.accept()
    log.info("[ws/transcribe] client connected")

    try:
        from agents.voice import stt as stt_mod
        backend = await asyncio.get_event_loop().run_in_executor(None, stt_mod.get_stt)
    except Exception as e:
        log.exception("STT init failed for WS client")
        try:
            await websocket.send_json({"type": "error", "message": f"STT unavailable: {e}"})
        finally:
            await websocket.close()
        return

    import torch
    from agents.voice.stt import pcm_int16_to_float32

    listener_cls_base = backend.transcript_event_listener

    class _Listener(listener_cls_base):
        def __init__(self):
            self.partial = ""
            self.completed: list[str] = []
            self.has_new_partial = False
            self.has_new_completion = False

        def on_line_text_changed(self, event):
            self.partial = event.line.text.strip()
            self.has_new_partial = True

        def on_line_completed(self, event):
            t = event.line.text.strip()
            if t:
                self.completed.append(t)
            self.partial = ""
            self.has_new_completion = True

        def full_text(self) -> str:
            parts = list(self.completed)
            if self.partial:
                parts.append(self.partial)
            return " ".join(parts)

        def reset(self):
            self.partial = ""
            self.completed.clear()
            self.has_new_partial = False
            self.has_new_completion = False

    listener = _Listener()
    stream = backend.transcriber.create_stream()
    stream.add_listener(listener)
    stream.start()

    vad_iter = backend.create_vad_iterator()
    is_speaking = False
    SAMPLE_RATE = 16000

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

            if message.get("type") == "websocket.disconnect":
                break

            if "bytes" in message:
                pcm_bytes = message["bytes"]
                if len(pcm_bytes) < 2:
                    continue
                audio_float = pcm_int16_to_float32(pcm_bytes)

                # Silero VAD — emits "speech true/false" so the UI can show a
                # "talking" indicator. Required 512-sample chunks.
                if vad_iter is not None:
                    audio_tensor = torch.tensor(audio_float, dtype=torch.float32)
                    for i in range(0, len(audio_tensor), 512):
                        chunk = audio_tensor[i:i + 512]
                        if len(chunk) < 512:
                            break
                        speech = vad_iter(chunk, return_seconds=False)
                        if speech is not None:
                            if "start" in speech and not is_speaking:
                                is_speaking = True
                                try:
                                    await websocket.send_json({"type": "vad", "speech": True})
                                except Exception:
                                    pass
                            elif "end" in speech and is_speaking:
                                is_speaking = False
                                try:
                                    await websocket.send_json({"type": "vad", "speech": False})
                                except Exception:
                                    pass

                # Feed Moonshine
                stream.add_audio(audio_float.tolist(), SAMPLE_RATE)

                if listener.has_new_partial:
                    listener.has_new_partial = False
                    txt = listener.full_text()
                    if txt:
                        try:
                            await websocket.send_json({"type": "partial", "text": txt})
                        except Exception:
                            pass

                if listener.has_new_completion:
                    listener.has_new_completion = False
                    txt = listener.full_text()
                    if txt:
                        try:
                            await websocket.send_json({"type": "partial", "text": txt})
                        except Exception:
                            pass

            elif "text" in message:
                try:
                    cmd = json.loads(message["text"])
                except (json.JSONDecodeError, TypeError):
                    continue
                action = cmd.get("command")
                if action == "stop":
                    stream.stop()
                    final_text = listener.full_text()
                    try:
                        await websocket.send_json({"type": "final", "text": final_text})
                    except Exception:
                        pass
                    listener.reset()
                    if vad_iter is not None:
                        vad_iter.reset_states()
                    is_speaking = False
                    stream.start()
                elif action == "reset":
                    stream.stop()
                    listener.reset()
                    if vad_iter is not None:
                        vad_iter.reset_states()
                    is_speaking = False
                    stream.start()
                    try:
                        await websocket.send_json({"type": "reset", "text": ""})
                    except Exception:
                        pass

    except WebSocketDisconnect:
        log.info("[ws/transcribe] client disconnected")
    except RuntimeError as e:
        # Starlette raises RuntimeError when receiving after disconnect
        if "disconnect" in str(e).lower():
            log.info("[ws/transcribe] client already disconnected")
        else:
            log.exception("[ws/transcribe] runtime error: %s", e)
    except Exception as e:
        log.exception("[ws/transcribe] error")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        try:
            stream.stop()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# HireFlow SPA — sole user-facing frontend
# ---------------------------------------------------------------------------
HIREFLOW_DIR = ROOT / "apps" / "hireflow" / "dist"

if (HIREFLOW_DIR / "assets").exists():
    app.mount(
        "/assets",
        StaticFiles(directory=HIREFLOW_DIR / "assets"),
        name="hireflow-assets",
    )
    log.info("mounted hireflow assets -> %s", HIREFLOW_DIR / "assets")
else:
    log.warning(
        "hireflow build missing (%s) — run `npm run build` in apps/hireflow",
        HIREFLOW_DIR,
    )


# Reserved server prefixes — anything else is owned by the SPA.
_SERVER_PREFIXES = (
    "api/", "ws/", "avatar/", "files/", ".well-known/",
    "docs", "redoc", "openapi.json",
)


@app.get("/{full_path:path}", include_in_schema=False)
async def hireflow_spa(full_path: str):
    """SPA fallback: serve hashed files from /apps/hireflow/dist when present,
    otherwise return index.html so client-side routing handles the URL."""
    if full_path.startswith(_SERVER_PREFIXES):
        raise HTTPException(404)
    candidate = HIREFLOW_DIR / full_path if full_path else None
    if candidate and candidate.is_file():
        return FileResponse(candidate)
    idx = HIREFLOW_DIR / "index.html"
    if idx.exists():
        return FileResponse(idx, media_type="text/html")
    return JSONResponse(
        {"message": "HireFlow API up — frontend build missing",
         "routes": ["/api/recruit/jobs", "/api/matching/health", "/docs"]},
        status_code=200,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def run() -> None:
    import uvicorn
    uvicorn.run(
        "main:app",
        host=_cfg["server"]["host"],
        port=int(_cfg["server"]["port"]),
        log_level=_cfg["server"]["log_level"].lower(),
    )


if __name__ == "__main__":
    run()
