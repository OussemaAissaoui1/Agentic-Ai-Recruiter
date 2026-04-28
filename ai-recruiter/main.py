"""AI Recruiter — Unified FastAPI Application.

Composes every BaseAgent in the platform under a single ASGI app:

    /                          static shell (apps/shell)
    /screening/*               apps/static/screening (matching frontend)
    /interview/*               apps/static/interview (NLP + vision-overlay frontend)
    /behavior/*                apps/static/behavior  (vision dashboard)

    /api/matching/*            matching agent
    /api/nlp/*                 nlp agent
    /api/vision/*              vision agent (HTTP)
    /ws/vision                 vision agent (WebSocket)
    /api/avatar/*              avatar agent
    /api/scoring/*             scoring agent
    /api/voice/*               voice agent (stub)
    /api/orchestrator/*        orchestrator agent

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

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from agents.avatar import AvatarAgentAdapter
from agents.matching import MatchingAgent
from agents.nlp.agent_adapter import NLPAgentAdapter
from agents.orchestrator import OrchestratorAgent
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
SHELL_DIR = ROOT / "apps" / "shell"
STATIC_ROOT = ROOT / "apps" / "static"

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
# Static frontends
# ---------------------------------------------------------------------------
def _safe_mount(path: str, dir_: Path, name: str) -> None:
    if dir_.exists() and any(dir_.iterdir()):
        app.mount(path, StaticFiles(directory=dir_, html=True), name=name)
        log.info("mounted static: %s -> %s", path, dir_)
    else:
        log.info("static dir empty (%s) — frontend not built yet", dir_)


_safe_mount("/screening", STATIC_ROOT / "screening", "screening")
_safe_mount("/interview", STATIC_ROOT / "interview", "interview")
_safe_mount("/behavior", STATIC_ROOT / "behavior", "behavior")


@app.get("/")
async def shell_root():
    """Top-level shell linking to the three sub-apps."""
    idx = SHELL_DIR / "index.html"
    if idx.exists():
        return FileResponse(idx, media_type="text/html")
    return JSONResponse(
        {"message": "AI Recruiter API",
         "routes": ["/screening/", "/interview/", "/behavior/", "/docs"]},
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
