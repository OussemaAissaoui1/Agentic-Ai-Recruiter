"""BaseAgent adapter around the existing AvatarAgent (Three.js-friendly server)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel

from core.contracts import (
    A2ATask,
    A2ATaskResult,
    AgentCard,
    BaseAgent,
    Capability,
    HealthState,
    HealthStatus,
)
from core.observability import bind_agent_logger

log = bind_agent_logger("avatar")
_AGENT_DIR = Path(__file__).resolve().parent
_CARD_PATH = _AGENT_DIR / "agent_card.json"


class AvatarAgentAdapter(BaseAgent):
    name = "avatar"
    version = "0.2.0"

    def __init__(self) -> None:
        from .agent import AvatarAgent, AvatarConfig
        self._inner = AvatarAgent(config=AvatarConfig())
        self._router: Optional[APIRouter] = None
        self._init_error: Optional[str] = None

    async def startup(self, app_state: Any) -> None:
        try:
            await self._inner.initialize()
        except Exception as e:
            self._init_error = f"{type(e).__name__}: {e}"

    async def shutdown(self) -> None:
        pass

    @property
    def router(self) -> APIRouter:
        if self._router is None:
            self._router = self._build_router()
        return self._router

    def card(self) -> AgentCard:
        if _CARD_PATH.exists():
            return AgentCard.model_validate(json.loads(_CARD_PATH.read_text()))
        return AgentCard(
            name=self.name, version=self.version,
            description="3D recruiter avatar (GLB) + viseme extraction for lip sync.",
            routes=["/api/avatar/glb", "/api/avatar/visemes", "/api/avatar/health"],
            capabilities=[
                Capability(name="get_glb",
                           description="Return the GLB avatar model bytes."),
                Capability(name="extract_visemes",
                           description="Extract viseme timeline from WAV bytes."),
            ],
        )

    async def health(self) -> HealthStatus:
        if self._init_error:
            return HealthStatus(agent=self.name, state=HealthState.ERROR,
                                version=self.version, error=self._init_error)
        s = self._inner.get_status()
        state = HealthState.READY if s["ready"] else HealthState.LOADING
        return HealthStatus(agent=self.name, state=state, version=self.version, detail=s)

    async def handle_task(self, task: A2ATask) -> A2ATaskResult:
        try:
            if task.capability == "get_glb":
                p = self._inner.get_glb_path()
                return A2ATaskResult(task_id=task.task_id, success=True,
                                     result={"path": str(p)})
            if task.capability == "extract_visemes":
                import base64
                wav = base64.b64decode(task.payload["wav_b64"])
                timeline = self._inner.extract_visemes(wav)
                return A2ATaskResult(task_id=task.task_id, success=True,
                                     result={"timeline": timeline})
            return A2ATaskResult(task_id=task.task_id, success=False,
                                 error=f"unknown capability: {task.capability}")
        except Exception as e:
            return A2ATaskResult(task_id=task.task_id, success=False,
                                 error=f"{type(e).__name__}: {e}")

    # ------------------------------------------------------------------ router
    def _build_router(self) -> APIRouter:
        router = APIRouter(tags=["avatar"])
        inner = self._inner

        class VisemeRequest(BaseModel):
            wav_b64: str

        @router.get("/health")
        async def health():
            return (await self.health()).model_dump()

        @router.get("/glb")
        async def get_glb():
            try:
                p = inner.get_glb_path()
                return Response(content=p.read_bytes(),
                                media_type="model/gltf-binary",
                                headers={"Cache-Control": "public, max-age=86400"})
            except Exception as e:
                raise HTTPException(503, f"avatar unavailable: {e}")

        @router.post("/visemes")
        async def extract_visemes(req: VisemeRequest):
            import base64
            try:
                wav = base64.b64decode(req.wav_b64)
                return {"timeline": inner.extract_visemes(wav)}
            except Exception as e:
                raise HTTPException(500, f"viseme extraction failed: {e}")

        return router
