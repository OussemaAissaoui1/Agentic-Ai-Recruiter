"""Voice Agent — ASR + prosody (stub).

Note: behavioral / vocal-stress analysis lives in the Vision Agent
(`agents/vision/`), which owns the multimodal stress-detection pipeline
end-to-end. This agent is reserved for ASR + prosody features that the
NLP agent does not already cover (e.g., faster-whisper transcription).

It conforms to the BaseAgent contract so it can be registered, surfaced in
`/api/health`, and dispatched via the orchestrator — but it returns 501 on
all task capabilities until ASR is wired in.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter

from core.contracts import (
    A2ATask, A2ATaskResult, AgentCard, BaseAgent, Capability,
    HealthState, HealthStatus,
)


class VoiceAgent(BaseAgent):
    name = "voice"
    version = "0.0.1"

    def __init__(self) -> None:
        self._router: Optional[APIRouter] = None

    async def startup(self, app_state: Any) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    @property
    def router(self) -> APIRouter:
        if self._router is None:
            r = APIRouter(tags=["voice"])

            @r.get("/health")
            async def health():
                return (await self.health()).model_dump()

            self._router = r
        return self._router

    def card(self) -> AgentCard:
        return AgentCard(
            name=self.name, version=self.version,
            description="ASR + prosody (planned). Behavioral vocal-stress lives in the Vision Agent.",
            routes=["/api/voice/health"],
            capabilities=[
                Capability(name="transcribe", description="ASR (planned, 501)"),
                Capability(name="analyze_prosody", description="Pitch / tempo / rhythm (planned, 501)"),
            ],
        )

    async def health(self) -> HealthStatus:
        return HealthStatus(
            agent=self.name, state=HealthState.FALLBACK, version=self.version,
            fallback_reasons=["voice agent is a stub — ASR/prosody not wired yet"],
        )

    async def handle_task(self, task: A2ATask) -> A2ATaskResult:
        return A2ATaskResult(task_id=task.task_id, success=False,
                             error="voice agent: not implemented")
