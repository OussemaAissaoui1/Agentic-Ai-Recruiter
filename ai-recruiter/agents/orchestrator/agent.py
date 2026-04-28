"""Orchestrator Agent — central A2A dispatcher.

Today this is a *stub* that exposes the in-process registry through HTTP so
external callers can route tasks to any agent without knowing internal
addressing. It will become a LangGraph state machine that drives the full
matching → interview → scoring lifecycle in a future phase.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request

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

log = bind_agent_logger("orchestrator")


class OrchestratorAgent(BaseAgent):
    name = "orchestrator"
    version = "0.2.0"

    def __init__(self) -> None:
        self._router: Optional[APIRouter] = None
        self._registry = None  # filled in startup

    async def startup(self, app_state: Any) -> None:
        self._registry = app_state.registry

    async def shutdown(self) -> None:
        self._registry = None

    @property
    def router(self) -> APIRouter:
        if self._router is None:
            self._router = self._build_router()
        return self._router

    def card(self) -> AgentCard:
        return AgentCard(
            name=self.name, version=self.version,
            description="A2A dispatcher and (future) interview-flow state machine.",
            routes=["/api/orchestrator/health", "/api/orchestrator/dispatch",
                    "/api/orchestrator/agents"],
            capabilities=[
                Capability(name="dispatch",
                           description="Route an A2ATask to its target agent in-process."),
                Capability(name="list_agents",
                           description="Return the names of every registered agent."),
            ],
        )

    async def health(self) -> HealthStatus:
        return HealthStatus(
            agent=self.name, state=HealthState.READY, version=self.version,
            detail={"agents": self._registry.names() if self._registry else []},
        )

    async def handle_task(self, task: A2ATask) -> A2ATaskResult:
        if task.capability == "list_agents":
            return A2ATaskResult(task_id=task.task_id, success=True,
                                 result={"agents": self._registry.names()})
        if task.capability == "dispatch":
            inner = A2ATask.model_validate(task.payload)
            return await self._registry.dispatch(inner)
        return A2ATaskResult(task_id=task.task_id, success=False,
                             error=f"unknown capability: {task.capability}")

    # ------------------------------------------------------------------ router
    def _build_router(self) -> APIRouter:
        router = APIRouter(tags=["orchestrator"])

        @router.get("/health")
        async def health():
            return (await self.health()).model_dump()

        @router.get("/agents")
        async def list_agents():
            cards = self._registry.cards()
            return {name: card.model_dump() for name, card in cards.items()}

        @router.post("/dispatch")
        async def dispatch(task: A2ATask):
            if self._registry is None:
                raise HTTPException(503, "registry not ready")
            result = await self._registry.dispatch(task)
            return result.model_dump()

        return router
