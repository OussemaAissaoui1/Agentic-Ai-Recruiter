"""Scoring Agent — collects per-modality scores, writes session JSONL.

This is a deliberately minimal stub. Future state: weighted aggregation
across NLP / vision / voice signals into a hiring recommendation. Today it:

  - accepts `record_score` tasks from any agent and appends them to
    `artifacts/scoring/<session_id>.jsonl`;
  - returns the raw score history on `get_scores`;
  - returns 501 on `compute_final` (intentional — out of scope for now).

The contract surface stays stable so when we add weighted aggregation,
nothing upstream changes.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from configs import resolve
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

log = bind_agent_logger("scoring")


# Module-scope so FastAPI/Pydantic 2.12 can resolve forward references for /docs.
class ScoreIn(BaseModel):
    source: str                      # "nlp" | "vision" | "voice" | …
    kind: str                        # e.g. "answer_quality", "engagement"
    value: float
    payload: Dict[str, Any] = {}


class ScoringAgent(BaseAgent):
    name = "scoring"
    version = "0.2.0"

    def __init__(self) -> None:
        self._router: Optional[APIRouter] = None
        self._dir: Optional[Path] = None

    async def startup(self, app_state: Any) -> None:
        self._dir = resolve("artifacts/scoring")
        self._dir.mkdir(parents=True, exist_ok=True)

    async def shutdown(self) -> None:
        pass

    @property
    def router(self) -> APIRouter:
        if self._router is None:
            self._router = self._build_router()
        return self._router

    def card(self) -> AgentCard:
        return AgentCard(
            name=self.name, version=self.version,
            description="Collects per-modality scores into per-session JSONL.",
            routes=["/api/scoring/health", "/api/scoring/{session_id}/scores",
                    "/api/scoring/{session_id}/score"],
            capabilities=[
                Capability(name="record_score",
                           description="Append one score record for a session."),
                Capability(name="get_scores",
                           description="Return all recorded scores for a session."),
                Capability(name="compute_final",
                           description="Compute final hiring recommendation (NOT IMPLEMENTED)."),
            ],
        )

    async def health(self) -> HealthStatus:
        return HealthStatus(
            agent=self.name, state=HealthState.READY, version=self.version,
            detail={"dir": str(self._dir)},
        )

    # ------------------------------------------------------------------ a2a
    async def handle_task(self, task: A2ATask) -> A2ATaskResult:
        try:
            if task.capability == "record_score":
                self._append(task.payload["session_id"], task.payload["score"])
                return A2ATaskResult(task_id=task.task_id, success=True, result={"ok": True})
            if task.capability == "get_scores":
                return A2ATaskResult(task_id=task.task_id, success=True,
                                     result={"scores": self._read(task.payload["session_id"])})
            if task.capability == "compute_final":
                return A2ATaskResult(task_id=task.task_id, success=False,
                                     error="compute_final is not implemented yet")
            return A2ATaskResult(task_id=task.task_id, success=False,
                                 error=f"unknown capability: {task.capability}")
        except Exception as e:
            return A2ATaskResult(task_id=task.task_id, success=False,
                                 error=f"{type(e).__name__}: {e}")

    # ------------------------------------------------------------------ helpers
    def _path(self, session_id: str) -> Path:
        assert self._dir is not None
        return self._dir / f"{session_id}.jsonl"

    def _append(self, session_id: str, score: Dict[str, Any]) -> None:
        record = {"ts": time.time(), **score}
        with self._path(session_id).open("a") as f:
            f.write(json.dumps(record) + "\n")

    def _read(self, session_id: str) -> List[Dict[str, Any]]:
        p = self._path(session_id)
        if not p.exists():
            return []
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]

    # ------------------------------------------------------------------ router
    def _build_router(self) -> APIRouter:
        router = APIRouter(tags=["scoring"])

        @router.get("/health")
        async def health():
            return (await self.health()).model_dump()

        @router.post("/{session_id}/score")
        async def post_score(session_id: str, score: ScoreIn):
            self._append(session_id, score.model_dump())
            return {"ok": True}

        @router.get("/{session_id}/scores")
        async def get_scores(session_id: str):
            return {"session_id": session_id, "scores": self._read(session_id)}

        @router.post("/{session_id}/final")
        async def final(session_id: str):
            raise HTTPException(501, "Final score aggregation not implemented yet")

        return router
