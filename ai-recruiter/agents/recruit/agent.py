"""Recruit agent — sqlite-backed CRUD + AI JD generation for HireFlow."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, List, Optional

from fastapi import APIRouter

from core.contracts import (
    A2ATask,
    A2ATaskResult,
    AgentCard,
    BaseAgent,
    Capability,
    HealthState,
    HealthStatus,
)

from . import db as recruit_db

log = logging.getLogger("agents.recruit")

_AGENT_DIR = Path(__file__).resolve().parent
_DB_PATH = _AGENT_DIR / "data" / "recruit.db"


class RecruitAgent(BaseAgent):
    """Backend for the HireFlow frontend.

    Owns four sqlite tables (jobs, applications, notifications, interviews)
    and exposes CRUD over them plus an SSE-streaming JD generator.
    """

    name = "recruit"
    version = "0.1.0"

    def __init__(self) -> None:
        self._router: Optional[APIRouter] = None
        self._conn = None  # sqlite3.Connection
        self._registry = None  # type: ignore[assignment]
        self._lock = asyncio.Lock()
        self._init_error: Optional[str] = None
        self._db_path = _DB_PATH

    # ------------------------------------------------------------------ lifecycle
    async def startup(self, app_state: Any) -> None:
        try:
            # The platform's lifespan calls startup with an `app_state` that
            # may carry a registry — pull it out so JD generation can reach
            # the NLP agent at runtime. We tolerate any shape: AgentRegistry,
            # FastAPI state, dict, or None.
            self._registry = self._extract_registry(app_state)
            self._conn = await asyncio.to_thread(recruit_db.init_db, self._db_path)
            inserted = await asyncio.to_thread(recruit_db.seed_if_empty, self._conn)
            log.info(
                "recruit.startup ok db=%s seeded=%d", self._db_path, inserted
            )
        except Exception as e:
            self._init_error = f"{type(e).__name__}: {e}"
            log.exception("recruit.startup failed: %s", e)

    async def shutdown(self) -> None:
        try:
            if self._conn is not None:
                self._conn.close()
        except Exception:
            pass
        self._conn = None

    # ------------------------------------------------------------------ surface
    @property
    def router(self) -> APIRouter:
        if self._router is None:
            from .api import build_router
            self._router = build_router(self)
        return self._router

    @property
    def conn(self):
        if self._conn is None:
            # Lazy fallback — if startup never fired (e.g. when imported in a
            # script), open the DB on demand so tests still work.
            self._conn = recruit_db.init_db(self._db_path)
            recruit_db.seed_if_empty(self._conn)
        return self._conn

    @property
    def registry(self):
        return self._registry

    def card(self) -> AgentCard:
        return AgentCard(
            name=self.name,
            version=self.version,
            description=(
                "HireFlow backend: jobs/applications/notifications CRUD + "
                "AI-powered JD generation with NLP-agent fallback."
            ),
            routes=[
                "/api/recruit/health",
                "/api/recruit/jobs",
                "/api/recruit/jobs/{id}",
                "/api/recruit/jobs/generate",
                "/api/recruit/applications",
                "/api/recruit/applications/{id}",
                "/api/recruit/applications/{id}/invite-interview",
                "/api/recruit/notifications",
                "/api/recruit/notifications/{id}",
                "/api/recruit/interviews",
                "/api/recruit/interviews/{id}",
            ],
            capabilities=self._capabilities(),
        )

    @staticmethod
    def _capabilities() -> List[Capability]:
        return [
            Capability(name="jobs_crud",
                       description="Create/read/update/delete job postings."),
            Capability(name="applications_crud",
                       description="Track candidate applications and stages."),
            Capability(name="notifications",
                       description="Per-user notification feed."),
            Capability(name="generate_jd",
                       description="Stream a job description from inputs."),
        ]

    # ------------------------------------------------------------------ health
    async def health(self) -> HealthStatus:
        if self._init_error:
            return HealthStatus(
                agent=self.name, state=HealthState.ERROR, version=self.version,
                error=self._init_error,
            )
        if self._conn is None:
            return HealthStatus(
                agent=self.name, state=HealthState.LOADING, version=self.version,
            )
        try:
            jobs_count = await asyncio.to_thread(recruit_db.count_jobs, self._conn)
        except Exception as e:
            return HealthStatus(
                agent=self.name, state=HealthState.ERROR, version=self.version,
                error=f"db probe failed: {e}",
            )
        nlp_ready = self._nlp_ready()
        reasons: List[str] = []
        if not nlp_ready:
            reasons.append("NLP agent not warm — JD generation uses templated fallback")
        return HealthStatus(
            agent=self.name,
            state=HealthState.READY if nlp_ready else HealthState.FALLBACK,
            version=self.version,
            fallback_reasons=reasons,
            detail={
                "db_path": str(self._db_path),
                "jobs_count": jobs_count,
                "nlp_ready": nlp_ready,
            },
        )

    # ------------------------------------------------------------------ a2a
    async def handle_task(self, task: A2ATask) -> A2ATaskResult:
        # The recruit agent is HTTP-first; orchestrator-driven flows aren't
        # wired yet. Surface a clean "not implemented" so misrouted tasks fail
        # loudly rather than silently.
        return A2ATaskResult(
            task_id=task.task_id, success=False,
            error=f"capability not exposed via A2A yet: {task.capability}",
        )

    # ------------------------------------------------------------------ helpers
    def _extract_registry(self, app_state: Any):
        """Best-effort: dig the AgentRegistry out of whatever startup got."""
        if app_state is None:
            return None
        # AgentRegistry duck-typing
        for attr in ("get", "all", "names"):
            if not hasattr(app_state, attr):
                break
        else:
            return app_state
        # FastAPI app.state
        for key in ("registry", "agent_registry"):
            obj = getattr(app_state, key, None)
            if obj is not None:
                return obj
        if isinstance(app_state, dict):
            return app_state.get("registry") or app_state.get("agent_registry")
        return None

    def _nlp_agent(self):
        if self._registry is None:
            return None
        try:
            return self._registry.get("nlp")
        except Exception:
            return None

    def _nlp_ready(self) -> bool:
        agent = self._nlp_agent()
        if agent is None:
            return False
        inner = getattr(agent, "_inner", None)
        if inner is None:
            return False
        try:
            return bool(inner.is_ready)
        except Exception:
            return False
