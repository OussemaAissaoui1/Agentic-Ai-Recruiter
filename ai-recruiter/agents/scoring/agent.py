"""Scoring Agent — produces hiring-decision reports from recorded interviews.

POST /{interview_id}/run kicks off (or returns cached) scoring. Reads
transcript, CV, and JD from the recruit SQLite DB; uses Groq for the LLM
calls; persists JSON + Markdown under artifacts/scoring/<interview_id>_report.*
and writes the headline numbers back to interviews.scores_json.
"""
from __future__ import annotations

import asyncio
import os
import sqlite3
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Response

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

from .llm_client import GroqClient, GroqError
from .reader import load_context
from .report import (
    delete_cache,
    read_cache,
    write_back_scores,
    write_cache,
)
from .schema import InterviewReport, ScoreRequest
from .scorer import score_interview

log = bind_agent_logger("scoring")

DEFAULT_MODEL = os.environ.get("SCORING_MODEL", "llama-3.3-70b-versatile")
DEFAULT_CONCURRENCY = int(os.environ.get("SCORING_CONCURRENCY", "4"))
DEFAULT_TIMEOUT = float(os.environ.get("SCORING_TIMEOUT_SEC", "30"))


class ScoringAgent(BaseAgent):
    name = "scoring"
    version = "0.3.0"

    def __init__(self) -> None:
        self._router: Optional[APIRouter] = None
        self._artifacts_dir: Optional[Path] = None
        self._recruit_conn: Optional[sqlite3.Connection] = None
        self._groq: Optional[GroqClient] = None

    async def startup(self, app_state: Any) -> None:
        # Tests inject these directly; in the real app they come via
        # configs.resolve and the recruit agent's connection.
        self._artifacts_dir = (
            getattr(app_state, "artifacts_dir", None)
            or resolve("artifacts/scoring")
        )
        self._artifacts_dir.mkdir(parents=True, exist_ok=True)

        self._recruit_conn = (
            getattr(app_state, "recruit_conn", None)
            or _open_recruit_conn_default()
        )

        injected = getattr(app_state, "fake_groq_client", None)
        if injected is not None:
            self._groq = injected
        else:
            try:
                self._groq = GroqClient.from_env()
            except GroqError as exc:
                log.warning("scoring.startup: %s — /run will return 503", exc)
                self._groq = None

    async def shutdown(self) -> None:
        if self._groq is not None and hasattr(self._groq, "aclose"):
            try:
                await self._groq.aclose()
            except Exception:
                pass

    @property
    def router(self) -> APIRouter:
        if self._router is None:
            self._router = self._build_router()
        return self._router

    def card(self) -> AgentCard:
        return AgentCard(
            name=self.name, version=self.version,
            description="Generates per-answer + overall hiring report from a completed interview.",
            routes=[
                "/api/scoring/health",
                "/api/scoring/{interview_id}/run",
                "/api/scoring/{interview_id}/report",
                "/api/scoring/{interview_id}/report.md",
            ],
            capabilities=[
                Capability(name="score_interview",
                           description="Run scoring for an interview (or return cached report)."),
                Capability(name="get_report",
                           description="Return cached InterviewReport JSON."),
                Capability(name="delete_report",
                           description="Drop cached report so the next run regenerates."),
            ],
        )

    async def health(self) -> HealthStatus:
        ready = self._groq is not None and self._recruit_conn is not None
        # Core enum exposes FALLBACK rather than DEGRADED; semantics are the
        # same: we boot in this state when GROQ_API_KEY is missing or the
        # recruit DB couldn't be opened, so /run will surface 503 until fixed.
        state = HealthState.READY if ready else HealthState.FALLBACK
        return HealthStatus(
            agent=self.name, state=state, version=self.version,
            detail={
                "groq": self._groq is not None,
                "recruit_db": self._recruit_conn is not None,
                "artifacts_dir": str(self._artifacts_dir) if self._artifacts_dir else None,
                "model": DEFAULT_MODEL,
            },
        )

    # ---------------------------------------------------------------- a2a
    async def handle_task(self, task: A2ATask) -> A2ATaskResult:
        try:
            if task.capability == "score_interview":
                interview_id = task.payload["interview_id"]
                req = ScoreRequest.model_validate(task.payload.get("request") or {})
                report = await self._run(interview_id, req)
                return A2ATaskResult(task_id=task.task_id, success=True,
                                     result=report.model_dump())
            if task.capability == "get_report":
                interview_id = task.payload["interview_id"]
                report = read_cache(self._artifacts_dir, interview_id)
                if report is None:
                    return A2ATaskResult(task_id=task.task_id, success=False,
                                         error="not_found")
                return A2ATaskResult(task_id=task.task_id, success=True,
                                     result=report.model_dump())
            if task.capability == "delete_report":
                interview_id = task.payload["interview_id"]
                ok = delete_cache(self._artifacts_dir, interview_id)
                return A2ATaskResult(task_id=task.task_id, success=True,
                                     result={"deleted": ok})
            return A2ATaskResult(task_id=task.task_id, success=False,
                                 error=f"unknown capability: {task.capability}")
        except HTTPException as exc:
            return A2ATaskResult(task_id=task.task_id, success=False,
                                 error=f"{exc.status_code}: {exc.detail}")
        except Exception as exc:
            return A2ATaskResult(task_id=task.task_id, success=False,
                                 error=f"{type(exc).__name__}: {exc}")

    # ---------------------------------------------------------------- core
    async def _run(self, interview_id: str, req: ScoreRequest) -> InterviewReport:
        if not req.force:
            cached = read_cache(self._artifacts_dir, interview_id)
            if cached is not None:
                return cached

        if self._groq is None:
            raise HTTPException(503, "GROQ_API_KEY is not configured")
        if self._recruit_conn is None:
            raise HTTPException(503, "recruit DB is not configured")

        try:
            ctx = await asyncio.to_thread(
                load_context, self._recruit_conn, interview_id,
                req.transcript,
            )
        except LookupError as exc:
            raise HTTPException(404, str(exc))

        if not ctx.transcript:
            raise HTTPException(422, "no transcript available for this interview")

        report = await score_interview(
            client=self._groq,
            model=DEFAULT_MODEL,
            interview_id=ctx.interview_id,
            candidate_name=ctx.candidate_name,
            job_title=ctx.job_title,
            cv_text=ctx.cv_text,
            jd_text=ctx.jd_text,
            transcript=ctx.transcript,
            behavior_summary=ctx.behavior_summary,
            concurrency=DEFAULT_CONCURRENCY,
            timeout=DEFAULT_TIMEOUT,
        )

        try:
            await asyncio.to_thread(write_cache, self._artifacts_dir, report)
        except OSError as exc:
            log.warning("scoring.write_cache failed: %s", exc)
        try:
            await asyncio.to_thread(write_back_scores, self._recruit_conn, report)
        except Exception as exc:
            log.warning("scoring.write_back_scores failed: %s", exc)
        return report

    # ---------------------------------------------------------------- router
    def _build_router(self) -> APIRouter:
        router = APIRouter(tags=["scoring"])

        @router.get("/health")
        async def health():
            return (await self.health()).model_dump()

        @router.post("/{interview_id}/run")
        async def run(interview_id: str, req: ScoreRequest):
            report = await self._run(interview_id, req)
            return report.model_dump()

        @router.get("/{interview_id}/report")
        async def get_report(interview_id: str):
            cached = read_cache(self._artifacts_dir, interview_id)
            if cached is None:
                raise HTTPException(404, "report not found")
            return cached.model_dump()

        @router.get("/{interview_id}/report.md")
        async def get_markdown(interview_id: str):
            md_path = self._artifacts_dir / f"{interview_id}_report.md"
            if not md_path.exists():
                raise HTTPException(404, "markdown report not found")
            return Response(md_path.read_text(), media_type="text/markdown")

        @router.delete("/{interview_id}/report")
        async def delete_report(interview_id: str):
            ok = delete_cache(self._artifacts_dir, interview_id)
            return {"deleted": ok}

        return router


def _open_recruit_conn_default() -> Optional[sqlite3.Connection]:
    """Open the recruit DB read/write so write-back works.

    Best-effort: returns None on any error. Tests inject a connection
    directly via app_state.recruit_conn.
    """
    try:
        from agents.recruit import db as recruit_db
        path = resolve("agents/recruit/data/recruit.db")
        return recruit_db.init_db(path)
    except Exception as exc:
        log.warning("scoring._open_recruit_conn_default failed: %s", exc)
        return None
