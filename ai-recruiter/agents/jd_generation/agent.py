"""JDGenerationAgent — BaseAgent subclass registered in main.py.

Owns: Neo4j client lifecycle, FastAPI router, health probe, agent card.
"""

from __future__ import annotations

import asyncio
import logging
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

from .config import JDConfig, load_jd_config
from .graph.client import Neo4jNotConfigured, close_client, get_client
from .graph.schema import apply_schema

log = logging.getLogger("agents.jd_generation")


class JDGenerationAgent(BaseAgent):
    """Graph-RAG JD generation agent.

    `startup` attempts to open the Neo4j driver and apply the schema. If
    that fails (env vars missing, network blocked) the agent comes up in
    ERROR state — health surfaces a clear reason and other routes still
    respond with 503 instead of crashing the whole platform.
    """

    name = "jd"
    version = "0.1.0"

    def __init__(self) -> None:
        self._router: Optional[APIRouter] = None
        self._cfg: Optional[JDConfig] = None
        self._startup_error: Optional[str] = None
        self._schema_ok: bool = False

    # ------------------------------------------------------------------ lifecycle
    async def startup(self, app_state: Any) -> None:  # noqa: ARG002 — unused
        try:
            self._cfg = load_jd_config()
            if not self._cfg.neo4j.is_configured:
                self._startup_error = (
                    "Neo4j env vars not set (JD_NEO4J_URI / JD_NEO4J_USER / JD_NEO4J_PASSWORD). "
                    "Agent will start in ERROR state."
                )
                log.warning("jd_generation.startup: %s", self._startup_error)
                return
            client = await asyncio.to_thread(get_client, self._cfg)
            await asyncio.to_thread(client.verify)
            await asyncio.to_thread(apply_schema, client)
            self._schema_ok = True
            log.info("jd_generation.startup ok db=%s", self._cfg.neo4j.database)
        except Neo4jNotConfigured as e:
            self._startup_error = str(e)
            log.warning("jd_generation.startup: %s", e)
        except Exception as e:  # noqa: BLE001 — surface to health, don't crash
            self._startup_error = f"{type(e).__name__}: {e}"
            log.exception("jd_generation.startup failed")

    async def shutdown(self) -> None:
        await asyncio.to_thread(close_client)

    # ------------------------------------------------------------------ surface
    @property
    def router(self) -> APIRouter:
        if self._router is None:
            from .api import build_router
            self._router = build_router(self)
        return self._router

    @property
    def config(self) -> JDConfig:
        return self._cfg or load_jd_config()

    @property
    def startup_error(self) -> Optional[str]:
        return self._startup_error

    @property
    def is_ready(self) -> bool:
        return self._startup_error is None and self._schema_ok

    # ------------------------------------------------------------------ discovery
    def card(self) -> AgentCard:
        return AgentCard(
            name=self.name,
            version=self.version,
            description=(
                "Graph-RAG JD generator backed by Neo4j + Groq. Produces a "
                "job description with evidence panel and a rejection-feedback "
                "loop. Coexists with the simpler agents/recruit/jd_gen.py."
            ),
            routes=[
                "/api/jd/health",
                "/api/jd/generate",
                "/api/jd/{jd_id}",
                "/api/jd/{jd_id}/approve",
                "/api/jd/{jd_id}/reject",
                "/api/jd/graph/dump",
                "/api/jd/graph/upload",
                "/api/jd/graph/seed-default",
            ],
            capabilities=self._capabilities(),
        )

    @staticmethod
    def _capabilities() -> List[Capability]:
        return [
            Capability(
                name="generate_jd",
                description=("Generate a JD via graph-RAG over the employee graph. "
                             "Streams tool-call progress over SSE."),
            ),
            Capability(
                name="reject_jd",
                description=("Persist a recruiter rejection. Free-text reasons are "
                             "classified into a 7-item taxonomy."),
            ),
            Capability(
                name="approve_jd",
                description="Mark a draft JD as approved.",
            ),
            Capability(
                name="dump_graph",
                description="D3-shaped JSON dump of the JD graph for visualization.",
            ),
            Capability(
                name="upload_graph",
                description=("Replace or augment the JD graph from a JSON or "
                             "CSV-zip-bundle upload."),
            ),
        ]

    # ------------------------------------------------------------------ health
    async def health(self) -> HealthStatus:
        if self._startup_error is not None:
            return HealthStatus(
                agent=self.name, state=HealthState.ERROR, version=self.version,
                error=self._startup_error,
            )
        if not self._schema_ok or self._cfg is None:
            return HealthStatus(
                agent=self.name, state=HealthState.LOADING, version=self.version,
            )
        cfg = self._cfg
        fallback_reasons: List[str] = []
        if not cfg.llm.is_configured:
            fallback_reasons.append(
                "GROQ_API_KEY not set — agent loop will fail at generation time"
            )

        # Quick probe — counts. Wrap in try/except so a transient blip doesn't
        # flip the agent to ERROR.
        try:
            client = await asyncio.to_thread(get_client, cfg)
            rows = await asyncio.to_thread(
                client.read, "MATCH (n) RETURN count(n) AS n",
            )
            node_count = rows[0]["n"] if rows else 0
        except Exception as e:  # noqa: BLE001
            return HealthStatus(
                agent=self.name, state=HealthState.FALLBACK, version=self.version,
                fallback_reasons=[f"db probe failed: {e}"] + fallback_reasons,
            )

        state = HealthState.READY if not fallback_reasons else HealthState.FALLBACK
        return HealthStatus(
            agent=self.name, state=state, version=self.version,
            fallback_reasons=fallback_reasons,
            detail={
                "neo4j_database": cfg.neo4j.database,
                "node_count":      node_count,
                "llm_model":       cfg.llm.model,
            },
        )

    # ------------------------------------------------------------------ a2a
    async def handle_task(self, task: A2ATask) -> A2ATaskResult:
        # Like recruit/, this agent is HTTP-first. Orchestrator-driven flows
        # aren't wired yet — surface a clean "not implemented".
        return A2ATaskResult(
            task_id=task.task_id, success=False,
            error=f"capability not exposed via A2A yet: {task.capability}",
        )
