"""Matching Agent — pre-interview CV ↔ JD ranking and shortlist.

Wraps the Paper-derived `RankingPipeline` (TAPJFNN + 14-node bipartite GNN +
optional CAGA shortlist) plus the LLM `CVSignalExtractor` in the platform's
BaseAgent contract.

Fallback chain (declared in configs/runtime.yaml → matching.fallback_chain):
  GNN → TAPJFNN → cosine(MiniLM pooled). The pipeline already handles the
  fallback internally; this adapter only surfaces it via `health()`.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from core.observability import bind_agent_logger

log = bind_agent_logger("matching")

_AGENT_DIR = Path(__file__).resolve().parent
_CARD_PATH = _AGENT_DIR / "agent_card.json"


class MatchingAgent(BaseAgent):
    name = "matching"
    version = "0.2.0"

    def __init__(self) -> None:
        self._router: Optional[APIRouter] = None
        self._pipeline = None              # RankingPipeline (lazy)
        self._extractor = None             # CVSignalExtractor (lazy, singleton)
        self._init_lock = asyncio.Lock()
        self._init_error: Optional[str] = None

    # ------------------------------------------------------------------ lifecycle
    async def startup(self, app_state: Any) -> None:
        # Heavy ML loads stay lazy on first /api/matching/rank call.
        log.info("matching.startup ok (lazy init)")

    async def shutdown(self) -> None:
        self._pipeline = None
        self._extractor = None

    # ------------------------------------------------------------------ surface
    @property
    def router(self) -> APIRouter:
        if self._router is None:
            from .api import build_router
            self._router = build_router(self)
        return self._router

    def card(self) -> AgentCard:
        if _CARD_PATH.exists():
            data = json.loads(_CARD_PATH.read_text())
            return AgentCard.model_validate(data)
        return AgentCard(
            name=self.name,
            version=self.version,
            description="CV ↔ JD ranking with TAPJFNN+GNN, optional CAGA shortlist.",
            capabilities=self._capabilities(),
            routes=["/api/matching/health", "/api/matching/rank",
                    "/api/matching/extract-signals"],
        )

    @staticmethod
    def _capabilities() -> List[Capability]:
        return [
            Capability(
                name="rank",
                description="Rank CVs against a JD; returns sorted fit_scores.",
            ),
            Capability(
                name="extract_signals",
                description="LLM-extract per-CV hiring signals for the GA shortlist.",
            ),
            Capability(
                name="apply_ga",
                description="Run constraint-aware GA on a ranked list and return shortlist.",
            ),
        ]

    # ------------------------------------------------------------------ health
    async def health(self) -> HealthStatus:
        detail: Dict[str, Any] = {}
        reasons: List[str] = []
        state = HealthState.READY

        try:
            p = await self._ensure_pipeline()
        except Exception as e:
            return HealthStatus(
                agent=self.name, state=HealthState.ERROR, version=self.version,
                error=f"pipeline init failed: {e}",
            )

        detail["tapjfnn_loaded"] = p.tapjfnn is not None
        detail["gnn_loaded"] = p.gnn is not None
        detail["lda_loaded"] = p.lda_j is not None and p.lda_r is not None

        if not detail["gnn_loaded"]:
            reasons.append("GNN checkpoint missing — falling back to TAPJFNN/cosine")
            state = HealthState.FALLBACK
        if not detail["tapjfnn_loaded"]:
            reasons.append("TAPJFNN checkpoint missing — falling back to cosine")
            state = HealthState.FALLBACK
        if not detail["lda_loaded"]:
            reasons.append("LDA models missing — TAPJFNN topic attention disabled")
            state = HealthState.FALLBACK

        # LLM probe (non-loading)
        try:
            extractor = self._get_extractor()
            llm = await extractor.health()
            detail["llm"] = llm
            if not llm.get("has_token"):
                reasons.append("HF_TOKEN unset — extract-signals returns 503")
        except Exception as e:
            detail["llm"] = {"error": str(e)}
            reasons.append("LLM probe failed")

        return HealthStatus(
            agent=self.name, state=state, version=self.version,
            fallback_reasons=reasons, detail=detail,
        )

    # ------------------------------------------------------------------ a2a
    async def handle_task(self, task: A2ATask) -> A2ATaskResult:
        cap = task.capability
        try:
            if cap == "rank":
                jd = task.payload["job_description"]
                resumes = task.payload["resumes"]
                ranked = await self.rank(jd, resumes)
                return A2ATaskResult(task_id=task.task_id, success=True,
                                     result={"ranked": ranked})
            if cap == "extract_signals":
                jd = task.payload["job_description"]
                resumes = task.payload["resumes"]
                signals = await self.extract_signals(jd, resumes)
                return A2ATaskResult(task_id=task.task_id, success=True,
                                     result={"signals": signals})
            if cap == "apply_ga":
                ranked = task.payload["ranked"]
                ga_inputs = task.payload["ga_inputs"]
                constraints = task.payload["constraints"]
                ga = await self.apply_ga(ranked, ga_inputs, constraints)
                return A2ATaskResult(task_id=task.task_id, success=True,
                                     result={"ga": ga})
            return A2ATaskResult(
                task_id=task.task_id, success=False,
                error=f"unknown capability: {cap}",
            )
        except Exception as e:
            return A2ATaskResult(
                task_id=task.task_id, success=False, error=f"{type(e).__name__}: {e}",
            )

    # ------------------------------------------------------------------ business methods
    async def rank(self, job_description: str, resumes: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """resumes: [{id, text}]; returns sorted [{rank, id, fit_score, …}]."""
        p = await self._ensure_pipeline()
        ranked = await asyncio.to_thread(p.rank, job_description, resumes)
        return [
            {
                "rank": i + 1,
                "id": c.id,
                "fit_score": c.fit_score,
                "matching_entities": c.matching_entities,
                "resume_entities": c.resume_entities,
                "job_entities": c.job_entities,
            }
            for i, c in enumerate(ranked)
        ]

    async def extract_signals(
        self, job_description: str, resumes: List[Dict[str, str]]
    ) -> Dict[str, Dict[str, Any]]:
        from .backend.agents.cv_extractor import CandidateSignals  # local import
        extractor = self._get_extractor()
        raw = await extractor.extract_batch(job_description, resumes)
        out: Dict[str, Dict[str, Any]] = {}
        for cid, payload in raw.items():
            sig = CandidateSignals(**payload["signals"])
            out[cid] = {
                "metadata": sig.to_ga_metadata(cid),
                "signals": payload["signals"],
                "error": payload.get("error"),
            }
        return out

    async def apply_ga(
        self, ranked: List[Dict[str, Any]], ga_inputs: Dict[str, Dict[str, float]],
        constraints: Dict[str, Any],
    ) -> Dict[str, Any]:
        from .backend.inference import RankedCandidate, apply_ga as _apply_ga
        rcs = [RankedCandidate(
                id=r["id"], fit_score=r["fit_score"],
                matching_entities=r.get("matching_entities", {}),
                resume_entities=r.get("resume_entities", {}),
                job_entities=r.get("job_entities", {}),
            ) for r in ranked]
        return await asyncio.to_thread(_apply_ga, rcs, ga_inputs, constraints)

    # ------------------------------------------------------------------ helpers
    async def _ensure_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        async with self._init_lock:
            if self._pipeline is None:
                # Import here so torch_geometric is only loaded when used.
                from .backend.inference import RankingPipeline
                self._pipeline = await asyncio.to_thread(RankingPipeline)
                log.info(
                    "matching.pipeline ready",
                    tapjfnn=self._pipeline.tapjfnn is not None,
                    gnn=self._pipeline.gnn is not None,
                )
        return self._pipeline

    def _get_extractor(self):
        if self._extractor is None:
            from .backend.agents.cv_extractor import get_extractor
            self._extractor = get_extractor()
        return self._extractor

    @property
    def upload_dir(self) -> Path:
        from .backend.config import UPLOAD_DIR
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        return Path(UPLOAD_DIR)
