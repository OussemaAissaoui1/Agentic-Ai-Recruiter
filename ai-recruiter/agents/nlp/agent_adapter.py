"""BaseAgent wrapper around the existing NLPAgent.

The heavy `NLPAgent` (vLLM Llama-3.1-8B + Qwen scorer/refiner + Kokoro TTS)
stays exactly as-is. This adapter exposes it through the platform contract
so the orchestrator can dispatch tasks and the unified app can mount the
router.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, List, Optional

from fastapi import APIRouter

from configs import load_runtime, resolve
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

log = bind_agent_logger("nlp")

_AGENT_DIR = Path(__file__).resolve().parent
_CARD_PATH = _AGENT_DIR / "agent_card.json"


class NLPAgentAdapter(BaseAgent):
    name = "nlp"
    version = "0.2.0"

    def __init__(self) -> None:
        self._inner = None
        self._router: Optional[APIRouter] = None
        self._init_error: Optional[str] = None

    # ------------------------------------------------------------------ lifecycle
    async def startup(self, app_state: Any) -> None:
        cfg = load_runtime().get("nlp", {})
        # Context window: env override > config > 8192 default. 2048 is too
        # small for system prompt + CV + JD + history (vLLM rejects long
        # prompts with "maximum context length is N tokens").
        max_model_len = int(
            os.environ.get("NLP_MAX_MODEL_LEN", cfg.get("max_model_len", 8192))
        )
        try:
            from .agent import NLPAgent
            self._inner = NLPAgent(
                model_path=str(resolve(cfg.get("model_path", "model_cache"))),
                max_model_len=max_model_len,
                enable_scorer=cfg.get("enable_scorer", True),
                scorer_model=cfg.get("scorer_model", "Qwen/Qwen2.5-1.5B-Instruct"),
                enable_refiner=cfg.get("enable_refiner", True),
                refiner_model=cfg.get("refiner_model", "Qwen/Qwen2.5-1.5B-Instruct"),
                enable_tts=cfg.get("enable_tts", True),
                tts_voice=cfg.get("tts_voice", "af_heart"),
                gpu_memory_utilization=load_runtime()
                    .get("gpu_policy", {}).get("vllm_mem_fraction", 0.55),
                use_compact_prompt=False,
                auto_cleanup_gpu=False,
                load_timeout_sec=300,
            )
            log.info("nlp.startup ok (lazy vllm, max_model_len=%d)", max_model_len)
        except Exception as e:
            self._init_error = f"{type(e).__name__}: {e}"
            log.exception("nlp.startup failed: %s", e)

    async def shutdown(self) -> None:
        # vLLM holds the GPU; let process exit reclaim it.
        self._inner = None

    # ------------------------------------------------------------------ surface
    @property
    def router(self) -> APIRouter:
        if self._router is None:
            from .api import build_router
            self._router = build_router(self)
        return self._router

    @property
    def inner(self):
        if self._inner is None:
            raise RuntimeError("NLP agent not initialized — wait for /api/nlp/warmup")
        return self._inner

    def card(self) -> AgentCard:
        if _CARD_PATH.exists():
            return AgentCard.model_validate(json.loads(_CARD_PATH.read_text()))
        return AgentCard(
            name=self.name, version=self.version,
            description="LLM-driven recruiter (Alex). Streams text + audio.",
            transports=["http"],
            routes=[
                "/api/nlp/health", "/api/nlp/warmup",
                "/api/nlp/session", "/api/nlp/chat", "/api/nlp/stream",
            ],
            capabilities=self._capabilities(),
        )

    @staticmethod
    def _capabilities() -> List[Capability]:
        return [
            Capability(name="generate_question",
                       description="Generate the next interview question (with audio)."),
            Capability(name="analyze_response",
                       description="Lightweight analysis of a candidate answer."),
            Capability(name="reset_session",
                       description="Reset state for a session_id."),
        ]

    # ------------------------------------------------------------------ health
    async def health(self) -> HealthStatus:
        if self._init_error:
            return HealthStatus(
                agent=self.name, state=HealthState.ERROR, version=self.version,
                error=self._init_error,
            )
        if self._inner is None:
            return HealthStatus(
                agent=self.name, state=HealthState.LOADING, version=self.version,
            )

        s = self._inner.get_status()
        ready = s["engine_ready"]
        reasons: List[str] = []
        if not s["engine_ready"]:
            reasons.append("vLLM engine not warmed up — POST /api/nlp/warmup")
        if not s["scorer_ready"] and s["scorer_loaded"]:
            reasons.append("scorer init issue")
        if not s["tts_ready"] and s["tts_loaded"]:
            reasons.append("TTS init issue")

        state = HealthState.READY if ready else HealthState.LOADING
        if reasons and ready:
            state = HealthState.FALLBACK
        return HealthStatus(
            agent=self.name, state=state, version=self.version,
            fallback_reasons=reasons, detail=s,
        )

    # ------------------------------------------------------------------ a2a
    async def handle_task(self, task: A2ATask) -> A2ATaskResult:
        try:
            if task.capability == "generate_question":
                from .agent import QuestionGenerationRequest
                req = QuestionGenerationRequest(**task.payload)
                resp = await self.inner.generate_question(req)
                return A2ATaskResult(
                    task_id=task.task_id, success=True,
                    result={"question": resp.question, "reasoning": resp.reasoning,
                            "is_follow_up": resp.is_follow_up,
                            "topic_depth": resp.topic_depth,
                            "turn_count": resp.turn_count},
                )
            if task.capability == "analyze_response":
                sid = task.payload["session_id"]
                ans = task.payload["answer"]
                return A2ATaskResult(task_id=task.task_id, success=True,
                                     result=self.inner.analyze_response(sid, ans))
            if task.capability == "reset_session":
                self.inner.reset_session(task.payload["session_id"])
                return A2ATaskResult(task_id=task.task_id, success=True, result={})
            return A2ATaskResult(task_id=task.task_id, success=False,
                                 error=f"unknown capability: {task.capability}")
        except Exception as e:
            return A2ATaskResult(task_id=task.task_id, success=False,
                                 error=f"{type(e).__name__}: {e}")
