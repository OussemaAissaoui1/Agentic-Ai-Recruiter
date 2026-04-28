"""Vision Agent — real-time multimodal behavioral analysis during the interview.

This agent owns the entire stress-detection stack from the upstream repo:
  - MediaPipe FaceMesh + Pose + Hands (visual)
  - HSEmotion EfficientNet (valence / arousal)
  - Wav2Vec2 + StudentNet (audio stress) — StudentNet head is fallback today
  - Welford personal-baseline calibration (45s)
  - Multi-dimension fusion → Cognitive Load / Arousal / Engagement / Confidence
  - PDF + JSON session reports

The Vision Agent is mounted into the unified app and exposes:
  HTTP  /api/vision/{health, session/start, session/{id}/question, report/{id}}
  WS    /ws/vision

During an interview the candidate's browser opens BOTH:
  - the chat stream against /api/nlp (text/audio question + answer)
  - the WebSocket against /ws/vision (jpegs + PCM) for live behavioral scoring

These two pipelines are independent — the only shared state is the
session_id that lets us mark each generated question on the vision timeline
via /api/vision/session/{id}/question (called from the unified app or the
interview UI).
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

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

log = bind_agent_logger("vision")

_AGENT_DIR = Path(__file__).resolve().parent
_CARD_PATH = _AGENT_DIR / "agent_card.json"


class VisionAgent(BaseAgent):
    name = "vision"
    version = "0.2.0"

    def __init__(self) -> None:
        self._router: Optional[APIRouter] = None
        self._ws_router: Optional[APIRouter] = None
        self._initialized = False
        self._init_error: Optional[str] = None

    # ------------------------------------------------------------------ lifecycle
    async def startup(self, app_state: Any) -> None:
        cfg = load_runtime()
        v = cfg.get("vision", {})
        paths = cfg.get("paths", {})

        weights_dir = resolve(paths.get("vision_weights", "agents/vision/weights"))
        report_dir = resolve(paths.get("vision_reports", "artifacts/vision/reports"))
        report_dir.mkdir(parents=True, exist_ok=True)
        # MediaPipe .task files are resolved relative to stress_detection/'s parent
        # (agents/vision/models/), no env var needed.

        # Optional weights — explicit None when absent, so the underlying
        # detectors fall back to defaults (untrained head).
        student_w = weights_dir / "studentnet.pt"
        visual_w = weights_dir / "visual_mlp.pt"

        try:
            from .stress_detection.web_server import init_pipeline
            await asyncio.to_thread(
                init_pipeline,
                studentnet_weights=str(student_w) if student_w.exists() else None,
                visual_mlp_weights=str(visual_w) if visual_w.exists() else None,
                use_body_pose=v.get("use_body_pose", True),
                device=None,                                    # auto cuda/cpu
                enable_video=v.get("enable_video", True),
                enable_audio=v.get("enable_audio", True),
                calibration_duration=v.get("calibration_duration", 45),
                report_output=str(report_dir),
                enable_hands=v.get("enable_hands", True),
                audit_mode=False,
            )
            self._initialized = True
            log.info("vision.startup ok",
                     studentnet=student_w.exists(), visual_mlp=visual_w.exists())
        except Exception as e:
            self._init_error = f"{type(e).__name__}: {e}"
            log.exception("vision.startup failed: %s", e)

    async def shutdown(self) -> None:
        # Detectors hold MediaPipe / torch resources; let GC handle them.
        from .stress_detection import web_server as ws
        ws._visual_detector = None
        ws._audio_detector = None
        ws._fusion = None
        ws._gesture_analyzer = None
        ws._dimension_fusion = None

    # ------------------------------------------------------------------ surface
    @property
    def router(self) -> APIRouter:
        if self._router is None:
            from .stress_detection.web_server import http_router
            self._router = http_router
        return self._router

    @property
    def ws_router(self) -> APIRouter:
        """Mounted at /ws/vision by the unified app."""
        if self._ws_router is None:
            from .stress_detection.web_server import ws_router as r
            self._ws_router = r
        return self._ws_router

    def card(self) -> AgentCard:
        if _CARD_PATH.exists():
            return AgentCard.model_validate(json.loads(_CARD_PATH.read_text()))
        return AgentCard(
            name=self.name, version=self.version,
            description="Real-time multimodal behavioral analysis (4-D profile).",
            transports=["http", "websocket"],
            routes=[
                "/api/vision/health",
                "/api/vision/session/start",
                "/api/vision/session/{session_id}/question",
                "/api/vision/report/{session_id}",
                "/api/vision/report/{session_id}/pdf",
                "/ws/vision",
            ],
            capabilities=self._capabilities(),
        )

    @staticmethod
    def _capabilities() -> List[Capability]:
        return [
            Capability(name="start_session",
                       description="Create a new behavioral-analysis session."),
            Capability(name="mark_question",
                       description="Mark an interviewer question on the timeline."),
            Capability(name="get_report",
                       description="Return the JSON behavioral report for a session."),
        ]

    # ------------------------------------------------------------------ health
    async def health(self) -> HealthStatus:
        from .stress_detection import web_server as ws
        weights_dir = resolve(load_runtime()["paths"]["vision_weights"])
        student_w = weights_dir / "studentnet.pt"
        visual_w = weights_dir / "visual_mlp.pt"

        detail = {
            "visual_loaded": ws._visual_detector is not None,
            "audio_loaded": ws._audio_detector is not None,
            "gesture_loaded": ws._gesture_analyzer is not None,
            "dimension_fusion_loaded": ws._dimension_fusion is not None,
            "studentnet_weights_present": student_w.exists(),
            "visual_mlp_weights_present": visual_w.exists(),
            "active_sessions": len(ws.sessions),
        }

        reasons: List[str] = []
        if not student_w.exists():
            reasons.append("StudentNet weights missing — audio uses untrained head")
        if not visual_w.exists():
            reasons.append("Visual MLP weights missing — visual stress label disabled")

        if self._init_error:
            return HealthStatus(
                agent=self.name, state=HealthState.ERROR, version=self.version,
                error=self._init_error, detail=detail,
            )

        if not self._initialized:
            return HealthStatus(
                agent=self.name, state=HealthState.LOADING, version=self.version,
                detail=detail,
            )

        state = HealthState.FALLBACK if reasons else HealthState.READY
        return HealthStatus(
            agent=self.name, state=state, version=self.version,
            fallback_reasons=reasons, detail=detail,
        )

    # ------------------------------------------------------------------ a2a
    async def handle_task(self, task: A2ATask) -> A2ATaskResult:
        cap = task.capability
        from .stress_detection import web_server as ws

        try:
            if cap == "start_session":
                # Mirrors POST /api/vision/session/start
                import time
                import uuid
                sid = str(uuid.uuid4())[:12]
                cfg = load_runtime().get("vision", {})
                cal = cfg.get("calibration_duration", 45)
                from .stress_detection.session_aggregator import SessionAggregator
                from .stress_detection.baseline import PersonalBaselineCalibrator
                from .stress_detection.report_generator import InterviewReportGenerator
                ws.sessions[sid] = {
                    "aggregator": SessionAggregator(),
                    "calibrator": PersonalBaselineCalibrator(
                        calibration_duration_seconds=cal),
                    "report_gen": InterviewReportGenerator(),
                    "candidate_id": task.payload.get("candidate_id", "candidate"),
                    "created_at": time.time(),
                }
                return A2ATaskResult(task_id=task.task_id, success=True,
                                     result={"session_id": sid})

            if cap == "mark_question":
                sid = task.payload["session_id"]
                label = task.payload["label"]
                sess = ws.sessions.get(sid)
                if not sess:
                    return A2ATaskResult(task_id=task.task_id, success=False,
                                         error=f"session not found: {sid}")
                import time
                from .stress_detection.fusion import DimensionScores
                ts = time.time()
                sess["aggregator"].add_frame(
                    DimensionScores(timestamp=ts), question_label=label,
                )
                return A2ATaskResult(task_id=task.task_id, success=True,
                                     result={"timestamp": ts})

            if cap == "get_report":
                sid = task.payload["session_id"]
                sess = ws.sessions.get(sid)
                if not sess:
                    return A2ATaskResult(task_id=task.task_id, success=False,
                                         error=f"session not found: {sid}")
                summary = sess["aggregator"].get_session_summary()
                timeline = sess["aggregator"].get_timeline_data()
                report = sess["report_gen"].generate_json(
                    summary=summary, timeline_data=timeline,
                    session_id=sid,
                    candidate_id=sess.get("candidate_id", "candidate"),
                )
                return A2ATaskResult(task_id=task.task_id, success=True,
                                     result={"report": report})

            return A2ATaskResult(task_id=task.task_id, success=False,
                                 error=f"unknown capability: {cap}")
        except Exception as e:
            return A2ATaskResult(task_id=task.task_id, success=False,
                                 error=f"{type(e).__name__}: {e}")
