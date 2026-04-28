"""
Web Server for Stress Detection
=================================
FastAPI + WebSocket server that receives video frames and audio
from the browser, runs them through the stress detection pipeline,
and streams results back in real time.

Browser (getUserMedia) → WebSocket → decode frame/audio → pipeline → results → WebSocket → browser overlay
"""

import asyncio
import base64
import json
import logging
import os
import time
import uuid
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
from pathlib import Path
from pydantic import BaseModel

from .audio_model import AudioStressDetector, resample_audio
from .baseline import PersonalBaselineCalibrator
from .body_language import GestureAnalyzer
from .fusion import DimensionScores, FusionStrategy, MultiDimensionFusion, MultimodalFusion
from .report_generator import InterviewReportGenerator
from .session_aggregator import SessionAggregator
from .visual_model import VisualStressDetector

logger = logging.getLogger(__name__)

# Two routers — the unified app mounts http_router under /api/vision and
# ws_router at root (so /ws becomes /ws/vision via prefix).
http_router = APIRouter(tags=["vision"])
ws_router = APIRouter()

# ── Global pipeline state (initialized on startup) ──
_visual_detector: Optional[VisualStressDetector] = None
_audio_detector: Optional[AudioStressDetector] = None
_fusion: Optional[MultimodalFusion] = None
_gesture_analyzer: Optional[GestureAnalyzer] = None
_dimension_fusion: Optional[MultiDimensionFusion] = None
_config: dict = {}

# ── Session management (in-memory) ──
sessions: dict[str, dict] = {}  # session_id → {aggregator, calibrator, report_gen, ...}


def init_pipeline(
    studentnet_weights: Optional[str] = None,
    visual_mlp_weights: Optional[str] = None,
    use_body_pose: bool = True,
    device: Optional[str] = None,
    enable_video: bool = True,
    enable_audio: bool = True,
    calibration_duration: int = 45,
    report_output: str = "./reports/",
    enable_hands: bool = True,
    audit_mode: bool = False,
):
    """Initialize the ML models. Called once at server startup."""
    global _visual_detector, _audio_detector, _fusion, _gesture_analyzer, _dimension_fusion, _config

    _config = {
        "use_body_pose": use_body_pose,
        "enable_video": enable_video,
        "enable_audio": enable_audio,
        "calibration_duration": calibration_duration,
        "report_output": report_output,
        "enable_hands": enable_hands,
        "audit_mode": audit_mode,
    }

    if enable_audio:
        logger.info("Initializing audio stress detector...")
        _audio_detector = AudioStressDetector(
            studentnet_weights=studentnet_weights,
            device=device,
        )

    if enable_video:
        logger.info("Initializing visual stress detector...")
        _visual_detector = VisualStressDetector(
            weights_path=visual_mlp_weights,
            use_body_pose=use_body_pose,
            device=device,
        )

    if enable_hands:
        logger.info("Initializing gesture analyzer (MediaPipe Hands)...")
        try:
            _gesture_analyzer = GestureAnalyzer()
        except Exception as e:
            logger.warning("Failed to initialize GestureAnalyzer: %s", e)
            _gesture_analyzer = None

    _fusion = MultimodalFusion(
        strategy=FusionStrategy.WEIGHTED_AVERAGE,
        device=device,
    )
    _dimension_fusion = MultiDimensionFusion()
    logger.info("Pipeline initialized.")


# ── Serve the frontend ──
_PROJECT_ROOT = Path(__file__).parent.parent
_REACT_DIST = _PROJECT_ROOT / "frontend" / "dist"
_STATIC_DIR = _PROJECT_ROOT / "static"


@http_router.get("/health")
async def health():
    return {
        "status": "ok",
        "models": {
            "visual": _visual_detector is not None,
            "audio": _audio_detector is not None,
            "fusion": _fusion is not None,
            "gesture": _gesture_analyzer is not None,
            "dimension_fusion": _dimension_fusion is not None,
        },
        "config": _config,
    }


# ── Session REST endpoints ──

class SessionStartRequest(BaseModel):
    candidate_id: Optional[str] = "candidate"

class QuestionMarkerRequest(BaseModel):
    label: str


@http_router.post("/session/start")
async def session_start(req: SessionStartRequest):
    session_id = str(uuid.uuid4())[:12]
    cal_duration = _config.get("calibration_duration", 45)
    sessions[session_id] = {
        "aggregator": SessionAggregator(),
        "calibrator": PersonalBaselineCalibrator(calibration_duration_seconds=cal_duration),
        "report_gen": InterviewReportGenerator(),
        "candidate_id": req.candidate_id or "candidate",
        "created_at": time.time(),
    }
    logger.info("Session created: %s (candidate=%s)", session_id, req.candidate_id)
    return {"session_id": session_id}


@http_router.post("/session/{session_id}/question")
async def session_question(session_id: str, req: QuestionMarkerRequest):
    sess = sessions.get(session_id)
    if not sess:
        return JSONResponse({"error": "session not found"}, status_code=404)
    ts = time.time()
    sess["aggregator"].add_frame(
        DimensionScores(timestamp=ts), question_label=req.label
    )
    return {"ok": True, "timestamp": ts}


@http_router.get("/report/{session_id}")
async def get_report_json(session_id: str):
    sess = sessions.get(session_id)
    if not sess:
        return JSONResponse({"error": "session not found"}, status_code=404)
    aggregator: SessionAggregator = sess["aggregator"]
    report_gen: InterviewReportGenerator = sess["report_gen"]
    summary = aggregator.get_session_summary()
    timeline = aggregator.get_timeline_data()
    report = report_gen.generate_json(
        summary=summary,
        timeline_data=timeline,
        session_id=session_id,
        candidate_id=sess.get("candidate_id", "candidate"),
    )
    return report


@http_router.get("/report/{session_id}/pdf")
async def get_report_pdf(session_id: str):
    sess = sessions.get(session_id)
    if not sess:
        return JSONResponse({"error": "session not found"}, status_code=404)
    aggregator: SessionAggregator = sess["aggregator"]
    report_gen: InterviewReportGenerator = sess["report_gen"]
    summary = aggregator.get_session_summary()
    timeline = aggregator.get_timeline_data()
    pdf_bytes = report_gen.generate_pdf(
        summary=summary,
        timeline_data=timeline,
        session_id=session_id,
        candidate_id=sess.get("candidate_id", "candidate"),
        question_markers=aggregator.question_markers,
    )
    # Optionally save to disk
    report_dir = _config.get("report_output", "./reports/")
    try:
        os.makedirs(report_dir, exist_ok=True)
        pdf_path = os.path.join(report_dir, f"report_{session_id}.pdf")
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)
        logger.info("PDF report saved to %s", pdf_path)
    except Exception as e:
        logger.warning("Could not save PDF to disk: %s", e)

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=report_{session_id}.pdf"},
    )


def _decode_frame(data_url: str) -> Optional[np.ndarray]:
    """Decode a base64 data URL (from canvas.toDataURL) to a BGR numpy array."""
    try:
        # Strip "data:image/jpeg;base64," prefix
        if "," in data_url:
            data_url = data_url.split(",", 1)[1]
        img_bytes = base64.b64decode(data_url)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        logger.warning("Failed to decode frame: %s", e)
        return None


def _decode_audio(audio_b64: str, sample_rate: int = 44100) -> Optional[np.ndarray]:
    """Decode base64-encoded float32 PCM audio to numpy array, resample to 16kHz."""
    try:
        raw = base64.b64decode(audio_b64)
        waveform = np.frombuffer(raw, dtype=np.float32)
        if sample_rate != 16000:
            waveform = resample_audio(waveform, sample_rate, 16000)
        return waveform
    except Exception as e:
        logger.warning("Failed to decode audio: %s", e)
        return None


@ws_router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket, session_id: Optional[str] = Query(default=None)):
    await ws.accept()
    logger.info("WebSocket client connected (session_id=%s)", session_id)

    # Resolve session (or create ephemeral one)
    sess = sessions.get(session_id) if session_id else None
    if sess is None:
        cal_duration = _config.get("calibration_duration", 45)
        sess = {
            "aggregator": SessionAggregator(),
            "calibrator": PersonalBaselineCalibrator(calibration_duration_seconds=cal_duration),
            "report_gen": InterviewReportGenerator(),
            "candidate_id": "candidate",
            "created_at": time.time(),
        }
        if session_id:
            sessions[session_id] = sess

    aggregator: SessionAggregator = sess["aggregator"]
    calibrator: PersonalBaselineCalibrator = sess["calibrator"]

    # Shared state across receiver and processor tasks
    latest_frame: dict = {"data": None, "id": 0}
    audio_buffer: list[np.ndarray] = []
    last_audio_result: Optional[dict] = None
    last_visual_result: Optional[dict] = None
    last_gesture_result: Optional[dict] = None
    last_dimension_scores: Optional[DimensionScores] = None
    last_detections: dict = {"face": None, "body": None}
    processing = False
    frame_recv_count = 0
    frame_proc_count = 0
    frames_dropped = 0
    conn_start = time.time()
    last_log_time = time.time()
    calibrating = False
    last_notable_check: list = []

    loop = asyncio.get_event_loop()
    closed = False

    def _run_visual_inference(frame: np.ndarray) -> dict:
        """Run visual inference in a thread (blocking call)."""
        return _visual_detector.process_frame_with_detections(frame)

    def _run_audio_inference(waveform: np.ndarray) -> dict:
        """Run audio inference in a thread (blocking call)."""
        return _audio_detector.predict(waveform, 16000)

    async def _log_stats():
        nonlocal last_log_time
        now = time.time()
        if now - last_log_time >= 5.0:
            elapsed = now - conn_start
            logger.info(
                "[WS Stats] %.0fs | recv=%d proc=%d dropped=%d | "
                "proc_rate=%.1f fps | visual=%s audio=%s",
                elapsed, frame_recv_count, frame_proc_count, frames_dropped,
                frame_proc_count / max(elapsed, 1),
                "yes" if last_visual_result else "no",
                "yes" if last_audio_result else "no",
            )
            last_log_time = now

    async def receiver():
        """Receive messages from the client. Only keep the latest video frame."""
        nonlocal frame_recv_count, frames_dropped, closed, last_audio_result

        async def _run_audio_async(waveform):
            """Run audio inference without blocking the receiver."""
            nonlocal last_audio_result
            try:
                result = await loop.run_in_executor(
                    None, _run_audio_inference, waveform
                )
                last_audio_result = result
            except Exception as e:
                logger.warning("Audio inference error: %s", e)

        try:
            while not closed:
                raw = await ws.receive_text()
                msg = json.loads(raw)
                msg_type = msg.get("type")

                if msg_type == "video_frame":
                    frame_recv_count += 1
                    frame = _decode_frame(msg["data"])
                    if frame is not None:
                        old_id = latest_frame["id"]
                        latest_frame["data"] = frame
                        latest_frame["id"] = frame_recv_count
                        # Count frames we overwrote without processing
                        if processing and old_id > 0:
                            frames_dropped += 1

                elif msg_type == "audio_chunk":
                    sr = msg.get("sample_rate", 44100)
                    chunk = _decode_audio(msg["data"], sample_rate=sr)
                    if chunk is not None:
                        audio_buffer.append(chunk)
                        total_samples = sum(len(c) for c in audio_buffer)
                        # ~1s of audio at 16kHz after resampling
                        if total_samples >= 16000 and _audio_detector is not None:
                            waveform = np.concatenate(audio_buffer)
                            audio_buffer.clear()
                            # Fire-and-forget: don't block receiver while
                            # Wav2Vec2 runs (~1-3s on CPU)
                            asyncio.ensure_future(_run_audio_async(waveform))

                elif msg_type == "ping":
                    await ws.send_text(json.dumps({"type": "pong"}))

                elif msg_type == "calibration_start":
                    calibrator.start()
                    calibrating = True
                    await ws.send_text(json.dumps({
                        "type": "calibration_progress",
                        "progress": 0.0,
                    }))

                elif msg_type == "calibration_complete":
                    calibrating = False
                    success = calibrator.complete()
                    await ws.send_text(json.dumps({
                        "type": "calibration_done",
                        "success": success,
                    }))

                elif msg_type == "question_marker":
                    label = msg.get("label", "")
                    if label and last_dimension_scores is not None:
                        aggregator.add_frame(last_dimension_scores, question_label=label)
                    await ws.send_text(json.dumps({
                        "type": "question_marked",
                        "label": label,
                        "timestamp": time.time(),
                    }))

                elif msg_type == "session_end":
                    report_gen = sess["report_gen"]
                    summary = aggregator.get_session_summary()
                    timeline = aggregator.get_timeline_data()
                    report_json = report_gen.generate_json(
                        summary=summary,
                        timeline_data=timeline,
                        session_id=session_id or "ephemeral",
                        candidate_id=sess.get("candidate_id", "candidate"),
                    )
                    await ws.send_text(json.dumps({
                        "type": "report_ready",
                        "report_json": report_json,
                    }))

        except WebSocketDisconnect:
            logger.info("WebSocket receiver: client disconnected")
            closed = True
        except Exception as e:
            logger.error("WebSocket receiver error: %s", e)
            closed = True

    async def processor():
        """Process the latest frame and send results. Skips stale frames."""
        nonlocal processing, frame_proc_count, closed
        nonlocal last_visual_result, last_detections, last_gesture_result, last_dimension_scores
        nonlocal last_notable_check

        last_processed_id = 0

        def _run_gesture_inference(frame, pose_lms, face_bbox):
            """Run gesture analysis in thread (blocking call)."""
            if _gesture_analyzer is None:
                return None
            return _gesture_analyzer.analyze(frame, pose_landmarks=pose_lms, face_bbox=face_bbox)

        try:
            while not closed:
                # Wait for a new frame
                if latest_frame["data"] is None or latest_frame["id"] <= last_processed_id:
                    await asyncio.sleep(0.02)  # 50Hz check rate
                    await _log_stats()
                    continue

                # Grab the latest frame atomically
                frame = latest_frame["data"]
                fid = latest_frame["id"]
                last_processed_id = fid
                processing = True

                t0 = time.time()

                # Run visual inference in thread pool (non-blocking)
                visual_result = None
                detections = {"face": None, "body": None}
                if _visual_detector is not None:
                    try:
                        out = await loop.run_in_executor(
                            None, _run_visual_inference, frame
                        )
                        visual_result = out["prediction"]
                        detections = out["detections"]
                    except Exception as e:
                        logger.warning("Visual inference error (frame %d): %s", fid, e)

                last_visual_result = visual_result
                last_detections = detections

                # Run gesture analysis (hands + posture from existing pose)
                gesture_result = None
                if _gesture_analyzer is not None and visual_result is not None:
                    try:
                        pose_lms_raw = visual_result.get("pose_landmarks_raw")
                        face_bbox = None
                        if detections.get("face") and detections["face"].get("face_bbox"):
                            face_bbox = detections["face"]["face_bbox"]
                        gesture_result = await loop.run_in_executor(
                            None, _run_gesture_inference, frame, pose_lms_raw, face_bbox
                        )
                    except Exception as e:
                        logger.warning("Gesture inference error (frame %d): %s", fid, e)
                last_gesture_result = gesture_result

                # Baseline calibration: feed features
                if calibrating and visual_result is not None:
                    calibration_features = {}
                    for k in ["eye_contact_score", "head_nod_rate", "brow_furrow_intensity",
                              "valence", "arousal"]:
                        if k in visual_result:
                            calibration_features[k] = visual_result[k]
                    if gesture_result:
                        calibration_features.update(gesture_result)
                    if last_audio_result and "probabilities" in last_audio_result:
                        calibration_features["audio_stress_prob"] = last_audio_result["probabilities"].get("stressed", 0.5)
                    calibrator.add_features(calibration_features)

                    # Send calibration progress
                    try:
                        await ws.send_text(json.dumps({
                            "type": "calibration_progress",
                            "progress": round(calibrator.calibration_progress, 3),
                        }))
                    except Exception:
                        pass

                # Fuse (legacy)
                fused_result = None
                if _fusion is not None:
                    try:
                        fused_result = _fusion.fuse(
                            visual_result=visual_result,
                            audio_result=last_audio_result,
                        )
                    except Exception as e:
                        logger.warning("Fusion error: %s", e)

                # Multi-dimension fusion
                dim_scores = None
                if _dimension_fusion is not None:
                    try:
                        # Baseline-normalize if calibrated
                        norm_visual = visual_result
                        norm_gesture = gesture_result
                        if calibrator.is_calibrated and visual_result:
                            norm_visual = dict(visual_result)
                            for k in ["eye_contact_score", "head_nod_rate", "brow_furrow_intensity"]:
                                if k in norm_visual and norm_visual[k] is not None:
                                    stats = calibrator.baseline_stats.get(k)
                                    if stats and stats["std"] > 1e-9:
                                        norm_visual[k] = (float(norm_visual[k]) - stats["mean"]) / stats["std"]

                        dim_scores = _dimension_fusion.fuse(
                            visual_result=norm_visual,
                            gesture_result=norm_gesture,
                            audio_result=last_audio_result,
                            baseline_calibrated=calibrator.is_calibrated,
                        )
                        last_dimension_scores = dim_scores
                        aggregator.add_frame(dim_scores)
                    except Exception as e:
                        logger.warning("Dimension fusion error: %s", e)

                frame_proc_count += 1
                infer_ms = (time.time() - t0) * 1000

                processing = False

                # Build response — always include legacy "result" type fields
                # Strip non-serializable pose_landmarks_raw before sending
                visual_for_response = None
                if visual_result is not None:
                    visual_for_response = {
                        k: v for k, v in visual_result.items()
                        if k != "pose_landmarks_raw"
                    }

                response = {
                    "type": "result",
                    "frame_id": fid,
                    "timestamp": time.time(),
                    "visual": visual_for_response,
                    "audio": last_audio_result,
                    "fused": fused_result,
                    "detections": detections,
                    "latency_ms": round(infer_ms, 1),
                    "calibration_progress": round(calibrator.calibration_progress, 3),
                    "is_calibrated": calibrator.is_calibrated,
                }

                # Add dimension scores
                if dim_scores is not None:
                    response["dimension_scores"] = dim_scores.to_dict()

                # Add session stats every 30 frames
                if frame_proc_count % 30 == 0:
                    try:
                        response["session_stats"] = aggregator.get_session_summary()
                    except Exception:
                        pass

                # Add gesture results
                if gesture_result is not None:
                    response["gesture"] = gesture_result

                try:
                    await ws.send_text(json.dumps(response))
                except Exception:
                    closed = True
                    break

                # Check for notable moments
                if dim_scores is not None:
                    try:
                        new_notable = aggregator.get_notable_moments()
                        if len(new_notable) > len(last_notable_check):
                            for nm in new_notable[len(last_notable_check):]:
                                await ws.send_text(json.dumps({
                                    "type": "notable_moment",
                                    "dimension": nm["dimension"],
                                    "magnitude": nm["magnitude"],
                                    "timestamp": nm["timestamp"],
                                }))
                            last_notable_check = new_notable
                    except Exception:
                        pass

                await _log_stats()

        except Exception as e:
            logger.error("WebSocket processor error: %s", e)
            closed = True

    async def keepalive():
        """Send periodic pings to keep the WebSocket alive."""
        try:
            while not closed:
                await asyncio.sleep(15)
                if not closed:
                    try:
                        await ws.send_text(json.dumps({"type": "pong"}))
                    except Exception:
                        break
        except Exception:
            pass

    # Run receiver, processor, and keepalive concurrently
    try:
        await asyncio.gather(
            receiver(), processor(), keepalive(), return_exceptions=True
        )
    except Exception as e:
        logger.error("WebSocket gather error: %s", e)
    finally:
        elapsed = time.time() - conn_start
        logger.info(
            "[WS Done] %.0fs | recv=%d proc=%d dropped=%d",
            elapsed, frame_recv_count, frame_proc_count, frames_dropped,
        )
        try:
            await ws.close()
        except Exception:
            pass


# Static frontend serving is handled by the unified app (apps/static/behavior).
