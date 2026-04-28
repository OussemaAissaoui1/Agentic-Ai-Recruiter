"""
Body Language / Gesture Analysis Module
=========================================
Uses MediaPipe Hands for hand gesture analysis and reuses existing
MediaPipe Pose landmarks (from VisualStressDetector) for posture metrics.

IMPORTANT: Does NOT create a second MediaPipe Pose instance. Pose landmarks
are passed into analyze() from the existing BodyPoseExtractor.
"""

import logging
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import cv2
import numpy as np

_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

logger = logging.getLogger(__name__)


class GestureAnalyzer:
    """
    Analyzes hand gestures and body posture for behavioral profiling.

    Hand-based metrics (requires MediaPipe Hands):
      - self_touch_score
      - fidget_score
      - gesture_rate

    Posture metrics (reuses existing MediaPipe Pose landmarks):
      - posture_openness
      - forward_lean
      - postural_stability
    """

    def __init__(self, target_fps: int = 15):
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            HandLandmarker,
            HandLandmarkerOptions,
            RunningMode,
        )

        hand_model_path = os.path.join(_MODELS_DIR, "hand_landmarker.task")

        # MediaPipe Hands may not have a .task file; fall back to solutions API
        self._use_task_api = os.path.exists(hand_model_path)
        self._hand_landmarker = None
        self._hand_solutions = None
        self._frame_ts = 0

        if self._use_task_api:
            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=hand_model_path),
                running_mode=RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._hand_landmarker = HandLandmarker.create_from_options(options)
        else:
            import mediapipe as mp
            self._hand_solutions = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

        self._target_fps = target_fps

        # Thread pool for parallel hand inference
        self._pool = ThreadPoolExecutor(max_workers=1)

        # Frame skipping
        self._frame_counter = 0
        self._skip_interval = 2

        # ── Smoothing / history buffers ──
        # Self-touch EMA (alpha=0.15, ~2s smoothing at 15fps)
        self._self_touch_ema = 0.0
        self._self_touch_alpha = 0.15

        # Fidget: hand positions over 5s window (75 frames at 15fps)
        self._hand_positions: deque = deque(maxlen=75)

        # Gesture rate: displacements over 10s window (150 frames at 15fps)
        self._gesture_frames: deque = deque(maxlen=150)
        self._prev_hand_center: Optional[np.ndarray] = None

        # Postural stability: shoulder midpoint over 10s window
        self._shoulder_history: deque = deque(maxlen=150)

        # Cached result (for skipped frames)
        self._cached_result: dict = {
            "self_touch_score": None,
            "fidget_score": None,
            "gesture_rate": None,
            "posture_openness": None,
            "forward_lean": None,
            "postural_stability": None,
        }

    def analyze(
        self,
        frame_bgr: np.ndarray,
        pose_landmarks=None,
        face_bbox: Optional[dict] = None,
    ) -> dict:
        """
        Analyze a single frame for gesture and posture metrics.

        Parameters
        ----------
        frame_bgr : np.ndarray
            BGR image.
        pose_landmarks : list or None
            Raw MediaPipe Pose landmarks from BodyPoseExtractor (normalized).
            Used for posture_openness, forward_lean, postural_stability.
        face_bbox : dict or None
            {"x", "y", "w", "h"} normalized 0-1. Used for self-touch detection.

        Returns
        -------
        dict with keys: self_touch_score, fidget_score, gesture_rate,
             posture_openness, forward_lean, postural_stability
        """
        self._frame_counter += 1

        # Frame skipping for hands
        run_hands = (self._frame_counter % self._skip_interval == 0)

        hand_landmarks_list = None
        if run_hands:
            hand_landmarks_list = self._detect_hands(frame_bgr)

        # Update hand-based metrics
        h, w = frame_bgr.shape[:2]
        if hand_landmarks_list is not None:
            self._update_hand_metrics(hand_landmarks_list, face_bbox, w, h)

        # Update posture metrics from pose landmarks
        self._update_posture_metrics(pose_landmarks)

        return dict(self._cached_result)

    def _detect_hands(self, frame_bgr: np.ndarray):
        """Run MediaPipe Hands on a downscaled frame. Returns list of hand landmarks."""
        h_orig, w_orig = frame_bgr.shape[:2]
        scale = 320 / max(h_orig, w_orig)
        if scale < 1.0:
            small = cv2.resize(
                frame_bgr,
                (int(w_orig * scale), int(h_orig * scale)),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            small = frame_bgr

        frame_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        if self._use_task_api and self._hand_landmarker is not None:
            import mediapipe as mp
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            self._frame_ts += 33
            results = self._hand_landmarker.detect_for_video(mp_image, self._frame_ts)
            if not results.hand_landmarks:
                return []
            # Convert to list of list-of-dicts [{x,y,z}, ...]
            hands = []
            for hand_lms in results.hand_landmarks:
                hands.append([{"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand_lms])
            return hands
        elif self._hand_solutions is not None:
            results = self._hand_solutions.process(frame_rgb)
            if not results.multi_hand_landmarks:
                return []
            hands = []
            for hand_lms in results.multi_hand_landmarks:
                hands.append([{"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand_lms.landmark])
            return hands
        return []

    def _update_hand_metrics(self, hand_landmarks_list, face_bbox, frame_w, frame_h):
        """Update self_touch, fidget, gesture_rate from detected hands."""
        if not hand_landmarks_list:
            # Decay self-touch via EMA toward 0
            self._self_touch_ema *= (1.0 - self._self_touch_alpha)
            self._cached_result["self_touch_score"] = round(self._self_touch_ema, 4)
            return

        # Collect all hand landmark positions (normalized)
        all_positions = []
        for hand_lms in hand_landmarks_list:
            for lm in hand_lms:
                all_positions.append(np.array([lm["x"], lm["y"]]))

        all_positions = np.array(all_positions)  # (N, 2)

        # ── Self-touch score ──
        if face_bbox is not None:
            nose_y = face_bbox["y"] + face_bbox["h"] * 0.5
            face_h = face_bbox["h"]
            threshold = 0.15  # 15% of frame height
            # Check if any hand landmark is within threshold of face vertical range
            face_top = nose_y - face_h * 0.4
            face_bottom = nose_y + face_h * 0.4
            face_cx = face_bbox["x"] + face_bbox["w"] * 0.5
            face_radius_x = face_bbox["w"] * 0.6

            near_face = 0
            for pos in all_positions:
                in_y = face_top - threshold <= pos[1] <= face_bottom + threshold
                in_x = abs(pos[0] - face_cx) < face_radius_x + threshold
                if in_y and in_x:
                    near_face += 1

            raw_touch = min(near_face / max(len(all_positions), 1), 1.0)
            self._self_touch_ema = (
                self._self_touch_alpha * raw_touch
                + (1.0 - self._self_touch_alpha) * self._self_touch_ema
            )
            self._cached_result["self_touch_score"] = round(
                float(np.clip(self._self_touch_ema, 0.0, 1.0)), 4
            )
        else:
            self._cached_result["self_touch_score"] = None

        # ── Fidget score ──
        hand_center = all_positions.mean(axis=0)
        self._hand_positions.append(hand_center)

        if len(self._hand_positions) >= 10:
            pos_array = np.array(list(self._hand_positions))
            variance = np.var(pos_array, axis=0).sum()  # Total positional variance
            # Normalize by frame area (positions are 0-1 normalized already)
            fidget_raw = float(np.clip(variance * 100.0, 0.0, 1.0))
            self._cached_result["fidget_score"] = round(fidget_raw, 4)
        else:
            self._cached_result["fidget_score"] = 0.0

        # ── Gesture rate ──
        if self._prev_hand_center is not None:
            displacement = float(np.linalg.norm(hand_center - self._prev_hand_center))
            is_gesture = displacement > 0.05  # 5% of frame width (normalized coords)
            self._gesture_frames.append(1.0 if is_gesture else 0.0)
        else:
            self._gesture_frames.append(0.0)
        self._prev_hand_center = hand_center

        if len(self._gesture_frames) >= 10:
            window_seconds = len(self._gesture_frames) / max(self._target_fps, 1)
            gesture_count = sum(self._gesture_frames)
            gpm = (gesture_count / max(window_seconds, 0.1)) * 60.0
            self._cached_result["gesture_rate"] = round(
                float(np.clip(gpm, 0.0, 120.0)), 2
            )
        else:
            self._cached_result["gesture_rate"] = 0.0

    def _update_posture_metrics(self, pose_landmarks):
        """Update posture metrics from existing MediaPipe Pose landmarks."""
        if pose_landmarks is None or len(pose_landmarks) < 25:
            return

        # Pose landmark indices: nose=0, left_shoulder=11, right_shoulder=12,
        # left_wrist=15, right_wrist=16
        try:
            nose = pose_landmarks[0]
            l_shoulder = pose_landmarks[11]
            r_shoulder = pose_landmarks[12]
            l_wrist = pose_landmarks[15]
            r_wrist = pose_landmarks[16]
        except (IndexError, TypeError):
            return

        def _get_xyz(lm):
            if hasattr(lm, "x"):
                return np.array([lm.x, lm.y, getattr(lm, "z", 0.0)])
            elif isinstance(lm, dict):
                return np.array([lm["x"], lm["y"], lm.get("z", 0.0)])
            return None

        nose_pt = _get_xyz(nose)
        ls_pt = _get_xyz(l_shoulder)
        rs_pt = _get_xyz(r_shoulder)
        lw_pt = _get_xyz(l_wrist)
        rw_pt = _get_xyz(r_wrist)

        if any(p is None for p in [nose_pt, ls_pt, rs_pt, lw_pt, rw_pt]):
            return

        shoulder_mid = (ls_pt + rs_pt) / 2.0
        shoulder_width = float(np.linalg.norm(ls_pt[:2] - rs_pt[:2])) + 1e-6

        # ── Posture openness ──
        wrist_dist = float(np.abs(lw_pt[0] - rw_pt[0]))
        openness = float(np.clip(wrist_dist / shoulder_width, 0.0, 1.0))
        self._cached_result["posture_openness"] = round(openness, 4)

        # ── Forward lean ──
        # Nose z relative to shoulder midpoint z
        z_diff = nose_pt[2] - shoulder_mid[2]
        lean = float(np.clip(z_diff * 5.0, -1.0, 1.0))  # Scale and clip
        self._cached_result["forward_lean"] = round(lean, 4)

        # ── Postural stability ──
        self._shoulder_history.append(shoulder_mid[:2].copy())
        if len(self._shoulder_history) >= 10:
            sh_array = np.array(list(self._shoulder_history))
            variance = np.var(sh_array, axis=0).sum()
            # Inverse: low variance = high stability
            stability = float(np.clip(1.0 - variance * 500.0, 0.0, 1.0))
            self._cached_result["postural_stability"] = round(stability, 4)
        else:
            self._cached_result["postural_stability"] = 0.5

    def get_hand_landmarks_for_overlay(self):
        """Return the last detected hand landmarks for visualization."""
        return getattr(self, "_last_hand_landmarks", None)
