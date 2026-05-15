"""
Visual Stress Detection Module
================================
Implements facial-landmark–based stress detection inspired by the StressID baseline
(Dalpaos et al., 2023) which uses Action Units (AUs) and gaze features.

Pipeline:
  Video frame → MediaPipe FaceMesh (468 landmarks) → AU-proxy features + gaze →
  temporal statistics → MLP classifier → stress score

Since OpenFace is not pip-installable, we use MediaPipe FaceMesh as a
cross-platform drop-in. We compute AU-like proxy features from landmark
geometry (distances & angles) that approximate the 18 FACS Action Units
used in the StressID baseline.

Also supports optional 3D body-pose features via MediaPipe Pose
(inspired by Richer et al., 2024 body-motion stress research).
"""

import logging
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

# Resolve model asset directory (models/ next to the package root)
_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Landmark indices for AU-proxy computation (MediaPipe FaceMesh 468 landmarks)
# ──────────────────────────────────────────────────────────────────────────────
# Eyebrow inner/outer raise (AU1/AU2)
LEFT_EYEBROW_INNER = 107
LEFT_EYEBROW_OUTER = 70
RIGHT_EYEBROW_INNER = 336
RIGHT_EYEBROW_OUTER = 300
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374
# Lip landmarks for AU-related mouth features
UPPER_LIP_TOP = 13
LOWER_LIP_BOTTOM = 14
LEFT_LIP_CORNER = 61
RIGHT_LIP_CORNER = 291
# Nose tip for reference distances
NOSE_TIP = 1
# Chin for face-height normalization
CHIN = 152
FOREHEAD = 10
# Cheek raise (AU6)
LEFT_CHEEK = 234
RIGHT_CHEEK = 454
# Nose wrinkle (AU9)
NOSE_BRIDGE = 6
# Eye corners for gaze proxy
LEFT_EYE_INNER = 133
LEFT_EYE_OUTER = 33
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263
# Iris centers (MediaPipe FaceMesh with refine_landmarks)
LEFT_IRIS_CENTER = 468  # index in refined landmarks
RIGHT_IRIS_CENTER = 473


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


class FacialFeatureExtractor:
    """
    Extracts AU-proxy features and gaze estimates from a single face frame
    using MediaPipe FaceMesh.

    Output feature vector (per-frame): 20 dimensions
      - 10 AU-proxy distances/ratios
      -  4 gaze-direction features
      -  6 head-pose proxy features (face orientation from landmarks)
    """

    FEATURE_DIM = 20

    def __init__(self, model_asset_path: Optional[str] = None):
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            FaceLandmarker,
            FaceLandmarkerOptions,
            RunningMode,
        )

        if model_asset_path is None:
            model_asset_path = os.path.join(_MODELS_DIR, "face_landmarker.task")

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_asset_path),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)
        self._frame_ts = 0  # monotonic timestamp for VIDEO mode
        self._last_face_bbox = None   # (x, y, w, h) normalized 0-1
        self._last_face_landmarks_norm = None  # list of {x, y} normalized

        # ── New feature tracking state ──
        # Eye contact EMA
        self._eye_contact_ema = 0.5
        self._eye_contact_alpha = 0.3
        # Head nod tracking (pitch proxy history)
        self._pitch_history = deque(maxlen=45)  # 3s at 15fps
        # Brow furrow 95th percentile tracker
        self._brow_max_seen = 0.01

    def get_last_detections(self) -> dict:
        """Return cached face bounding box and key landmarks (normalized 0-1)."""
        return {
            "face_bbox": self._last_face_bbox,
            "face_landmarks": self._last_face_landmarks_norm,
        }

    def extract(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract 20-D feature vector from a BGR frame.
        Returns None if no face detected.
        """
        import mediapipe as mp

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        self._frame_ts += 33  # ~30 fps in ms
        results = self.landmarker.detect_for_video(mp_image, self._frame_ts)

        if not results.face_landmarks:
            self._last_face_bbox = None
            self._last_face_landmarks_norm = None
            return None

        face_lms = results.face_landmarks[0]
        h, w, _ = frame_bgr.shape
        pts = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in face_lms])

        # Cache normalized bounding box
        xs = [lm.x for lm in face_lms]
        ys = [lm.y for lm in face_lms]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        pad = 0.05
        self._last_face_bbox = {
            "x": max(0, x_min - pad),
            "y": max(0, y_min - pad),
            "w": min(1, x_max - x_min + 2 * pad),
            "h": min(1, y_max - y_min + 2 * pad),
        }
        # Cache key landmarks (eyes, nose, mouth, brows) normalized
        key_indices = [
            FOREHEAD, CHIN, NOSE_TIP,
            LEFT_EYE_INNER, LEFT_EYE_OUTER, RIGHT_EYE_INNER, RIGHT_EYE_OUTER,
            LEFT_EYE_TOP, LEFT_EYE_BOTTOM, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM,
            LEFT_EYEBROW_INNER, LEFT_EYEBROW_OUTER, RIGHT_EYEBROW_INNER, RIGHT_EYEBROW_OUTER,
            UPPER_LIP_TOP, LOWER_LIP_BOTTOM, LEFT_LIP_CORNER, RIGHT_LIP_CORNER,
        ]
        self._last_face_landmarks_norm = [
            {"x": float(face_lms[i].x), "y": float(face_lms[i].y)}
            for i in key_indices if i < len(face_lms)
        ]

        # Normalize distances by face height
        face_height = _dist(pts[FOREHEAD], pts[CHIN]) + 1e-6

        features = []

        # ── AU-proxy features (10) ──
        # AU1: Inner brow raise (vertical distance brow-to-eye, normalized)
        left_inner_brow_raise = (pts[LEFT_EYE_TOP][1] - pts[LEFT_EYEBROW_INNER][1]) / face_height
        right_inner_brow_raise = (pts[RIGHT_EYE_TOP][1] - pts[RIGHT_EYEBROW_INNER][1]) / face_height
        features.append(left_inner_brow_raise)
        features.append(right_inner_brow_raise)

        # AU2: Outer brow raise
        left_outer_brow_raise = (pts[LEFT_EYE_TOP][1] - pts[LEFT_EYEBROW_OUTER][1]) / face_height
        right_outer_brow_raise = (pts[RIGHT_EYE_TOP][1] - pts[RIGHT_EYEBROW_OUTER][1]) / face_height
        features.append(left_outer_brow_raise)
        features.append(right_outer_brow_raise)

        # AU5/AU7: Eye openness (vertical eye aperture)
        left_eye_open = _dist(pts[LEFT_EYE_TOP], pts[LEFT_EYE_BOTTOM]) / face_height
        right_eye_open = _dist(pts[RIGHT_EYE_TOP], pts[RIGHT_EYE_BOTTOM]) / face_height
        features.append(left_eye_open)
        features.append(right_eye_open)

        # AU20/AU25: Lip separation & stretch
        lip_opening = _dist(pts[UPPER_LIP_TOP], pts[LOWER_LIP_BOTTOM]) / face_height
        lip_width = _dist(pts[LEFT_LIP_CORNER], pts[RIGHT_LIP_CORNER]) / face_height
        features.append(lip_opening)
        features.append(lip_width)

        # AU6: Cheek raise (vertical position of cheek relative to nose)
        left_cheek_raise = (pts[NOSE_TIP][1] - pts[LEFT_CHEEK][1]) / face_height
        right_cheek_raise = (pts[NOSE_TIP][1] - pts[RIGHT_CHEEK][1]) / face_height
        features.append(left_cheek_raise)
        features.append(right_cheek_raise)

        # ── Gaze features (4) ──
        if len(pts) > RIGHT_IRIS_CENTER:
            left_iris = pts[LEFT_IRIS_CENTER]
            right_iris = pts[RIGHT_IRIS_CENTER]
            left_eye_center = (pts[LEFT_EYE_INNER] + pts[LEFT_EYE_OUTER]) / 2.0
            right_eye_center = (pts[RIGHT_EYE_INNER] + pts[RIGHT_EYE_OUTER]) / 2.0
            left_gaze = (left_iris[:2] - left_eye_center[:2]) / face_height
            right_gaze = (right_iris[:2] - right_eye_center[:2]) / face_height
            features.extend(left_gaze.tolist())
            features.extend(right_gaze.tolist())
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])

        # ── Head pose proxy (6) ──
        # Use nose, forehead, chin, ears to approximate yaw/pitch/roll
        nose = pts[NOSE_TIP]
        forehead = pts[FOREHEAD]
        chin = pts[CHIN]
        # Pitch proxy: angle of forehead-chin line
        pitch_proxy = (forehead[1] - chin[1]) / face_height
        # Yaw proxy: asymmetry of nose position relative to eye midline
        eye_mid_x = (pts[LEFT_EYE_INNER][0] + pts[RIGHT_EYE_INNER][0]) / 2.0
        yaw_proxy = (nose[0] - eye_mid_x) / face_height
        # Roll proxy: tilt of the eye line
        eye_line = pts[RIGHT_EYE_OUTER] - pts[LEFT_EYE_OUTER]
        roll_proxy = float(np.arctan2(eye_line[1], eye_line[0]))
        # Z-depth of nose (relative to face plane)
        z_proxy = nose[2] / face_height
        # Forehead-chin z difference
        fz_proxy = (forehead[2] - chin[2]) / face_height
        # Inter-eye distance (useful for scale sanity)
        inter_eye = _dist(pts[LEFT_EYE_OUTER], pts[RIGHT_EYE_OUTER]) / face_height

        features.extend([pitch_proxy, yaw_proxy, roll_proxy, z_proxy, fz_proxy, inter_eye])

        return np.array(features, dtype=np.float32)

    def compute_extra_features(self, feature_vec: np.ndarray) -> dict:
        """
        Compute eye_contact_score, head_nod_rate, brow_furrow_intensity
        from the 20-D feature vector.
        """
        # ── Eye contact score ──
        # Gaze features are indices 10-13 (left_gaze_x, left_gaze_y, right_gaze_x, right_gaze_y)
        gaze_x = (feature_vec[10] + feature_vec[12]) / 2.0
        gaze_y = (feature_vec[11] + feature_vec[13]) / 2.0
        gaze_offset = float(np.sqrt(gaze_x**2 + gaze_y**2))
        max_gaze_offset = 0.3
        raw_eye_contact = 1.0 - (gaze_offset / max_gaze_offset)
        raw_eye_contact = float(np.clip(raw_eye_contact, 0.0, 1.0))
        self._eye_contact_ema = (
            self._eye_contact_alpha * raw_eye_contact
            + (1.0 - self._eye_contact_alpha) * self._eye_contact_ema
        )
        eye_contact_score = float(np.clip(self._eye_contact_ema, 0.0, 1.0))

        # ── Head nod rate ──
        # Feature 14 is pitch_proxy (index 14 in the 20-D vector)
        pitch = float(feature_vec[14])
        self._pitch_history.append(pitch)
        head_nod_rate = 0.0
        if len(self._pitch_history) >= 5:
            ph = np.array(list(self._pitch_history))
            diff = np.diff(ph)
            # 3-frame moving average of diff
            if len(diff) >= 3:
                kernel = np.ones(3) / 3.0
                smoothed = np.convolve(diff, kernel, mode='valid')
                zero_crossings = int(np.sum(np.abs(np.diff(np.sign(smoothed))) > 0))
                window_seconds = len(self._pitch_history) / 15.0
                head_nod_rate = (zero_crossings / max(window_seconds, 0.1)) * 60.0

        # ── Brow furrow intensity ──
        # Features 0,1,2,3 are inner/outer brow proxies
        brow_mean = float(np.mean(feature_vec[0:4]))
        brow_abs = abs(brow_mean)
        if brow_abs > self._brow_max_seen:
            self._brow_max_seen = brow_abs
        p95 = self._brow_max_seen
        brow_furrow_intensity = float(np.clip(brow_abs / max(p95, 1e-6), 0.0, 1.0))

        return {
            "eye_contact_score": round(eye_contact_score, 4),
            "head_nod_rate": round(head_nod_rate, 2),
            "brow_furrow_intensity": round(brow_furrow_intensity, 4),
        }


class BodyPoseExtractor:
    """
    Optional body-pose feature extractor using MediaPipe Pose.

    Extracts 12-D feature vector:
      - Shoulder slopes and widths
      - Upper-body lean angles
      - Arm position features
      - Movement velocity (between frames)

    Inspired by Richer et al. (2024) body-motion stress detection research.
    """

    FEATURE_DIM = 12

    def __init__(self, model_asset_path: Optional[str] = None):
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            PoseLandmarker,
            PoseLandmarkerOptions,
            RunningMode,
        )

        if model_asset_path is None:
            model_asset_path = os.path.join(_MODELS_DIR, "pose_landmarker.task")

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_asset_path),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = PoseLandmarker.create_from_options(options)
        self._frame_ts = 0
        self._prev_landmarks = None
        self._last_pose_landmarks_norm = None  # list of {x,y} normalized
        self._last_pose_bbox = None

    def get_last_detections(self) -> dict:
        """Return cached body bounding box and key landmarks (normalized 0-1)."""
        return {
            "body_bbox": self._last_pose_bbox,
            "body_landmarks": self._last_pose_landmarks_norm,
        }

    def extract(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract 12-D body-pose feature vector from a BGR frame.
        Returns None if no body detected.
        """
        import mediapipe as mp

        # Downsample for speed — pose doesn't need high resolution
        h_orig, w_orig = frame_bgr.shape[:2]
        scale = 320 / max(h_orig, w_orig)
        if scale < 1.0:
            small = cv2.resize(frame_bgr, (int(w_orig * scale), int(h_orig * scale)),
                               interpolation=cv2.INTER_LINEAR)
        else:
            small = frame_bgr

        frame_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        self._frame_ts += 33
        results = self.landmarker.detect_for_video(mp_image, self._frame_ts)

        if not results.pose_landmarks:
            self._prev_landmarks = None
            self._last_pose_landmarks_norm = None
            self._last_pose_bbox = None
            return None

        lm = results.pose_landmarks[0]
        h, w, _ = small.shape
        pts = np.array([[l.x * w, l.y * h, l.z * w] for l in lm])

        # Cache normalized body bounding box and key landmarks
        xs = [l.x for l in lm]
        ys = [l.y for l in lm]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        pad = 0.03
        self._last_pose_bbox = {
            "x": max(0, x_min - pad),
            "y": max(0, y_min - pad),
            "w": min(1, x_max - x_min + 2 * pad),
            "h": min(1, y_max - y_min + 2 * pad),
        }
        # Upper body: shoulders(11,12), elbows(13,14), wrists(15,16), hips(23,24)
        key_body = [11, 12, 13, 14, 15, 16, 23, 24, 0]  # 0=nose
        self._last_pose_landmarks_norm = [
            {"x": float(lm[i].x), "y": float(lm[i].y)}
            for i in key_body if i < len(lm)
        ]

        features = []

        # Shoulder landmarks
        left_shoulder = pts[11]
        right_shoulder = pts[12]
        # Hip landmarks
        left_hip = pts[23]
        right_hip = pts[24]
        # Elbow landmarks
        left_elbow = pts[13]
        right_elbow = pts[14]

        shoulder_width = _dist(left_shoulder, right_shoulder)
        norm = shoulder_width + 1e-6

        # 1-2: Shoulder slope (y-difference normalized)
        shoulder_dy = (left_shoulder[1] - right_shoulder[1]) / norm
        shoulder_dz = (left_shoulder[2] - right_shoulder[2]) / norm
        features.append(shoulder_dy)
        features.append(shoulder_dz)

        # 3: Torso lean (midpoint shoulders vs midpoint hips)
        mid_shoulder = (left_shoulder + right_shoulder) / 2.0
        mid_hip = (left_hip + right_hip) / 2.0
        torso_lean_x = (mid_shoulder[0] - mid_hip[0]) / norm
        torso_lean_z = (mid_shoulder[2] - mid_hip[2]) / norm
        features.append(torso_lean_x)
        features.append(torso_lean_z)

        # 5-6: Arm positions (elbow relative to shoulder)
        left_arm_drop = (left_elbow[1] - left_shoulder[1]) / norm
        right_arm_drop = (right_elbow[1] - right_shoulder[1]) / norm
        features.append(left_arm_drop)
        features.append(right_arm_drop)

        # 7-8: Arm lateral spread
        left_arm_spread = (left_elbow[0] - left_shoulder[0]) / norm
        right_arm_spread = (right_elbow[0] - right_shoulder[0]) / norm
        features.append(left_arm_spread)
        features.append(right_arm_spread)

        # 9-12: Movement velocity (frame-to-frame displacement of key joints)
        curr_key = np.array([
            mid_shoulder[:2], mid_hip[:2],
            left_elbow[:2], right_elbow[:2]
        ])
        if self._prev_landmarks is not None:
            velocity = np.mean(np.linalg.norm(curr_key - self._prev_landmarks, axis=1))
            max_vel = np.max(np.linalg.norm(curr_key - self._prev_landmarks, axis=1))
        else:
            velocity = 0.0
            max_vel = 0.0
        self._prev_landmarks = curr_key

        features.append(float(velocity / norm))
        features.append(float(max_vel / norm))

        # 11-12: Shoulder-to-hip distance ratio and torso compactness
        torso_height = _dist(mid_shoulder, mid_hip)
        features.append(torso_height / norm)
        compactness = (_dist(left_shoulder, left_hip) + _dist(right_shoulder, right_hip)) / (2.0 * norm)
        features.append(compactness)

        return np.array(features, dtype=np.float32)


class TemporalAggregator:
    """
    Aggregates per-frame features over a sliding window and computes
    temporal statistics (mean, std, min, max, delta) as in the
    StressID baseline processing pipeline.
    """

    def __init__(self, window_size: int = 30, feature_dim: int = 20):
        self.window = deque(maxlen=window_size)
        self.feature_dim = feature_dim

    def add_frame(self, features: np.ndarray):
        self.window.append(features)

    def compute_stats(self) -> Optional[np.ndarray]:
        """
        Compute temporal statistics over the window.
        Returns a (feature_dim * 4)-D vector: [mean, std, min, max].
        Returns None if window is empty.
        """
        if len(self.window) == 0:
            return None

        arr = np.stack(list(self.window), axis=0)  # (T, D)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        mn = arr.min(axis=0)
        mx = arr.max(axis=0)

        return np.concatenate([mean, std, mn, mx]).astype(np.float32)


class VisualStressMLP(nn.Module):
    """
    MLP classifier for stress detection from temporal AU/gaze/pose features.

    Modeled after the StressID baseline MLP that classifies stress from
    facial AU temporal statistics.

    Default input dimension:
      - Face features: 20 dims × 4 stats = 80
      - Body features (optional): 12 dims × 4 stats = 48
      - Total: 80 or 128
    """

    def __init__(self, input_dim: int = 80, hidden_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


# ──────────────────────────────────────────────────────────────────────────────
# HSEmotion-based Emotion Classifier (pretrained on AffectNet)
# ──────────────────────────────────────────────────────────────────────────────


class HSEmotionClassifier:
    """
    Wraps the HSEmotion pretrained EfficientNet-B0 model (trained on AffectNet ~450K images)
    for real-time facial emotion recognition with valence/arousal regression.

    Model: enet_b0_8_va_mtl
      - 8 emotion classes: Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise
      - 2 continuous outputs: valence (-1..+1), arousal (-1..+1)

    Stress mapping:
      stress = f(negative emotions, high arousal, low valence)
      - Happiness + high valence → NOT stressed
      - Fear/Anger/Sadness + high arousal + low valence → stressed
    """

    EMOTION_LABELS = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']

    STRESS_WEIGHTS = {
        'Anger': 0.75, 'Contempt': 0.40, 'Disgust': 0.50, 'Fear': 0.85,
        'Happiness': -0.60, 'Neutral': -0.10, 'Sadness': 0.70, 'Surprise': 0.15,
    }

    def __init__(self, model_name: str = 'enet_b0_8_va_mtl', device: str = 'cpu'):
        from hsemotion.facial_emotions import HSEmotionRecognizer
        import hsemotion.facial_emotions as _hse_mod

        # Patch torch.load for hsemotion (it loads full model objects, not just state_dicts)
        # PyTorch 2.6+ defaults weights_only=True which blocks this
        _original_torch_load = torch.load
        def _patched_load(f, *args, **kwargs):
            kwargs.setdefault('weights_only', False)
            return _original_torch_load(f, *args, **kwargs)
        torch.load = _patched_load
        try:
            self.device = device
            self.fer = HSEmotionRecognizer(model_name=model_name, device=device)
        finally:
            torch.load = _original_torch_load

        self._last_result = None
        logger.info("HSEmotion model '%s' loaded on %s", model_name, device)

    def predict(self, face_rgb: np.ndarray) -> dict:
        """Run emotion prediction on an RGB face crop (any size)."""
        emotion_label, scores = self.fer.predict_emotions(face_rgb, logits=False)

        if self.fer.is_mtl:
            emotion_probs = scores[:-2]
            valence = float(scores[-2])
            arousal = float(scores[-1])
        else:
            emotion_probs = scores
            valence = 0.0
            arousal = 0.0

        emotions = {}
        for i, label in enumerate(self.EMOTION_LABELS):
            if i < len(emotion_probs):
                emotions[label] = round(float(emotion_probs[i]), 4)

        # Stress from weighted emotion probabilities
        emotion_stress = sum(emotions.get(e, 0) * w for e, w in self.STRESS_WEIGHTS.items())
        emotion_stress_norm = float(np.clip((emotion_stress + 0.6) / 1.45, 0, 1))

        # Low valence → stressed, high arousal → stressed
        valence_stress = float(np.clip((1.0 - valence) / 2.0, 0, 1))
        arousal_stress = float(np.clip((arousal + 1.0) / 2.0, 0, 1))

        stress_prob = float(np.clip(
            emotion_stress_norm * 0.50 + valence_stress * 0.30 + arousal_stress * 0.20,
            0.01, 0.99,
        ))

        self._last_result = {
            "label": "stressed" if stress_prob > 0.5 else "not_stressed",
            "confidence": max(stress_prob, 1.0 - stress_prob),
            "probabilities": {"not_stressed": 1.0 - stress_prob, "stressed": stress_prob},
            "emotion": emotion_label,
            "emotions": emotions,
            "valence": round(valence, 3),
            "arousal": round(arousal, 3),
        }
        return self._last_result

    def get_last_result(self) -> Optional[dict]:
        return self._last_result


class VisualStressDetector:
    """
    End-to-end visual stress detector combining:
      - MediaPipe FaceMesh → AU-proxy + gaze features (20-D per frame)
      - Optional MediaPipe Pose → body-pose features (12-D per frame)
      - Temporal aggregation → statistics over sliding window
      - MLP classifier → binary stress prediction

    Mirrors the StressID baseline (Dalpaos et al., 2023) approach
    but uses MediaPipe instead of OpenFace.
    """

    LABELS = ["not_stressed", "stressed"]

    def __init__(
        self,
        weights_path: Optional[str] = None,
        use_body_pose: bool = False,
        window_size: int = 30,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_body_pose = use_body_pose

        # Feature extractors
        self.face_extractor = FacialFeatureExtractor()
        self.body_extractor = BodyPoseExtractor() if use_body_pose else None

        # Temporal aggregation
        face_dim = FacialFeatureExtractor.FEATURE_DIM
        body_dim = BodyPoseExtractor.FEATURE_DIM if use_body_pose else 0
        total_dim = face_dim + body_dim
        self.temporal = TemporalAggregator(
            window_size=window_size, feature_dim=total_dim
        )

        # MLP classifier (kept for when trained weights are available)
        mlp_input = total_dim * 4  # mean, std, min, max
        self.classifier = VisualStressMLP(input_dim=mlp_input).to(self.device)

        if weights_path:
            logger.info("Loading visual stress MLP weights from %s", weights_path)
            state = torch.load(weights_path, map_location=self.device, weights_only=True)
            self.classifier.load_state_dict(state, strict=False)
        self.classifier.eval()

        # ── HSEmotion pretrained classifier (replaces heuristic) ──
        logger.info("Loading HSEmotion pretrained emotion model...")
        self.emotion_classifier = HSEmotionClassifier(
            model_name='enet_b0_8_va_mtl', device=self.device
        )

        # ── Parallelization & frame-skipping for body pose ──
        self._pose_pool = ThreadPoolExecutor(max_workers=1) if use_body_pose else None
        self._pose_frame_counter = 0
        self._pose_skip_interval = 2  # run pose every Nth frame
        self._cached_body_feats = None

    def process_frame(self, frame_bgr: np.ndarray) -> Optional[dict]:
        """
        Process a single video frame. Accumulates features in the temporal
        window and returns a prediction once enough frames are available.

        Args:
            frame_bgr: BGR image (numpy array from cv2).

        Returns:
            dict with prediction or None if face not detected or window not full.
        """
        # Start body pose extraction in parallel (on every Nth frame)
        pose_future = None
        if self.use_body_pose and self.body_extractor is not None:
            self._pose_frame_counter += 1
            if self._pose_frame_counter % self._pose_skip_interval == 0:
                pose_future = self._pose_pool.submit(self.body_extractor.extract, frame_bgr)

        # Run face extraction on main thread
        face_feats = self.face_extractor.extract(frame_bgr)
        if face_feats is None:
            # Wait for pose future to avoid dangling threads
            if pose_future is not None:
                pose_future.result()
            return None

        # Collect body pose result (from parallel future or cache)
        if pose_future is not None:
            body_feats = pose_future.result()
            if body_feats is not None:
                self._cached_body_feats = body_feats
        body_feats = self._cached_body_feats

        if self.use_body_pose:
            if body_feats is not None:
                combined = np.concatenate([face_feats, body_feats])
            else:
                combined = np.concatenate([
                    face_feats,
                    np.zeros(BodyPoseExtractor.FEATURE_DIM, dtype=np.float32),
                ])
        else:
            combined = face_feats

        self.temporal.add_frame(combined)

        # ── Use HSEmotion on the face crop ──
        result = self._classify_with_hsemotion(frame_bgr)

        # Add new fine-grained features from the 20-D face vector
        if result is not None:
            extra = self.face_extractor.compute_extra_features(face_feats)
            result["eye_contact_score"] = extra["eye_contact_score"]
            result["head_nod_rate"] = extra["head_nod_rate"]
            result["brow_furrow_intensity"] = extra["brow_furrow_intensity"]

        # Expose raw pose landmarks for GestureAnalyzer
        raw_pose = None
        if self.use_body_pose and self.body_extractor is not None:
            raw_pose = self._get_raw_pose_landmarks()
        if result is not None:
            result["pose_landmarks_raw"] = raw_pose

        return result

    def _classify_with_hsemotion(self, frame_bgr: np.ndarray) -> Optional[dict]:
        """Crop the face using MediaPipe bbox, run through HSEmotion pretrained model."""
        face_det = self.face_extractor.get_last_detections()
        bbox = face_det.get("face_bbox")
        if bbox is None:
            return None

        h, w = frame_bgr.shape[:2]
        x1 = max(0, int(bbox["x"] * w))
        y1 = max(0, int(bbox["y"] * h))
        x2 = min(w, int((bbox["x"] + bbox["w"]) * w))
        y2 = min(h, int((bbox["y"] + bbox["h"]) * h))

        if x2 - x1 < 20 or y2 - y1 < 20:
            return None

        face_crop = frame_bgr[y1:y2, x1:x2]
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

        result = self.emotion_classifier.predict(face_rgb)
        logger.debug(
            "HSEmotion: %s (stress=%.2f, val=%.2f, aro=%.2f)",
            result["emotion"], result["probabilities"]["stressed"],
            result["valence"], result["arousal"],
        )
        return result

    def get_temporal_features(self) -> Optional[np.ndarray]:
        """Return the current temporal feature statistics vector."""
        return self.temporal.compute_stats()

    def _get_raw_pose_landmarks(self):
        """Return the raw pose landmarks from BodyPoseExtractor for reuse."""
        if self.body_extractor is None:
            return None
        # Access cached landmarks from the last detect call
        det = self.body_extractor.get_last_detections()
        return det.get("body_landmarks")

    def process_frame_with_detections(self, frame_bgr: np.ndarray) -> dict:
        """
        Process a frame and return prediction + detection metadata for overlay rendering.
        Always returns a dict (never None).
        """
        result = self.process_frame(frame_bgr)

        detections = {"face": None, "body": None}
        face_det = self.face_extractor.get_last_detections()
        if face_det["face_bbox"]:
            detections["face"] = face_det

        if self.use_body_pose and self.body_extractor is not None:
            body_det = self.body_extractor.get_last_detections()
            if body_det["body_bbox"]:
                detections["body"] = body_det

        return {
            "prediction": result,
            "detections": detections,
        }
