"""
Multimodal Fusion Module
=========================
Combines visual (face/body) and audio (voice) stress scores
into a unified stress assessment.

Supports three fusion strategies:
  1. **Late Fusion (Weighted Average)** – Weighted combination of independent
     model output probabilities. Simple, no retraining needed.
  2. **Late Fusion (Meta-Classifier)** – An MLP trained on stacked model outputs.
  3. **Early Fusion** – Concatenation of raw feature vectors from both modalities
     into a single classifier (requires end-to-end training).

Also provides MultiDimensionFusion for four-dimensional behavioral profiling.

Reference: StressID (Dalpaos et al., 2023) demonstrates that multimodal
fusion improves over unimodal baselines.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class FusionStrategy(str, Enum):
    WEIGHTED_AVERAGE = "weighted_average"
    META_CLASSIFIER = "meta_classifier"
    EARLY_FUSION = "early_fusion"


class WeightedAverageFusion:
    """
    Late fusion via weighted average of stress probabilities.

    stress_score = w_visual * p_visual + w_audio * p_audio

    If one modality is unavailable, falls back to the available one.
    """

    def __init__(self, visual_weight: float = 0.5, audio_weight: float = 0.5):
        total = visual_weight + audio_weight
        self.visual_weight = visual_weight / total
        self.audio_weight = audio_weight / total

    def fuse(
        self,
        visual_probs: Optional[dict] = None,
        audio_probs: Optional[dict] = None,
    ) -> dict:
        """
        Fuse visual and audio stress probabilities.

        Args:
            visual_probs: {"not_stressed": p0, "stressed": p1} or None
            audio_probs: {"not_stressed": p0, "stressed": p1} or None

        Returns:
            dict with fused label, confidence, and probabilities.
        """
        if visual_probs is None and audio_probs is None:
            return {
                "label": "unknown",
                "confidence": 0.0,
                "probabilities": {"not_stressed": 0.5, "stressed": 0.5},
                "modalities_used": [],
            }

        if visual_probs is None:
            return {
                "label": "stressed" if audio_probs["stressed"] > 0.5 else "not_stressed",
                "confidence": max(audio_probs["stressed"], audio_probs["not_stressed"]),
                "probabilities": audio_probs,
                "modalities_used": ["audio"],
            }

        if audio_probs is None:
            return {
                "label": "stressed" if visual_probs["stressed"] > 0.5 else "not_stressed",
                "confidence": max(visual_probs["stressed"], visual_probs["not_stressed"]),
                "probabilities": visual_probs,
                "modalities_used": ["visual"],
            }

        fused_stressed = (
            self.visual_weight * visual_probs["stressed"]
            + self.audio_weight * audio_probs["stressed"]
        )
        fused_not_stressed = 1.0 - fused_stressed

        label = "stressed" if fused_stressed > 0.5 else "not_stressed"
        return {
            "label": label,
            "confidence": max(fused_stressed, fused_not_stressed),
            "probabilities": {
                "not_stressed": fused_not_stressed,
                "stressed": fused_stressed,
            },
            "modalities_used": ["visual", "audio"],
        }


class MetaClassifierFusion(nn.Module):
    """
    Late fusion via a small MLP meta-classifier.

    Input: concatenation of visual_probs and audio_probs (4-D) → hidden → 2-class output.

    Must be trained on validation data where both modality outputs are available.
    """

    def __init__(self, input_dim: int = 4, hidden_dim: int = 16, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EarlyFusionClassifier(nn.Module):
    """
    Early fusion classifier: concatenates raw visual temporal features
    and audio embeddings into a single input vector.

    Default input: 80 (visual temporal stats) + 512 (Wav2Vec2 embedding) = 592.
    """

    def __init__(self, visual_dim: int = 80, audio_dim: int = 512, hidden_dim: int = 256, num_classes: int = 2):
        super().__init__()
        input_dim = visual_dim + audio_dim
        self.net = nn.Sequential(
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
        return self.net(x)


class MultimodalFusion:
    """
    Unified fusion interface that dispatches to the selected strategy.
    """

    LABELS = ["not_stressed", "stressed"]

    def __init__(
        self,
        strategy: FusionStrategy = FusionStrategy.WEIGHTED_AVERAGE,
        visual_weight: float = 0.5,
        audio_weight: float = 0.5,
        meta_weights_path: Optional[str] = None,
        early_weights_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.strategy = strategy
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if strategy == FusionStrategy.WEIGHTED_AVERAGE:
            self.fuser = WeightedAverageFusion(visual_weight, audio_weight)

        elif strategy == FusionStrategy.META_CLASSIFIER:
            self.meta_clf = MetaClassifierFusion().to(self.device)
            if meta_weights_path:
                state = torch.load(meta_weights_path, map_location=self.device, weights_only=True)
                self.meta_clf.load_state_dict(state)
            self.meta_clf.eval()

        elif strategy == FusionStrategy.EARLY_FUSION:
            self.early_clf = EarlyFusionClassifier().to(self.device)
            if early_weights_path:
                state = torch.load(early_weights_path, map_location=self.device, weights_only=True)
                self.early_clf.load_state_dict(state)
            self.early_clf.eval()

    def fuse(
        self,
        visual_result: Optional[dict] = None,
        audio_result: Optional[dict] = None,
        visual_features: Optional[np.ndarray] = None,
        audio_embedding: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Fuse results from visual and audio models.

        Args:
            visual_result: Output dict from VisualStressDetector.process_frame()
            audio_result: Output dict from AudioStressDetector.predict()
            visual_features: Raw temporal stats (for early fusion)
            audio_embedding: Raw 512-D Wav2Vec2 embedding (for early fusion)

        Returns:
            Fused prediction dict.
        """
        if self.strategy == FusionStrategy.WEIGHTED_AVERAGE:
            v_probs = visual_result["probabilities"] if visual_result else None
            a_probs = audio_result["probabilities"] if audio_result else None
            return self.fuser.fuse(v_probs, a_probs)

        elif self.strategy == FusionStrategy.META_CLASSIFIER:
            return self._meta_fuse(visual_result, audio_result)

        elif self.strategy == FusionStrategy.EARLY_FUSION:
            return self._early_fuse(visual_features, audio_embedding)

        raise ValueError(f"Unknown fusion strategy: {self.strategy}")

    @torch.no_grad()
    def _meta_fuse(self, visual_result: Optional[dict], audio_result: Optional[dict]) -> dict:
        v_probs = (
            [visual_result["probabilities"]["not_stressed"], visual_result["probabilities"]["stressed"]]
            if visual_result
            else [0.5, 0.5]
        )
        a_probs = (
            [audio_result["probabilities"]["not_stressed"], audio_result["probabilities"]["stressed"]]
            if audio_result
            else [0.5, 0.5]
        )
        x = torch.tensor(v_probs + a_probs, dtype=torch.float32).unsqueeze(0).to(self.device)
        logits = self.meta_clf(x)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        label_idx = int(np.argmax(probs))
        modalities = []
        if visual_result:
            modalities.append("visual")
        if audio_result:
            modalities.append("audio")

        return {
            "label": self.LABELS[label_idx],
            "confidence": float(probs[label_idx]),
            "probabilities": {self.LABELS[i]: float(probs[i]) for i in range(2)},
            "modalities_used": modalities,
        }

    @torch.no_grad()
    def _early_fuse(
        self, visual_features: Optional[np.ndarray], audio_embedding: Optional[np.ndarray]
    ) -> dict:
        v_feat = visual_features if visual_features is not None else np.zeros(80, dtype=np.float32)
        a_feat = audio_embedding if audio_embedding is not None else np.zeros(512, dtype=np.float32)
        combined = np.concatenate([v_feat, a_feat])
        x = torch.tensor(combined, dtype=torch.float32).unsqueeze(0).to(self.device)
        logits = self.early_clf(x)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        label_idx = int(np.argmax(probs))
        modalities = []
        if visual_features is not None:
            modalities.append("visual")
        if audio_embedding is not None:
            modalities.append("audio")

        return {
            "label": self.LABELS[label_idx],
            "confidence": float(probs[label_idx]),
            "probabilities": {self.LABELS[i]: float(probs[i]) for i in range(2)},
            "modalities_used": modalities,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Multi-Dimension Fusion — four behavioral dimensions
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class DimensionScores:
    """Container for four-dimensional behavioral profile scores."""
    cognitive_load: float = 0.0
    emotional_arousal: float = 0.0
    engagement_level: float = 0.0
    confidence_level: float = 0.0
    baseline_deviation: float = 0.0
    timestamp: float = 0.0
    modalities_used: list = field(default_factory=list)
    baseline_calibrated: bool = False
    warning_flags: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "cognitive_load": round(self.cognitive_load, 4),
            "emotional_arousal": round(self.emotional_arousal, 4),
            "engagement_level": round(self.engagement_level, 4),
            "confidence_level": round(self.confidence_level, 4),
            "baseline_deviation": round(self.baseline_deviation, 4),
            "timestamp": self.timestamp,
            "modalities_used": list(self.modalities_used),
            "baseline_calibrated": self.baseline_calibrated,
            "warning_flags": list(self.warning_flags),
        }


class MultiDimensionFusion:
    """
    Fuse visual, body/gesture, and audio results into a four-dimensional
    behavioral profile (DimensionScores).

    All inputs should be baseline-normalized before calling fuse() when
    a calibrated baseline is available.
    """

    def fuse(
        self,
        visual_result: Optional[dict] = None,
        gesture_result: Optional[dict] = None,
        audio_result: Optional[dict] = None,
        baseline_calibrated: bool = False,
    ) -> DimensionScores:
        warnings_list = []
        modalities = []

        # ── Extract inputs with defaults ──
        brow_furrow = _safe_get(visual_result, "brow_furrow_intensity", None)
        eye_contact = _safe_get(visual_result, "eye_contact_score", None)
        head_nod_rate = _safe_get(visual_result, "head_nod_rate", None)
        valence = _safe_get(visual_result, "valence", None)
        arousal = _safe_get(visual_result, "arousal", None)

        fidget = _safe_get(gesture_result, "fidget_score", None)
        self_touch = _safe_get(gesture_result, "self_touch_score", None)
        posture_open = _safe_get(gesture_result, "posture_openness", None)
        stability = _safe_get(gesture_result, "postural_stability", None)
        forward_lean = _safe_get(gesture_result, "forward_lean", None)
        gesture_rate = _safe_get(gesture_result, "gesture_rate", None)

        audio_stress_prob = None
        if audio_result and "probabilities" in audio_result:
            audio_stress_prob = audio_result["probabilities"].get("stressed")

        if visual_result:
            modalities.append("visual")
        if gesture_result:
            modalities.append("gesture")
        if audio_result:
            modalities.append("audio")

        # Defaults + warnings for null values
        def _default(val, default, flag):
            if val is None:
                warnings_list.append(flag)
                return default
            return float(val)

        brow_f = _default(brow_furrow, 0.5, "brow_furrow_null")
        eye_c = _default(eye_contact, 0.5, "eye_contact_null")
        audio_p = _default(audio_stress_prob, 0.5, "audio_null")
        arousal_v = _default(arousal, 0.0, "arousal_null")
        fidget_v = _default(fidget, 0.5, "fidget_null_default")
        head_nod_v = _default(head_nod_rate, 0.0, "head_nod_null")
        fwd_lean_v = _default(forward_lean, 0.0, "forward_lean_null")
        gesture_r = _default(gesture_rate, 0.0, "gesture_rate_null")
        posture_o = _default(posture_open, 0.5, "posture_openness_null")
        stab_v = _default(stability, 0.5, "stability_null")
        self_t = _default(self_touch, 0.5, "self_touch_null")
        val_v = _default(valence, 0.0, "valence_null")

        if not baseline_calibrated:
            warnings_list.append("baseline_insufficient")

        # ── Normalize hsemotion arousal: from [-1,1] to [0,1] ──
        hsemotion_arousal_norm = float(np.clip((arousal_v + 1.0) / 2.0, 0.0, 1.0))

        # ── Cognitive Load ──
        cognitive_load = (
            0.5 * brow_f
            + 0.3 * (1.0 - eye_c)
            + 0.2 * audio_p
        )

        # ── Emotional Arousal ──
        emotional_arousal = (
            0.4 * hsemotion_arousal_norm
            + 0.35 * audio_p
            + 0.25 * fidget_v
        )

        # ── Engagement Level ──
        head_nod_norm = float(np.clip(head_nod_v / 30.0, 0.0, 1.0))
        fwd_lean_norm = float(np.clip((fwd_lean_v + 1.0) / 2.0, 0.0, 1.0))
        gesture_norm = float(np.clip(gesture_r / 60.0, 0.0, 1.0))
        engagement_level = (
            0.35 * eye_c
            + 0.25 * head_nod_norm
            + 0.25 * fwd_lean_norm
            + 0.15 * gesture_norm
        )

        # ── Confidence Level ──
        valence_norm = float(np.clip((val_v + 1.0) / 2.0, 0.0, 1.0))
        confidence_level = (
            0.3 * posture_o
            + 0.3 * stab_v
            + 0.25 * (1.0 - self_t)
            + 0.15 * valence_norm
        )

        # ── Baseline Deviation ──
        all_inputs = [
            brow_furrow, eye_contact, audio_stress_prob,
            arousal, fidget, head_nod_rate,
            forward_lean, gesture_rate, posture_open,
            stability, self_touch, valence,
        ]
        non_null = [abs(float(v)) for v in all_inputs if v is not None]
        baseline_deviation = float(np.mean(non_null)) if non_null else 0.0

        # Clip all dimensions
        cognitive_load = float(np.clip(cognitive_load, 0.0, 1.0))
        emotional_arousal = float(np.clip(emotional_arousal, 0.0, 1.0))
        engagement_level = float(np.clip(engagement_level, 0.0, 1.0))
        confidence_level = float(np.clip(confidence_level, 0.0, 1.0))

        return DimensionScores(
            cognitive_load=cognitive_load,
            emotional_arousal=emotional_arousal,
            engagement_level=engagement_level,
            confidence_level=confidence_level,
            baseline_deviation=baseline_deviation,
            timestamp=time.time(),
            modalities_used=modalities,
            baseline_calibrated=baseline_calibrated,
            warning_flags=warnings_list,
        )


def _safe_get(d: Optional[dict], key: str, default=None):
    """Safely get a value from a dict that may be None."""
    if d is None:
        return default
    return d.get(key, default)
