"""
Session Aggregator
===================
Stores the complete session history of DimensionScores in memory and
provides summary statistics, notable moment detection, and timeline
data for visualization and reporting.
"""

import logging
import time
from collections import deque
from typing import Optional

import numpy as np

from .fusion import DimensionScores

logger = logging.getLogger(__name__)

_DIMENSION_KEYS = ["cognitive_load", "emotional_arousal", "engagement_level", "confidence_level"]


class SessionAggregator:
    """
    Accumulates DimensionScores over a session and provides analytics.

    Parameters
    ----------
    maxlen : int
        Maximum number of frames to store (default 10800 = 12 min at 15 fps).
    """

    def __init__(self, maxlen: int = 10800):
        self._history: deque[DimensionScores] = deque(maxlen=maxlen)
        self._question_markers: list[tuple[float, str]] = []
        self._start_time: Optional[float] = None

    # ── Public API ──────────────────────────────────────────────────────

    def add_frame(self, scores: DimensionScores, question_label: Optional[str] = None):
        """Append a DimensionScores frame to the session history."""
        if self._start_time is None:
            self._start_time = scores.timestamp

        self._history.append(scores)

        if question_label is not None:
            self._question_markers.append((scores.timestamp, question_label))

    @property
    def frames_count(self) -> int:
        return len(self._history)

    def get_notable_moments(self, z_threshold: float = 2.0) -> list[dict]:
        """
        Detect frames where any dimension deviates by z_threshold standard
        deviations from the session running mean.

        Returns list of {timestamp, dimension, magnitude, value}.
        Enforces minimum 5-second gap between reported moments.
        """
        if len(self._history) < 10:
            return []

        # Compute running mean/std for each dimension
        arrays = {k: [] for k in _DIMENSION_KEYS}
        timestamps = []
        for s in self._history:
            timestamps.append(s.timestamp)
            for k in _DIMENSION_KEYS:
                arrays[k].append(getattr(s, k))

        moments = []
        last_moment_time = -999.0

        for i in range(10, len(timestamps)):
            t = timestamps[i]
            if t - last_moment_time < 5.0:
                continue

            for k in _DIMENSION_KEYS:
                window = arrays[k][:i]
                mean = float(np.mean(window))
                std = float(np.std(window))
                if std < 1e-6:
                    continue
                val = arrays[k][i]
                z = abs(val - mean) / std
                if z >= z_threshold:
                    moments.append({
                        "timestamp": round(t, 3),
                        "dimension": k,
                        "magnitude": round(z, 2),
                        "value": round(val, 4),
                    })
                    last_moment_time = t
                    break  # One moment per timestamp to avoid duplicates

        return moments

    def get_session_summary(self) -> dict:
        """
        Compute full session summary with per-dimension statistics,
        overall arc, notable moments, and question breakdowns.
        """
        if not self._history:
            return {
                "duration_seconds": 0.0,
                "per_dimension": {},
                "overall_arc": "stable",
                "notable_moments": [],
                "question_breakdowns": {},
                "frames_analyzed": 0,
                "baseline_calibrated": False,
            }

        timestamps = np.array([s.timestamp for s in self._history])
        duration = float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0

        per_dimension = {}
        trends = {}
        for k in _DIMENSION_KEYS:
            vals = np.array([getattr(s, k) for s in self._history])
            mean_v = float(np.mean(vals))
            std_v = float(np.std(vals))
            min_v = float(np.min(vals))
            max_v = float(np.max(vals))

            # Linear regression trend (slope)
            if len(vals) >= 2:
                x = np.arange(len(vals), dtype=np.float64)
                coeffs = np.polyfit(x, vals.astype(np.float64), 1)
                trend = float(coeffs[0])
            else:
                trend = 0.0

            per_dimension[k] = {
                "mean": round(mean_v, 4),
                "std": round(std_v, 4),
                "min": round(min_v, 4),
                "max": round(max_v, 4),
                "trend": round(trend, 6),
            }
            trends[k] = trend

        # Overall arc
        eng_trend = trends.get("engagement_level", 0)
        conf_trend = trends.get("confidence_level", 0)
        any_high_std = any(per_dimension[k]["std"] > 0.2 for k in _DIMENSION_KEYS)

        if eng_trend > 0.01 and conf_trend > 0.01:
            overall_arc = "improving"
        elif eng_trend < -0.01 and conf_trend < -0.01:
            overall_arc = "declining"
        elif any_high_std:
            overall_arc = "variable"
        else:
            overall_arc = "stable"

        # Question breakdowns
        question_breakdowns = {}
        if self._question_markers:
            sorted_markers = sorted(self._question_markers, key=lambda x: x[0])
            for idx, (qtime, qlabel) in enumerate(sorted_markers):
                if idx + 1 < len(sorted_markers):
                    next_time = sorted_markers[idx + 1][0]
                else:
                    next_time = float("inf")

                q_frames = [
                    s for s in self._history
                    if qtime <= s.timestamp < next_time
                ]
                if q_frames:
                    breakdown = {}
                    for k in _DIMENSION_KEYS:
                        vals = [getattr(s, k) for s in q_frames]
                        breakdown[k] = round(float(np.mean(vals)), 4)
                    question_breakdowns[qlabel] = breakdown

        baseline_calibrated = self._history[0].baseline_calibrated if self._history else False

        return {
            "duration_seconds": round(duration, 2),
            "per_dimension": per_dimension,
            "overall_arc": overall_arc,
            "notable_moments": self.get_notable_moments(),
            "question_breakdowns": question_breakdowns,
            "frames_analyzed": len(self._history),
            "baseline_calibrated": baseline_calibrated,
        }

    def get_timeline_data(self, max_points: int = 300) -> list[dict]:
        """
        Downsample session history to max_points using uniform stride.

        Each point: {timestamp, cognitive_load, emotional_arousal,
                     engagement_level, confidence_level}
        """
        if not self._history:
            return []

        n = len(self._history)
        stride = max(1, n // max_points)
        points = []
        start_ts = self._history[0].timestamp

        for i in range(0, n, stride):
            s = self._history[i]
            points.append({
                "timestamp": round(s.timestamp - start_ts, 3),
                "cognitive_load": round(s.cognitive_load, 4),
                "emotional_arousal": round(s.emotional_arousal, 4),
                "engagement_level": round(s.engagement_level, 4),
                "confidence_level": round(s.confidence_level, 4),
            })

        return points[:max_points]

    @property
    def question_markers(self) -> list[tuple[float, str]]:
        return list(self._question_markers)
