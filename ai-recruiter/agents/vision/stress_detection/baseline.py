"""
Personal Baseline Calibrator
==============================
Collects visual and audio features during a calibration window,
computes per-feature mean/std using Welford's online algorithm,
and provides z-score normalization for downstream fusion.
"""

import logging
import time
import warnings
from typing import Optional

logger = logging.getLogger(__name__)


class _WelfordAccumulator:
    """Running mean/variance via Welford's online algorithm."""

    __slots__ = ("count", "mean", "m2")

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, value: float):
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)

    @property
    def std(self) -> float:
        return self.variance ** 0.5


class PersonalBaselineCalibrator:
    """
    Collects feature observations during a calibration window and
    computes per-feature z-score normalization statistics.

    Parameters
    ----------
    calibration_duration_seconds : int
        Wall-clock time for the calibration window (default 45 s).
    target_fps : int
        Expected frame rate used to estimate target frame count.
    """

    def __init__(self, calibration_duration_seconds: int = 45, target_fps: int = 15):
        self.calibration_duration_seconds = calibration_duration_seconds
        self.target_fps = target_fps
        self._target_frames = calibration_duration_seconds * target_fps

        self._accumulators: dict[str, _WelfordAccumulator] = {}
        self._frames_collected = 0
        self._start_time: Optional[float] = None
        self._frozen = False
        self._baseline_stats: dict[str, dict] = {}
        self._calibrated = False

    # ── Public API ──────────────────────────────────────────────────────

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    @property
    def calibration_progress(self) -> float:
        """0.0 → 1.0 based on elapsed time vs calibration_duration_seconds."""
        if self._frozen:
            return 1.0
        if self._start_time is None:
            return 0.0
        elapsed = time.time() - self._start_time
        return min(elapsed / max(self.calibration_duration_seconds, 1), 1.0)

    @property
    def baseline_stats(self) -> dict:
        """Per-feature {mean, std} snapshot frozen at calibration end."""
        return dict(self._baseline_stats)

    def start(self):
        """Begin (or restart) the calibration window."""
        self._accumulators.clear()
        self._frames_collected = 0
        self._start_time = time.time()
        self._frozen = False
        self._calibrated = False
        self._baseline_stats.clear()
        logger.info("Baseline calibration started (duration=%ds)", self.calibration_duration_seconds)

    def add_features(self, feature_dict: dict):
        """
        Feed a feature observation during calibration.

        Parameters
        ----------
        feature_dict : dict
            Keys are feature names, values are floats or None.
            None values are silently skipped.
        """
        if self._frozen:
            return

        if self._start_time is None:
            self.start()

        for key, value in feature_dict.items():
            if value is None:
                continue
            try:
                val = float(value)
            except (TypeError, ValueError):
                continue
            if key not in self._accumulators:
                self._accumulators[key] = _WelfordAccumulator()
            self._accumulators[key].update(val)

        self._frames_collected += 1

    def complete(self) -> bool:
        """
        Freeze the baseline. Returns True if enough data was collected.
        """
        if self._frozen:
            return self._calibrated

        self._frozen = True

        if self._frames_collected < 20:
            warnings.warn(
                f"Baseline calibration insufficient: only {self._frames_collected} "
                f"frames collected (need >= 20). Raw scores will be used.",
                stacklevel=2,
            )
            self._calibrated = False
            logger.warning(
                "Baseline calibration FAILED — %d frames < 20 minimum",
                self._frames_collected,
            )
            return False

        # Freeze stats
        for key, acc in self._accumulators.items():
            self._baseline_stats[key] = {
                "mean": acc.mean,
                "std": acc.std,
                "count": acc.count,
            }

        self._calibrated = True
        logger.info(
            "Baseline calibration complete: %d frames, %d features",
            self._frames_collected,
            len(self._baseline_stats),
        )
        return True

    def normalize(self, feature_dict: dict) -> dict:
        """
        Z-score normalize a feature dict using the frozen baseline.

        If not calibrated, returns the input dict unchanged.
        """
        if not self._calibrated:
            return dict(feature_dict)

        result = {}
        for key, value in feature_dict.items():
            if value is None:
                result[key] = None
                continue
            stats = self._baseline_stats.get(key)
            if stats is None or stats["std"] < 1e-9:
                result[key] = value
            else:
                result[key] = (float(value) - stats["mean"]) / stats["std"]
        return result

    def reset(self):
        """Fully reset calibrator state."""
        self._accumulators.clear()
        self._frames_collected = 0
        self._start_time = None
        self._frozen = False
        self._calibrated = False
        self._baseline_stats.clear()
