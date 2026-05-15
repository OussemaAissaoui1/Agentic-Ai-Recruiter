"""Vision agent — real-time multimodal behavioral analysis.

Owns MediaPipe (face/pose/hands), HSEmotion, Wav2Vec2/StudentNet, baseline
calibration, dimension fusion, session aggregation, and PDF/JSON reports.
"""

from .agent import VisionAgent

__all__ = ["VisionAgent"]
