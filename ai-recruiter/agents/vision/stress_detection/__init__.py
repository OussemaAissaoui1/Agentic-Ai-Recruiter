"""
Stress Detection Pipeline - Multimodal stress detection using visual and audio cues.

Modules:
    audio_model         - Audio stress detection via StudentNet (Wav2Vec2 embeddings)
    visual_model        - Visual stress detection via MediaPipe facial landmarks + AU-based MLP
    fusion              - Multimodal late/early fusion of visual and audio stress scores
    pipeline            - Real-time orchestration of camera/mic → preprocessing → inference → fusion
    baseline            - Personal baseline calibrator (Welford z-score normalization)
    body_language       - Gesture and posture analysis via MediaPipe Hands + existing Pose
    session_aggregator  - Session-level tracking and analytics
    report_generator    - JSON and PDF interview report generation
"""

__version__ = "0.2.0"

from .baseline import PersonalBaselineCalibrator
from .body_language import GestureAnalyzer
from .fusion import DimensionScores, MultiDimensionFusion
from .report_generator import InterviewReportGenerator
from .session_aggregator import SessionAggregator
