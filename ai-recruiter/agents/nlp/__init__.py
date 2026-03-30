"""
NLP Agent Package

Provides the AI Recruiter's natural language processing capabilities:
- Interview question generation
- Response analysis
- State tracking
"""

from .agent import NLPAgent, QuestionGenerationRequest, QuestionGenerationResponse
from .vllm_engine import VLLMRecruiterEngine, GenerationMetrics
from .interview_state import InterviewStateTracker, AnswerAnalysis

__version__ = "1.0.0"

__all__ = [
    "NLPAgent",
    "QuestionGenerationRequest",
    "QuestionGenerationResponse",
    "VLLMRecruiterEngine",
    "GenerationMetrics",
    "InterviewStateTracker",
    "AnswerAnalysis",
]
