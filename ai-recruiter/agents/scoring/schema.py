"""Pydantic data contracts for the scoring agent.

Single source of truth for the shapes that move between the reader, the
scorer, the report builder, and the API surface.
"""
from __future__ import annotations

from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field

Score05 = Annotated[int, Field(ge=0, le=5)]


class TranscriptTurn(BaseModel):
    """One Q/A pair from a recorded interview."""
    q: str
    a: str


class TurnScore(BaseModel):
    """Per-answer assessment.

    Scores may be None when the LLM failed for this turn after the retry
    budget was exhausted; the rationale then carries the error string.
    """
    turn_index: int
    question: str
    answer: str
    technical_score: Optional[Score05]
    technical_rationale: str
    coherence_score: Optional[Score05]
    coherence_rationale: str


class OverallAssessment(BaseModel):
    recommendation: Literal["strong_hire", "hire", "lean_hire", "no_hire"]
    technical_avg: float
    coherence_avg: float
    summary: str
    strengths: list[str]
    concerns: list[str]


class InterviewReport(BaseModel):
    interview_id: str
    candidate_name: Optional[str]
    job_title: Optional[str]
    generated_at: float
    model: str
    turns: list[TurnScore]
    overall: OverallAssessment


class ScoreRequest(BaseModel):
    """Request body for POST /api/scoring/{interview_id}/run."""
    force: bool = False
    transcript: Optional[list[TranscriptTurn]] = None
