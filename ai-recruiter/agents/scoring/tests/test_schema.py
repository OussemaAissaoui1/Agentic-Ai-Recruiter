import pytest
from pydantic import ValidationError

from agents.scoring.schema import (
    TranscriptTurn,
    TurnScore,
    OverallAssessment,
    InterviewReport,
    ScoreRequest,
)


def test_transcript_turn_q_a_shape():
    t = TranscriptTurn(q="What's your stack?", a="Python and Postgres.")
    assert t.q.startswith("What")
    assert t.a == "Python and Postgres."


def test_turn_score_accepts_none_on_failure():
    s = TurnScore(
        turn_index=0,
        question="q?",
        answer="a",
        technical_score=None,
        technical_rationale="api error: 500",
        coherence_score=None,
        coherence_rationale="api error: 500",
    )
    assert s.technical_score is None


def test_turn_score_rejects_out_of_range():
    with pytest.raises(ValidationError):
        TurnScore(
            turn_index=0, question="q", answer="a",
            technical_score=7, technical_rationale="x",
            coherence_score=3, coherence_rationale="x",
        )


def test_overall_recommendation_is_enum_constrained():
    with pytest.raises(ValidationError):
        OverallAssessment(
            recommendation="maybe",
            technical_avg=3.0, coherence_avg=3.0,
            summary="x", strengths=["a"], concerns=["b"],
        )


def test_interview_report_round_trip():
    r = InterviewReport(
        interview_id="int-1",
        candidate_name="Sarah",
        job_title="DE",
        generated_at=1700000000.0,
        model="llama-3.3-70b-versatile",
        turns=[],
        overall=OverallAssessment(
            recommendation="hire",
            technical_avg=4.0, coherence_avg=4.5,
            summary="solid", strengths=["a"], concerns=["b"],
        ),
    )
    blob = r.model_dump_json()
    again = InterviewReport.model_validate_json(blob)
    assert again.interview_id == "int-1"


def test_score_request_defaults():
    req = ScoreRequest()
    assert req.force is False
    assert req.transcript is None
