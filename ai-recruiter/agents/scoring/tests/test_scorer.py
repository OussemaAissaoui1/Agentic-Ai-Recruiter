import pytest

from agents.scoring.llm_client import GroqError
from agents.scoring.scorer import score_interview
from agents.scoring.schema import TranscriptTurn


def _ctx():
    return dict(
        interview_id="int-1",
        candidate_name="Sarah",
        job_title="DE",
        cv_text="cv",
        jd_text="jd",
        transcript=[
            TranscriptTurn(q="q1?", a="a1"),
            TranscriptTurn(q="q2?", a="a2"),
        ],
    )


async def test_score_interview_happy_path(fake_groq_factory):
    replies = [
        # per-turn 1
        {"technical_score": 4, "technical_rationale": "good detail",
         "coherence_score": 5, "coherence_rationale": "clear"},
        # per-turn 2
        {"technical_score": 3, "technical_rationale": "ok",
         "coherence_score": 4, "coherence_rationale": "ok"},
        # overall
        {"recommendation": "hire",
         "summary": "solid candidate",
         "strengths": ["a", "b"],
         "concerns": ["c"]},
    ]
    client = fake_groq_factory(replies)
    report = await score_interview(client=client, model="m-1", **_ctx())

    assert report.interview_id == "int-1"
    assert len(report.turns) == 2
    assert report.turns[0].technical_score == 4
    assert report.turns[1].coherence_score == 4
    assert report.overall.recommendation == "hire"
    assert report.overall.technical_avg == pytest.approx(3.5)
    assert report.overall.coherence_avg == pytest.approx(4.5)
    assert client.calls[0]["user"].count("q1?") == 1


async def test_score_interview_per_turn_failure_isolated(fake_groq_factory):
    replies = [
        # turn 1 fails
        GroqError("groq 502: bad gateway"),
        # turn 2 ok
        {"technical_score": 5, "technical_rationale": "expert",
         "coherence_score": 5, "coherence_rationale": "great"},
        # overall ok
        {"recommendation": "hire", "summary": "x",
         "strengths": ["a"], "concerns": ["b"]},
    ]
    client = fake_groq_factory(replies)
    report = await score_interview(client=client, model="m-1", **_ctx())

    assert report.turns[0].technical_score is None
    assert "502" in report.turns[0].technical_rationale
    assert report.turns[1].technical_score == 5
    # avg computed only over non-None
    assert report.overall.technical_avg == pytest.approx(5.0)


async def test_score_interview_overall_failure_falls_back(fake_groq_factory):
    replies = [
        {"technical_score": 4, "technical_rationale": "x",
         "coherence_score": 4, "coherence_rationale": "x"},
        {"technical_score": 4, "technical_rationale": "x",
         "coherence_score": 4, "coherence_rationale": "x"},
        GroqError("groq 500: boom"),
    ]
    client = fake_groq_factory(replies)
    report = await score_interview(client=client, model="m-1", **_ctx())

    # Fallback rule: avg 4.0 → "hire"
    assert report.overall.recommendation == "hire"
    assert "500" in report.overall.summary
    assert report.turns[0].technical_score == 4


async def test_empty_transcript_raises():
    ctx = _ctx()
    ctx["transcript"] = []
    with pytest.raises(ValueError):
        await score_interview(client=None, model="m-1", **ctx)
