"""Live Groq smoke test. Skipped unless GROQ_API_KEY is set."""
import os

import pytest

from agents.scoring.llm_client import GroqClient
from agents.scoring.scorer import score_interview
from agents.scoring.schema import TranscriptTurn

pytestmark = pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set; skipping live smoke test",
)


async def test_real_groq_three_turn_interview():
    client = GroqClient.from_env()
    try:
        report = await score_interview(
            client=client,
            model=os.environ.get("SCORING_MODEL", "llama-3.3-70b-versatile"),
            interview_id="smoke-1",
            candidate_name="Test Candidate",
            job_title="Senior Data Engineer",
            cv_text="10 years Python and Spark.",
            jd_text="Build pipelines. Python, Spark, BigQuery.",
            transcript=[
                TranscriptTurn(
                    q="Walk me through your BigQuery migration.",
                    a="Dual-write pattern over six weeks; reconciled row counts daily.",
                ),
                TranscriptTurn(
                    q="What broke during the migration?",
                    a="Skewed keys caused shuffle hot spots; we salted them.",
                ),
                TranscriptTurn(
                    q="How did you validate parity?",
                    a="Per-partition checksums; mismatches paged on-call.",
                ),
            ],
        )
    finally:
        await client.aclose()

    assert len(report.turns) == 3
    assert all(t.technical_score is not None for t in report.turns)
    assert report.overall.recommendation in {
        "strong_hire", "hire", "lean_hire", "no_hire"
    }
    assert 0.0 <= report.overall.technical_avg <= 5.0
