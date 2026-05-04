import json
import time
from pathlib import Path

import pytest

from agents.scoring.report import (
    build_overall_fallback,
    compute_averages,
    render_markdown,
    read_cache,
    write_cache,
    write_back_scores,
)
from agents.scoring.schema import (
    InterviewReport,
    OverallAssessment,
    TurnScore,
)


def _turn(idx, t, c, t_text="x", c_text="y"):
    return TurnScore(
        turn_index=idx, question=f"q{idx}", answer=f"a{idx}",
        technical_score=t, technical_rationale=t_text,
        coherence_score=c, coherence_rationale=c_text,
    )


def test_compute_averages_ignores_none():
    turns = [_turn(0, 4, 5), _turn(1, None, None, "err", "err"), _turn(2, 2, 3)]
    t, c = compute_averages(turns)
    assert t == pytest.approx(3.0)
    assert c == pytest.approx(4.0)


def test_compute_averages_all_none_returns_zero():
    turns = [_turn(0, None, None, "err", "err")]
    assert compute_averages(turns) == (0.0, 0.0)


@pytest.mark.parametrize("avg, expected", [
    (4.5, "strong_hire"),
    (3.8, "hire"),
    (2.5, "lean_hire"),
    (1.0, "no_hire"),
])
def test_overall_fallback_thresholds(avg, expected):
    fb = build_overall_fallback(avg, avg)
    assert fb.recommendation == expected
    assert fb.strengths
    assert fb.concerns


def _full_report():
    return InterviewReport(
        interview_id="int-1", candidate_name="Sarah", job_title="DE",
        generated_at=1700000000.0, model="llama-3.3-70b-versatile",
        turns=[_turn(0, 4, 5), _turn(1, 3, 4)],
        overall=OverallAssessment(
            recommendation="hire", technical_avg=3.5, coherence_avg=4.5,
            summary="solid technical depth and clear communication.",
            strengths=["Spark expertise", "trade-off awareness"],
            concerns=["limited streaming background"],
        ),
    )


def test_render_markdown_contains_expected_sections():
    md = render_markdown(_full_report())
    assert "# Interview Report" in md
    assert "Sarah" in md and "DE" in md
    assert "**Recommendation:** hire" in md
    assert "Technical 3.5/5" in md
    assert "## Summary" in md
    assert "### Strengths" in md
    assert "Spark expertise" in md
    assert "## Per-Answer Breakdown" in md
    assert "### Turn 0" in md
    assert "Technical 4/5" in md


def test_render_markdown_handles_failed_turn():
    report = _full_report()
    report.turns[0] = _turn(0, None, None, "scoring failed: 502", "scoring failed: 502")
    md = render_markdown(report)
    assert "Technical n/a" in md
    assert "_(scoring failed)_" in md


def test_cache_round_trip(tmp_path: Path):
    report = _full_report()
    write_cache(tmp_path, report)
    json_path = tmp_path / "int-1_report.json"
    md_path = tmp_path / "int-1_report.md"
    assert json_path.exists() and md_path.exists()

    again = read_cache(tmp_path, "int-1")
    assert again is not None
    assert again.interview_id == "int-1"
    assert again.overall.recommendation == "hire"


def test_read_cache_missing_returns_none(tmp_path: Path):
    assert read_cache(tmp_path, "nope") is None


def test_write_back_scores_updates_db(recruit_conn):
    report = _full_report()
    report.interview_id = "int-test"
    write_back_scores(recruit_conn, report)
    row = recruit_conn.execute(
        "SELECT scores_json FROM interviews WHERE id=?", ("int-test",)
    ).fetchone()
    payload = json.loads(row["scores_json"])
    assert payload["recommendation"] == "hire"
    assert payload["technical_avg"] == 3.5
    assert "generated_at" in payload


def test_write_back_scores_swallows_sqlite_error(tmp_path: Path):
    """write_back_scores is best-effort: a closed connection must not raise."""
    import sqlite3
    conn = sqlite3.connect(":memory:")
    conn.close()
    # Should not raise
    write_back_scores(conn, _full_report())


def test_read_cache_returns_none_on_corrupt_json(tmp_path: Path):
    """A partial / corrupt JSON file in the cache directory yields None, not an exception."""
    (tmp_path / "int-x_report.json").write_text("{not valid json")
    assert read_cache(tmp_path, "int-x") is None
