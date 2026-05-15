"""Decision-capture hook tests.

Verifies that approving / rejecting an application:
  1. Writes a recruiter_decisions row (with the cached feature snapshot).
  2. Triggers a refit at every Nth decision (we patch the fit fn and count
     calls).
  3. Never raises — a hook failure must not block the transition.
"""
from __future__ import annotations

import json
import sqlite3
import time
from unittest.mock import patch

from agents.recruit.taste.capture import capture_decision


def _count_decisions(conn: sqlite3.Connection, recruiter_id: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM recruiter_decisions WHERE recruiter_id = ?",
        (recruiter_id,),
    ).fetchone()
    return int(row["n"])


def test_capture_writes_decision_row(recruit_conn: sqlite3.Connection) -> None:
    rid = "rec-A"
    decision_id = capture_decision(recruit_conn, rid, "app-test", "approved")

    assert decision_id is not None and decision_id.startswith("dec-")
    assert _count_decisions(recruit_conn, rid) == 1

    row = recruit_conn.execute(
        "SELECT * FROM recruiter_decisions WHERE id = ?",
        (decision_id,),
    ).fetchone()
    assert row["recruiter_id"] == rid
    assert row["application_id"] == "app-test"
    assert row["decision"] == "approved"
    assert row["features_version"] == 1

    snapshot = json.loads(row["features_snapshot_json"])
    # Feature snapshot reflects the seeded application (fit_score 0.78,
    # 3 matched skills, 1 missing) + behavioral means.
    assert snapshot["fit_score"] == 0.78
    assert snapshot["hard_skill_overlap"] == 3.0
    assert snapshot["missing_skills_count"] == 1.0
    assert snapshot["engagement_mean"] == 0.65


def test_capture_unknown_decision_is_ignored(
    recruit_conn: sqlite3.Connection,
) -> None:
    """Anything other than approved/rejected is dropped silently."""
    assert capture_decision(recruit_conn, "rec-A", "app-test", "interviewed") is None
    assert _count_decisions(recruit_conn, "rec-A") == 0


def test_capture_missing_application_returns_none(
    recruit_conn: sqlite3.Connection,
) -> None:
    """Hook is fail-soft: a bad app id does not raise."""
    assert capture_decision(recruit_conn, "rec-A", "nonexistent-app", "approved") is None
    assert _count_decisions(recruit_conn, "rec-A") == 0


def test_capture_triggers_refit_every_5th_decision(
    recruit_conn: sqlite3.Connection,
) -> None:
    """Refit fires at decision counts 5, 10, 15 — not at 4 or 6."""
    rid = "rec-refit"

    # Pre-seed 4 decisions directly so we don't trigger a refit ourselves.
    for i in range(4):
        recruit_conn.execute(
            "INSERT INTO recruiter_decisions (id, recruiter_id, application_id, "
            "features_snapshot_json, features_version, decision, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (f"pre-{i}", rid, "app-test", "{}", 1, "approved", time.time()),
        )

    # Now the 5th decision — through the hook — should trigger a refit.
    with patch("agents.recruit.taste.capture.fit_recruiter_weights") as mock_fit:
        capture_decision(recruit_conn, rid, "app-test", "rejected")
        assert mock_fit.call_count == 1, (
            f"expected refit on the 5th decision, got {mock_fit.call_count}"
        )

    # 6th decision must NOT trigger.
    with patch("agents.recruit.taste.capture.fit_recruiter_weights") as mock_fit:
        capture_decision(recruit_conn, rid, "app-test", "rejected")
        assert mock_fit.call_count == 0, (
            f"unexpected refit on the 6th decision, got {mock_fit.call_count}"
        )


def test_capture_never_raises(recruit_conn: sqlite3.Connection) -> None:
    """Even if every internal call blows up, capture_decision returns None."""
    with patch(
        "agents.recruit.taste.capture.get_or_compute_features",
        side_effect=RuntimeError("boom"),
    ):
        # Must not propagate — additive hook.
        result = capture_decision(recruit_conn, "rec-A", "app-test", "approved")
        assert result is None
