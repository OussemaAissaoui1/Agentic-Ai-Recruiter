"""Decision capture: hook into approve/reject transitions.

Called from the recruit API's update_application (rejected path) and
invite_interview (approved path). Records a (features, decision) row and
triggers a background refit at every Nth decision.

This is strictly additive: a failure here MUST NOT block the transition.
Caller wraps the call in try/except; we also swallow internally so a
stray exception doesn't propagate.
"""
from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
import uuid
from typing import Optional

from ._lock import DB_LOCK
from .features import FEATURES_VERSION, get_or_compute_features
from .learner import fit_recruiter_weights

log = logging.getLogger("agents.recruit.taste.capture")

DEFAULT_REFIT_EVERY = 5     # refit every Nth decision per recruiter
DEFAULT_RECRUITER_ID = "hr-default"


def capture_decision(
    conn: sqlite3.Connection,
    recruiter_id: str,
    application_id: str,
    decision: str,
    *,
    refit_every: int = DEFAULT_REFIT_EVERY,
    dwell_seconds: Optional[float] = None,
    notes_text: Optional[str] = None,
) -> Optional[str]:
    """Write a recruiter_decisions row + maybe schedule a refit.

    Returns the new decision id on success, None on any failure (the
    caller doesn't need this — it's for tests). All exceptions are caught
    and logged so the surrounding HTTP transition is never blocked.
    """
    if decision not in ("approved", "rejected"):
        log.warning("capture_decision: ignoring unknown decision %r", decision)
        return None

    try:
        feats = get_or_compute_features(conn, application_id)
    except LookupError:
        log.warning("capture_decision: application not found: %s", application_id)
        return None
    except Exception as e:                       # pragma: no cover (defensive)
        log.warning("capture_decision: feature compute failed: %s", e)
        return None

    decision_id = f"dec-{uuid.uuid4().hex[:12]}"
    try:
        import json as _json
        with DB_LOCK:
            conn.execute(
                "INSERT INTO recruiter_decisions (id, recruiter_id, application_id, "
                "features_snapshot_json, features_version, decision, dwell_seconds, "
                "notes_text, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (decision_id, recruiter_id, application_id,
                 _json.dumps(feats), FEATURES_VERSION, decision,
                 dwell_seconds, notes_text, time.time()),
            )
    except sqlite3.IntegrityError as e:
        log.warning("capture_decision: insert failed: %s", e)
        return None
    except Exception as e:                       # pragma: no cover
        log.exception("capture_decision: unexpected insert error: %s", e)
        return None

    # Refit-trigger check. Count decisions for this recruiter; if we hit a
    # multiple of refit_every, fire-and-forget the refit so the HTTP
    # response isn't blocked on sklearn.
    try:
        with DB_LOCK:
            n = _count_decisions(conn, recruiter_id)
        if refit_every > 0 and n > 0 and n % refit_every == 0:
            _schedule_refit(conn, recruiter_id)
    except Exception as e:                       # pragma: no cover
        log.warning("capture_decision: refit scheduling failed: %s", e)

    return decision_id


def _count_decisions(conn: sqlite3.Connection, recruiter_id: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM recruiter_decisions WHERE recruiter_id = ?",
        (recruiter_id,),
    ).fetchone()
    return int(row["n"]) if row else 0


def _schedule_refit(conn: sqlite3.Connection, recruiter_id: str) -> None:
    """Run the refit on a thread so the HTTP handler returns immediately.

    No event loop required: if we're inside one, schedule via to_thread;
    if not (e.g. CLI usage), run synchronously. The CLI seeder calls
    fit_recruiter_weights directly anyway.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop — run synchronously. This branch is exercised by
        # the test suite and the CLI seeder.
        try:
            fit_recruiter_weights(conn, recruiter_id)
        except Exception as e:                   # pragma: no cover
            log.warning("synchronous refit failed: %s", e)
        return

    async def _do_refit() -> None:
        try:
            await asyncio.to_thread(fit_recruiter_weights, conn, recruiter_id)
            log.info("recruiter_taste: refit complete for %s", recruiter_id)
        except Exception as e:                   # pragma: no cover
            log.warning("background refit failed for %s: %s", recruiter_id, e)

    loop.create_task(_do_refit())
