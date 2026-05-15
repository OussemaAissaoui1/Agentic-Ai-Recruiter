"""Feature cache for recruiter-taste learning.

Pulls a flat 10-dim feature vector for an application from three existing
sources:

  1. applications row (matching agent's output): fit_score, matched/missing
     skills counts.
  2. interviews.behavioral_json (vision agent's per-dimension stats):
     engagement / confidence / cognitive_load / emotional_arousal means.
  3. cached scoring report (artifacts/scoring/<id>_report.json): technical
     and coherence averages, recommendation tier.

The vector is persisted in candidate_features keyed by (application_id,
features_version). The version constant lets us bump the feature set
without invalidating the decision history (each decision row also stores
its own snapshot, so old decisions remain interpretable in their own
basis even after a bump).
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Optional

from configs import resolve

from ._lock import DB_LOCK

# Bump when adding/removing features. Decision rows keep their own
# features_snapshot_json + features_version, so historical decisions are
# never re-interpreted in a newer feature space.
FEATURES_VERSION = 1

# Order matters — corresponds 1:1 with the weight vector the learner fits.
FEATURE_NAMES: tuple[str, ...] = (
    "fit_score",
    "hard_skill_overlap",
    "missing_skills_count",
    "technical_avg",
    "coherence_avg",
    "recommendation_tier",
    "engagement_mean",
    "confidence_mean",
    "cognitive_load_mean",
    "emotional_arousal_mean",
)

# Plain-language labels for the UI (matches FEATURE_NAMES order).
FEATURE_LABELS: dict[str, str] = {
    "fit_score": "CV/JD fit score",
    "hard_skill_overlap": "matched hard skills",
    "missing_skills_count": "missing required skills",
    "technical_avg": "technical answer quality",
    "coherence_avg": "answer coherence",
    "recommendation_tier": "AI hiring recommendation",
    "engagement_mean": "engagement during interview",
    "confidence_mean": "confidence during interview",
    "cognitive_load_mean": "cognitive load (stress proxy)",
    "emotional_arousal_mean": "emotional arousal",
}

_RECOMMENDATION_TIER: dict[str, float] = {
    "no_hire": 0.0,
    "lean_hire": 1.0,
    "hire": 2.0,
    "strong_hire": 3.0,
}


def get_or_compute_features(
    conn: sqlite3.Connection,
    application_id: str,
    *,
    force: bool = False,
) -> dict[str, float]:
    """Return the cached feature vector for *application_id*.

    On miss (or when *force* is True), recomputes from applications +
    interviews + scoring artifacts, persists, and returns the fresh
    vector. On a hit at the current FEATURES_VERSION, returns the cached
    vector verbatim.
    """
    with DB_LOCK:
        if not force:
            cached = _read_cache(conn, application_id, FEATURES_VERSION)
            if cached is not None:
                return cached

        vec = _compute_features(conn, application_id)
        _write_cache(conn, application_id, FEATURES_VERSION, vec)
        return vec


def feature_vector_to_array(vec: dict[str, float]) -> list[float]:
    """Project a feature dict into the canonical FEATURE_NAMES order.

    Missing keys default to 0.0 (the learner standardises before fitting,
    so 0.0 means "use the population mean", which is the right
    no-information prior for cold rows).
    """
    return [float(vec.get(name, 0.0)) for name in FEATURE_NAMES]


# ─── internals ────────────────────────────────────────────────────────────────


def _read_cache(
    conn: sqlite3.Connection, application_id: str, version: int,
) -> Optional[dict[str, float]]:
    row = conn.execute(
        "SELECT features_json FROM candidate_features "
        "WHERE application_id = ? AND features_version = ?",
        (application_id, version),
    ).fetchone()
    if row is None:
        return None
    try:
        data = json.loads(row["features_json"])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _write_cache(
    conn: sqlite3.Connection,
    application_id: str,
    version: int,
    vec: dict[str, float],
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO candidate_features "
        "(application_id, features_version, features_json, computed_at) "
        "VALUES (?, ?, ?, ?)",
        (application_id, version, json.dumps(vec), time.time()),
    )


def _compute_features(
    conn: sqlite3.Connection, application_id: str,
) -> dict[str, float]:
    app_row = conn.execute(
        "SELECT id, fit_score, matched_skills_json, missing_skills_json "
        "FROM applications WHERE id = ?",
        (application_id,),
    ).fetchone()
    if app_row is None:
        raise LookupError(f"application not found: {application_id}")

    out: dict[str, float] = {name: 0.0 for name in FEATURE_NAMES}
    out["fit_score"] = float(app_row["fit_score"] or 0.0)
    out["hard_skill_overlap"] = float(_safe_len(app_row["matched_skills_json"]))
    out["missing_skills_count"] = float(_safe_len(app_row["missing_skills_json"]))

    # Pull the most recent completed interview for this application — that's
    # where vision and scoring outputs land.
    interview = conn.execute(
        "SELECT id, behavioral_json FROM interviews "
        "WHERE application_id = ? "
        "ORDER BY rowid DESC LIMIT 1",
        (application_id,),
    ).fetchone()
    interview_id: Optional[str] = None
    if interview is not None:
        interview_id = interview["id"]
        _fill_behavioral(out, interview["behavioral_json"])

    if interview_id is not None:
        _fill_scoring(out, interview_id)

    return out


def _fill_behavioral(out: dict[str, float], raw: Optional[str]) -> None:
    if not raw:
        return
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return
    if not isinstance(data, dict):
        return
    per_dim = data.get("per_dimension") or {}
    if not isinstance(per_dim, dict):
        return
    pairs = (
        ("engagement_mean", "engagement_level"),
        ("confidence_mean", "confidence_level"),
        ("cognitive_load_mean", "cognitive_load"),
        ("emotional_arousal_mean", "emotional_arousal"),
    )
    for out_key, src_key in pairs:
        d = per_dim.get(src_key) or {}
        if isinstance(d, dict) and isinstance(d.get("mean"), (int, float)):
            out[out_key] = float(d["mean"])


def _fill_scoring(out: dict[str, float], interview_id: str) -> None:
    """Read the cached scoring report — same path scoring/report.py uses."""
    path = resolve(f"artifacts/scoring/{interview_id}_report.json")
    if not Path(path).exists():
        return
    try:
        report = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return
    overall = report.get("overall") or {}
    if isinstance(overall.get("technical_avg"), (int, float)):
        out["technical_avg"] = float(overall["technical_avg"])
    if isinstance(overall.get("coherence_avg"), (int, float)):
        out["coherence_avg"] = float(overall["coherence_avg"])
    rec = overall.get("recommendation")
    if isinstance(rec, str) and rec in _RECOMMENDATION_TIER:
        out["recommendation_tier"] = _RECOMMENDATION_TIER[rec]


def _safe_len(raw: Optional[str]) -> int:
    if not raw:
        return 0
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return 0
    return len(data) if isinstance(data, list) else 0
