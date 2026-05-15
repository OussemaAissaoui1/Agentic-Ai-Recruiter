"""Per-recruiter preference learner.

Refit-from-scratch logistic regression with L2. No online SGD, no
incremental updates — every refit reads the recruiter's full
recruiter_decisions history and writes a fresh row to recruiter_profiles.

Public API:
  fit_recruiter_weights(conn, recruiter_id) -> RecruiterProfile
  score_candidate(conn, recruiter_id, features) -> RecruiterScore
  load_profile(conn, recruiter_id) -> Optional[RecruiterProfile]
  reset_profile(conn, recruiter_id) -> bool

The score is sigmoid(W·standardised(features) + intercept). We keep the
standardisation stats in the profile so score_candidate produces a number
on the same scale as the training set.
"""
from __future__ import annotations

import enum
import json
import math
import sqlite3
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression

from ._lock import DB_LOCK
from .features import FEATURE_LABELS, FEATURE_NAMES


class Confidence(str, enum.Enum):
    COLD = "cold"
    WARMING = "warming"
    WARM = "warm"

    @classmethod
    def for_n(cls, n: int) -> "Confidence":
        if n < 5:
            return cls.COLD
        if n < 50:
            return cls.WARMING
        return cls.WARM


@dataclass(frozen=True)
class RecruiterProfile:
    recruiter_id: str
    weights: dict[str, float]          # in FEATURE_NAMES space (z-scored)
    intercept: float
    feature_means: dict[str, float]    # standardisation stats from training
    feature_stds: dict[str, float]
    n_decisions: int
    confidence: Confidence
    version: int
    updated_at: float


@dataclass(frozen=True)
class RecruiterScore:
    score: float                                    # sigmoid in [0, 1]
    confidence: Confidence
    top_positive: list[tuple[str, float]]           # (plain-text label, contribution)
    top_negative: list[tuple[str, float]]


# ─── fit ──────────────────────────────────────────────────────────────────────


def fit_recruiter_weights(
    conn: sqlite3.Connection, recruiter_id: str,
) -> RecruiterProfile:
    """Refit weights from scratch on the recruiter's full decision history.

    Edge cases the POC handles gracefully:
      - Zero decisions → zero-weight profile, COLD.
      - All-same-decision (e.g. only approvals so far) → zero-weight
        profile (no signal to fit), CONFIDENCE based on count.
      - Very few decisions → still fits; confidence tier flags that the
        score isn't trustworthy yet (the UI mutes it).
    """
    with DB_LOCK:
        return _fit_recruiter_weights_locked(conn, recruiter_id)


def _fit_recruiter_weights_locked(
    conn: sqlite3.Connection, recruiter_id: str,
) -> RecruiterProfile:
    rows = conn.execute(
        "SELECT features_snapshot_json, decision FROM recruiter_decisions "
        "WHERE recruiter_id = ? ORDER BY created_at ASC",
        (recruiter_id,),
    ).fetchall()

    n = len(rows)
    if n == 0:
        return _persist_profile(
            conn, recruiter_id,
            weights={name: 0.0 for name in FEATURE_NAMES},
            intercept=0.0,
            feature_means={name: 0.0 for name in FEATURE_NAMES},
            feature_stds={name: 1.0 for name in FEATURE_NAMES},
            n_decisions=0,
        )

    X_list: list[list[float]] = []
    y_list: list[int] = []
    for row in rows:
        try:
            feats = json.loads(row["features_snapshot_json"])
        except json.JSONDecodeError:
            continue
        if not isinstance(feats, dict):
            continue
        X_list.append([float(feats.get(name, 0.0)) for name in FEATURE_NAMES])
        y_list.append(1 if row["decision"] == "approved" else 0)

    if not X_list:
        return _persist_profile(
            conn, recruiter_id,
            weights={name: 0.0 for name in FEATURE_NAMES},
            intercept=0.0,
            feature_means={name: 0.0 for name in FEATURE_NAMES},
            feature_stds={name: 1.0 for name in FEATURE_NAMES},
            n_decisions=n,
        )

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.int64)

    # Standardise. std=0 happens when a feature is constant in this set
    # (e.g. all candidates have technical_avg = 0 because no scoring ran);
    # treat std=0 as 1 so we don't divide by zero. The corresponding
    # weight will collapse to ~0 because every standardised value is 0.
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    safe_stds = np.where(stds < 1e-9, 1.0, stds)
    Xs = (X - means) / safe_stds

    # Single-class case: nothing to fit. Return zero weights.
    if len(set(y_list)) < 2:
        return _persist_profile(
            conn, recruiter_id,
            weights={name: 0.0 for name in FEATURE_NAMES},
            intercept=float(np.log(max(y.mean(), 1e-3) / max(1.0 - y.mean(), 1e-3))),
            feature_means={name: float(m) for name, m in zip(FEATURE_NAMES, means)},
            feature_stds={name: float(s) for name, s in zip(FEATURE_NAMES, safe_stds)},
            n_decisions=n,
        )

    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=500,
        random_state=42,
    )
    clf.fit(Xs, y)
    
    coef = clf.coef_[0]
    weights = {name: float(c) for name, c in zip(FEATURE_NAMES, coef)}
    return _persist_profile(
        conn, recruiter_id,
        weights=weights,
        intercept=float(clf.intercept_[0]),
        feature_means={name: float(m) for name, m in zip(FEATURE_NAMES, means)},
        feature_stds={name: float(s) for name, s in zip(FEATURE_NAMES, safe_stds)},
        n_decisions=n,
    )


# ─── score ────────────────────────────────────────────────────────────────────


def score_candidate(
    conn: sqlite3.Connection, recruiter_id: str, features: dict[str, float],
) -> RecruiterScore:
    """Sigmoid score + top contributing features for the UI."""
    with DB_LOCK:
        profile = load_profile(conn, recruiter_id)
    if profile is None:
        # No profile yet → neutral score, COLD confidence.
        return RecruiterScore(
            score=0.5,
            confidence=Confidence.COLD,
            top_positive=[],
            top_negative=[],
        )

    contributions: list[tuple[str, float]] = []
    logit = profile.intercept
    for name in FEATURE_NAMES:
        x = float(features.get(name, 0.0))
        z = (x - profile.feature_means[name]) / profile.feature_stds[name]
        contrib = profile.weights[name] * z
        logit += contrib
        contributions.append((name, contrib))

    # Sort by contribution magnitude, separated by sign.
    positives = sorted(
        ((n, c) for n, c in contributions if c > 0), key=lambda p: -p[1],
    )[:3]
    negatives = sorted(
        ((n, c) for n, c in contributions if c < 0), key=lambda p: p[1],
    )[:2]

    return RecruiterScore(
        score=_sigmoid(logit),
        confidence=profile.confidence,
        top_positive=[(FEATURE_LABELS[n], c) for n, c in positives],
        top_negative=[(FEATURE_LABELS[n], c) for n, c in negatives],
    )


# ─── persistence ──────────────────────────────────────────────────────────────


def load_profile(
    conn: sqlite3.Connection, recruiter_id: str,
) -> Optional[RecruiterProfile]:
    with DB_LOCK:
        row = conn.execute(
            "SELECT recruiter_id, weights_json, n_decisions, confidence_level, "
            "version, updated_at FROM recruiter_profiles WHERE recruiter_id = ?",
            (recruiter_id,),
        ).fetchone()
    if row is None:
        return None
    try:
        blob = json.loads(row["weights_json"])
    except json.JSONDecodeError:
        return None
    if not isinstance(blob, dict):
        return None
    return RecruiterProfile(
        recruiter_id=row["recruiter_id"],
        weights={k: float(v) for k, v in (blob.get("weights") or {}).items()},
        intercept=float(blob.get("intercept", 0.0)),
        feature_means={k: float(v) for k, v in (blob.get("means") or {}).items()},
        feature_stds={k: float(v) for k, v in (blob.get("stds") or {}).items()},
        n_decisions=int(row["n_decisions"]),
        confidence=Confidence(row["confidence_level"]),
        version=int(row["version"]),
        updated_at=float(row["updated_at"]),
    )


def reset_profile(conn: sqlite3.Connection, recruiter_id: str) -> bool:
    """Wipe the profile (not the decisions). Returns True if a row existed."""
    with DB_LOCK:
        cur = conn.execute(
            "DELETE FROM recruiter_profiles WHERE recruiter_id = ?",
            (recruiter_id,),
        )
        return cur.rowcount > 0


def _persist_profile(
    conn: sqlite3.Connection,
    recruiter_id: str,
    *,
    weights: dict[str, float],
    intercept: float,
    feature_means: dict[str, float],
    feature_stds: dict[str, float],
    n_decisions: int,
) -> RecruiterProfile:
    confidence = Confidence.for_n(n_decisions)
    blob = {
        "weights": weights,
        "intercept": intercept,
        "means": feature_means,
        "stds": feature_stds,
    }
    now = time.time()
    existing = conn.execute(
        "SELECT version FROM recruiter_profiles WHERE recruiter_id = ?",
        (recruiter_id,),
    ).fetchone()
    new_version = (int(existing["version"]) + 1) if existing else 1

    conn.execute(
        "INSERT OR REPLACE INTO recruiter_profiles "
        "(recruiter_id, weights_json, n_decisions, confidence_level, version, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (recruiter_id, json.dumps(blob), n_decisions, confidence.value, new_version, now),
    )

    return RecruiterProfile(
        recruiter_id=recruiter_id,
        weights=weights,
        intercept=intercept,
        feature_means=feature_means,
        feature_stds=feature_stds,
        n_decisions=n_decisions,
        confidence=confidence,
        version=new_version,
        updated_at=now,
    )

def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)
