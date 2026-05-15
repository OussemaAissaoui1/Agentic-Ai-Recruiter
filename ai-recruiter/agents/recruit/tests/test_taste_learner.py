"""Acceptance test: the learner recovers the persona's preferences.

For each of the 4 ground-truth personas, generate 200 synthetic decisions,
fit weights via the learner, and require Pearson correlation > 0.7
between recovered weights and the persona's true weights.

This is the success-criterion gate for the learning loop. If this test
fails, the loop is broken — don't ship the rest.
"""
from __future__ import annotations

import math
import sqlite3
import time

import pytest

from agents.recruit.taste import (
    Confidence,
    fit_recruiter_weights,
    score_candidate,
)
from agents.recruit.taste.features import FEATURE_NAMES
from agents.recruit.taste.synthetic import (
    PERSONAS,
    generate_synthetic_decisions,
    persona_weight_vector,
)


def _seed_decisions(
    conn: sqlite3.Connection, recruiter_id: str, persona: str, n: int,
) -> None:
    """Push synthetic decisions straight into recruiter_decisions.

    Bypasses the API capture hook (we test that separately) — this fixture
    is purely about whether the learner can fit. Each row's
    features_snapshot_json holds the synthetic feature dict; minimal
    application rows are seeded too so the FK on application_id holds.
    """
    import json as _json
    decisions = generate_synthetic_decisions(persona, n)
    for i, d in enumerate(decisions):
        app_id = f"synth-{recruiter_id}-{i}"
        conn.execute(
            "INSERT OR IGNORE INTO applications (id, job_id, stage, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (app_id, "job-test", "applied", time.time(), time.time()),
        )
        conn.execute(
            "INSERT INTO recruiter_decisions (id, recruiter_id, application_id, "
            "features_snapshot_json, features_version, decision, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (f"dec-{recruiter_id}-{i}", recruiter_id, app_id,
             _json.dumps(d.features), 1, d.decision, time.time()),
        )


def _pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n != len(b) or n < 2:
        return 0.0
    ma = sum(a) / n
    mb = sum(b) / n
    num = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    da = math.sqrt(sum((a[i] - ma) ** 2 for i in range(n)))
    db = math.sqrt(sum((b[i] - mb) ** 2 for i in range(n)))
    if da == 0.0 or db == 0.0:
        return 0.0
    return num / (da * db)


@pytest.mark.parametrize("persona", PERSONAS)
def test_learner_recovers_persona_weights(
    persona: str, recruit_conn: sqlite3.Connection,
) -> None:
    """500 decisions per persona; recovered weights ≈ true weights.

    Personas with weaker / more-balanced weight vectors (potential_spotter,
    generalist) need more data than the strong-signal ones (pragmatist,
    communicator) to recover at the 0.7 Pearson threshold. 500 is enough
    for all four under the default noise.
    """
    rid = f"rec-{persona}"
    _seed_decisions(recruit_conn, rid, persona, n=500)

    profile = fit_recruiter_weights(recruit_conn, rid)

    # The learner z-scores features before fitting, so its stored weights
    # are on a standardised scale. Convert back to raw scale for a direct
    # comparison with the persona's ground-truth (raw-scale) weights:
    #   w_raw ≈ w_z / std(feature)
    recovered_raw = [
        profile.weights[name] / profile.feature_stds[name]
        for name in FEATURE_NAMES
    ]
    truth = persona_weight_vector(persona)

    rho = _pearson(recovered_raw, truth)
    assert rho > 0.7, (
        f"persona {persona!r}: learner failed to recover weights "
        f"(Pearson={rho:.3f}). recovered_raw={recovered_raw} truth={truth}"
    )
    assert profile.n_decisions == 500
    assert profile.confidence == Confidence.WARM


def test_score_candidate_differs_across_personas(
    recruit_conn: sqlite3.Connection,
) -> None:
    """Same candidate, four personas → at least one pair differs by >0.1.

    This is the demo-gate signal: persona learning has to produce visibly
    different recruiter-fit scores for the same candidate, otherwise the
    whole feature is decorative.
    """
    for persona in PERSONAS:
        rid = f"rec-{persona}"
        _seed_decisions(recruit_conn, rid, persona, n=150)
        fit_recruiter_weights(recruit_conn, rid)

    # Construct a candidate that should split the personas: high coherence
    # (Communicator should love), high engagement+confidence (Potential-spotter),
    # low technical (Pragmatist should reject). Generalist sits in between.
    candidate = {
        "fit_score": 0.5,
        "hard_skill_overlap": 4.0,
        "missing_skills_count": 3.0,
        "technical_avg": 1.8,
        "coherence_avg": 4.7,
        "recommendation_tier": 1.0,
        "engagement_mean": 0.85,
        "confidence_mean": 0.80,
        "cognitive_load_mean": 0.25,
        "emotional_arousal_mean": 0.30,
    }

    scores = {
        persona: score_candidate(recruit_conn, f"rec-{persona}", candidate).score
        for persona in PERSONAS
    }

    # Sanity: scores in [0, 1].
    for s in scores.values():
        assert 0.0 <= s <= 1.0

    # At least one pair should differ by >0.1 — proves personas learned
    # different things from the same feature space.
    spread = max(scores.values()) - min(scores.values())
    assert spread > 0.1, f"persona scores indistinguishable: {scores}"


def test_score_candidate_returns_top_factors(
    recruit_conn: sqlite3.Connection,
) -> None:
    """The UI shows top 3 positive + top 2 negative factors per score."""
    rid = "rec-pragmatist"
    _seed_decisions(recruit_conn, rid, "pragmatist", n=200)
    fit_recruiter_weights(recruit_conn, rid)

    # Candidate constructed to give the pragmatist persona strictly more than
    # 2 negative contributions, so the "top 2" cap is actually exercised.
    # Pragmatist's negative-weighted features: missing_skills_count (-0.8),
    # cognitive_load_mean (-0.9), emotional_arousal_mean (-0.1). Above-mean
    # values for all three should yield three negatives — function should
    # return only the top 2.
    candidate = {
        "fit_score": 0.9,
        "hard_skill_overlap": 8.0,
        "missing_skills_count": 7.0,        # above mean → negative contribution
        "technical_avg": 4.5,
        "coherence_avg": 4.0,
        "recommendation_tier": 3.0,
        "engagement_mean": 0.7,
        "confidence_mean": 0.8,
        "cognitive_load_mean": 0.85,        # above mean → negative contribution
        "emotional_arousal_mean": 0.85,     # above mean → mild negative
    }
    result = score_candidate(recruit_conn, rid, candidate)
    assert len(result.top_positive) == 3
    assert len(result.top_negative) == 2
    # Each entry should be a (label, contribution) pair with a plain-text label.
    for label, _contribution in result.top_positive:
        assert isinstance(label, str) and label
    for label, _contribution in result.top_negative:
        assert isinstance(label, str) and label


def test_confidence_levels_are_correct(recruit_conn: sqlite3.Connection) -> None:
    """cold (<10), warming (10-49), warm (50+)."""
    # Cold: 5 decisions
    _seed_decisions(recruit_conn, "rec-cold", "generalist", n=5)
    p_cold = fit_recruiter_weights(recruit_conn, "rec-cold")
    assert p_cold.confidence == Confidence.COLD

    # Warming: 20
    _seed_decisions(recruit_conn, "rec-warming", "generalist", n=20)
    p_warming = fit_recruiter_weights(recruit_conn, "rec-warming")
    assert p_warming.confidence == Confidence.WARMING

    # Warm: 60
    _seed_decisions(recruit_conn, "rec-warm", "generalist", n=60)
    p_warm = fit_recruiter_weights(recruit_conn, "rec-warm")
    assert p_warm.confidence == Confidence.WARM
