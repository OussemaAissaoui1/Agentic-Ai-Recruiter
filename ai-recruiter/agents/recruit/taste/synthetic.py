"""Synthetic data for the recruiter-taste POC.

Four ground-truth recruiter personas as fixed weight vectors (in the
canonical FEATURE_NAMES basis). For each persona we can sample plausible
candidate feature vectors and label them with the persona's own
sigmoid(W·x + b) preference, plus a bit of noise. The learner is judged
by how well it recovers the persona's W from those decisions.

This is deliberately small — the personas are crude, the feature
distributions are uniform, and the noise is a single scalar. POC scope.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .features import FEATURE_NAMES


# ─── Persona ground-truth weights ─────────────────────────────────────────────
#
# Each entry is the relative weight on the corresponding FEATURE_NAME. The
# learner sees only labels — never these vectors — so successful recovery
# means the inferred weights correlate with these.
#
# Sign convention: positive = "I like more of this".
#
# Pragmatist:        cares about delivered output. Weights technical
#                    correctness, the AI recommendation tier, and CV/JD
#                    fit. Penalises high cognitive load (signs of struggle).
# Potential-spotter: looks past polish — cares about engagement and
#                    confidence over technical_avg. Tolerant of missing
#                    skills if the candidate seems engaged.
# Communicator:      coherence-first. Strong positive on coherence + low
#                    cognitive load + confidence. Less weight on raw
#                    technical_avg.
# Generalist:        everything balanced — no single dominant feature.
#                    Acts as the control persona.

_PERSONA_WEIGHTS: dict[str, dict[str, float]] = {
    "pragmatist": {
        "fit_score": 1.5,
        "hard_skill_overlap": 0.6,
        "missing_skills_count": -0.8,
        "technical_avg": 1.8,
        "coherence_avg": 0.4,
        "recommendation_tier": 1.2,
        "engagement_mean": 0.2,
        "confidence_mean": 0.3,
        "cognitive_load_mean": -0.9,
        "emotional_arousal_mean": -0.1,
    },
    "potential_spotter": {
        "fit_score": 0.4,
        "hard_skill_overlap": 0.2,
        "missing_skills_count": -0.2,   # tolerant
        "technical_avg": 0.5,
        "coherence_avg": 0.6,
        "recommendation_tier": 0.5,
        "engagement_mean": 1.7,
        "confidence_mean": 1.5,
        "cognitive_load_mean": -0.3,
        "emotional_arousal_mean": 0.4,
    },
    "communicator": {
        "fit_score": 0.4,
        "hard_skill_overlap": 0.2,
        "missing_skills_count": -0.3,
        "technical_avg": 0.5,
        "coherence_avg": 1.9,
        "recommendation_tier": 0.6,
        "engagement_mean": 0.7,
        "confidence_mean": 1.1,
        "cognitive_load_mean": -1.0,
        "emotional_arousal_mean": -0.4,
    },
    "generalist": {
        "fit_score": 0.7,
        "hard_skill_overlap": 0.5,
        "missing_skills_count": -0.5,
        "technical_avg": 0.7,
        "coherence_avg": 0.7,
        "recommendation_tier": 0.7,
        "engagement_mean": 0.6,
        "confidence_mean": 0.6,
        "cognitive_load_mean": -0.5,
        "emotional_arousal_mean": -0.1,
    },
}


PERSONAS: tuple[str, ...] = tuple(_PERSONA_WEIGHTS.keys())


@dataclass(frozen=True)
class SyntheticDecision:
    features: dict[str, float]
    decision: str  # "approved" | "rejected"
    score: float   # the persona's true probability for that candidate (debug)


def persona_weight_vector(persona: str) -> list[float]:
    """Ground-truth weight vector in FEATURE_NAMES order."""
    weights = _PERSONA_WEIGHTS[persona]
    return [float(weights[name]) for name in FEATURE_NAMES]


def generate_synthetic_decisions(
    persona: str,
    n: int,
    *,
    noise: float = 0.4,
    seed: int = 42,
) -> list[SyntheticDecision]:
    """Sample *n* plausible candidates and label them per *persona*.

    The persona's preference is sigmoid(W · x + b) with b chosen so that
    roughly half the population is approved. *noise* is a scalar standard
    deviation added to the logit before binarising — bigger noise =
    harder learning problem.
    """
    if persona not in _PERSONA_WEIGHTS:
        raise ValueError(f"unknown persona: {persona!r}; pick from {PERSONAS}")
    if n <= 0:
        raise ValueError("n must be positive")

    rng = random.Random(seed)
    weights = persona_weight_vector(persona)

    out: list[SyntheticDecision] = []
    for _ in range(n):
        feats = _sample_candidate(rng)
        logit = sum(w * feats[name] for w, name in zip(weights, FEATURE_NAMES))
        # Centre the logit at zero — makes the persona's approval rate
        # roughly balanced regardless of weight magnitudes.
        bias = _persona_bias(weights)
        eps = rng.gauss(0.0, noise)
        prob = _sigmoid(logit - bias + eps)
        decision = "approved" if rng.random() < prob else "rejected"
        out.append(SyntheticDecision(features=feats, decision=decision, score=prob))
    return out


# ─── internals ────────────────────────────────────────────────────────────────


def _sample_candidate(rng: random.Random) -> dict[str, float]:
    """Plausible candidate feature distribution.

    Ranges are eyeballed from the live feature semantics:
      - 0..1 for vision per-dimension means
      - 0..5 for technical/coherence avgs
      - 0..3 for recommendation_tier
      - 0..1 for fit_score
      - 0..15 for hard_skill_overlap and missing_skills_count
    """
    return {
        "fit_score": rng.uniform(0.2, 1.0),
        "hard_skill_overlap": float(rng.randint(0, 12)),
        "missing_skills_count": float(rng.randint(0, 8)),
        "technical_avg": rng.uniform(0.0, 5.0),
        "coherence_avg": rng.uniform(0.0, 5.0),
        "recommendation_tier": float(rng.randint(0, 3)),
        "engagement_mean": rng.uniform(0.0, 1.0),
        "confidence_mean": rng.uniform(0.0, 1.0),
        "cognitive_load_mean": rng.uniform(0.0, 1.0),
        "emotional_arousal_mean": rng.uniform(0.0, 1.0),
    }


def _persona_bias(weights: list[float]) -> float:
    """Logit offset that targets ~50% approval rate for a uniform sampler.

    Computed from the expected value of W·x under the sampler in
    _sample_candidate (each feature is uniform in its declared range).
    """
    midpoints = {
        "fit_score": 0.6,
        "hard_skill_overlap": 6.0,
        "missing_skills_count": 4.0,
        "technical_avg": 2.5,
        "coherence_avg": 2.5,
        "recommendation_tier": 1.5,
        "engagement_mean": 0.5,
        "confidence_mean": 0.5,
        "cognitive_load_mean": 0.5,
        "emotional_arousal_mean": 0.5,
    }
    return sum(w * midpoints[name] for w, name in zip(weights, FEATURE_NAMES))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)
