"""Recruiter-taste learning: per-recruiter preference fitting on top of
hire/reject decisions. Lives inside the recruit agent because it owns the
applications + decisions data.

Modules:
  features.py  — flat feature-vector cache, keyed by (app_id, version)
  learner.py   — sklearn LR fit + score; confidence tiers
  synthetic.py — 4 ground-truth personas + decision generator (POC seeding)
  cli.py       — `python -m agents.recruit.taste.cli seed_synthetic_recruiter`
"""

from .features import FEATURE_NAMES, FEATURES_VERSION, get_or_compute_features  # noqa: F401
from .learner import (  # noqa: F401
    Confidence,
    RecruiterScore,
    fit_recruiter_weights,
    load_profile,
    reset_profile,
    score_candidate,
)
