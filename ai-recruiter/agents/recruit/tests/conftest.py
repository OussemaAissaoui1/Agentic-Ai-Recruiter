"""Shared fixtures for recruit agent tests.

Mirrors the agents/scoring/tests/conftest.py pattern: a tmp-path SQLite
DB seeded with the canonical jobs/applications/interviews rows.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pytest

from agents.recruit import db as recruit_db


@pytest.fixture
def recruit_conn(tmp_path: Path) -> sqlite3.Connection:
    """Fresh recruit SQLite DB with a job + approved application + interview."""
    conn = recruit_db.init_db(tmp_path / "recruit.db")
    now = time.time()

    conn.execute(
        "INSERT INTO jobs (id, title, description, status, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("job-test", "Senior Software Engineer",
         "Backend services in Python. Strong systems background expected.",
         "open", now, now),
    )
    conn.execute(
        "INSERT INTO applications (id, job_id, candidate_name, candidate_email, "
        "cv_text, fit_score, matched_skills_json, missing_skills_json, "
        "stage, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("app-test", "job-test", "Alex Doe", "alex@example.com",
         "Python, Django, PostgreSQL, 6 years.",
         0.78,
         json.dumps(["Python", "PostgreSQL", "Django"]),
         json.dumps(["Kafka"]),
         "interviewed", now, now),
    )
    transcript = [
        {"q": "Tell me about your last project.",
         "a": "Migrated a Django monolith to async Python with FastAPI."},
    ]
    behavioral = {
        "duration_seconds": 540.0,
        "frames_analyzed": 2150,
        "baseline_calibrated": True,
        "overall_arc": "stable",
        "per_dimension": {
            "engagement_level":   {"mean": 0.65, "std": 0.10, "min": 0.4, "max": 0.85, "trend": 0.001},
            "confidence_level":   {"mean": 0.60, "std": 0.12, "min": 0.3, "max": 0.80, "trend": 0.000},
            "cognitive_load":     {"mean": 0.40, "std": 0.15, "min": 0.1, "max": 0.75, "trend": 0.001},
            "emotional_arousal":  {"mean": 0.30, "std": 0.10, "min": 0.1, "max": 0.55, "trend": -0.001},
        },
    }
    conn.execute(
        "INSERT INTO interviews (id, application_id, transcript_json, scores_json, "
        "behavioral_json, status, started_at, ended_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("int-test", "app-test", json.dumps(transcript), "{}",
         json.dumps(behavioral), "completed", now - 3600, now),
    )
    return conn
