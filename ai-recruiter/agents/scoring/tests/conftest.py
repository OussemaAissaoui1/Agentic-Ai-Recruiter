"""Shared fixtures for scoring agent tests."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import pytest

from agents.recruit import db as recruit_db


@pytest.fixture
def recruit_conn(tmp_path: Path) -> sqlite3.Connection:
    """Fresh recruit SQLite DB seeded with one job, application, interview."""
    conn = recruit_db.init_db(tmp_path / "recruit.db")
    now = time.time()

    conn.execute(
        "INSERT INTO jobs (id, title, description, status, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("job-test", "Senior Data Engineer",
         "Build pipelines for analytics. Python, Spark, Airflow.",
         "open", now, now),
    )
    conn.execute(
        "INSERT INTO applications (id, job_id, candidate_name, candidate_email, cv_text, "
        "stage, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("app-test", "job-test", "Sarah Chen", "sarah@example.com",
         "10 years building data platforms at Google. Python, Spark, BigQuery.",
         "approved", now, now),
    )
    transcript = [
        {"q": "Walk me through your BigQuery migration.",
         "a": "We moved 50TB from Hive to BigQuery using a dual-write pattern over 6 weeks."},
        {"q": "How did you validate row counts?",
         "a": "Daily reconciliation jobs comparing checksums per partition; mismatches paged us."},
        {"q": "What was the trickiest schema change?",
         "a": "Nested JSON columns - we flattened them with dataflow before load."},
    ]
    conn.execute(
        "INSERT INTO interviews (id, application_id, transcript_json, scores_json, "
        "behavioral_json, status, started_at, ended_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("int-test", "app-test", json.dumps(transcript), "{}", "{}", "completed",
         now - 3600, now),
    )
    return conn


@pytest.fixture
def empty_interview_conn(tmp_path: Path) -> sqlite3.Connection:
    """Recruit DB with an interview row whose transcript_json is empty."""
    conn = recruit_db.init_db(tmp_path / "recruit_empty.db")
    now = time.time()
    conn.execute(
        "INSERT INTO jobs (id, title, description, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?)",
        ("job-empty", "DE", "JD", now, now),
    )
    conn.execute(
        "INSERT INTO applications (id, job_id, candidate_name, cv_text, "
        "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        ("app-empty", "job-empty", "Anon", "cv", now, now),
    )
    conn.execute(
        "INSERT INTO interviews (id, application_id, transcript_json) VALUES (?, ?, ?)",
        ("int-empty", "app-empty", "[]"),
    )
    return conn


class FakeGroq:
    """Test double for GroqClient. Records calls; replies from a queue."""

    def __init__(self, replies: list[Any]) -> None:
        self._replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    async def chat_json(self, *, system: str, user: str, **kw: Any) -> Any:
        self.calls.append({"system": system, "user": user, **kw})
        if not self._replies:
            raise AssertionError("FakeGroq exhausted; add more replies")
        item = self._replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def aclose(self) -> None:
        return None


@pytest.fixture
def fake_groq_factory():
    def _make(replies: list[Any]) -> FakeGroq:
        return FakeGroq(replies)
    return _make
