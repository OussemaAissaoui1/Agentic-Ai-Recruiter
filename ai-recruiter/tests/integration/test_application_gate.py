"""Integration tests for the candidate approval gate.

These tests mount only the recruit agent's router against an isolated SQLite
database (one fresh DB per test) so they don't touch the dev DB and don't
need the full unified app's heavy ML startup.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agents.recruit import db as recruit_db
from agents.recruit.agent import RecruitAgent


@pytest.fixture
def client(tmp_path):
    app = FastAPI()
    agent = RecruitAgent()
    agent._db_path = tmp_path / "recruit.db"
    agent._conn = recruit_db.init_db(agent._db_path)
    recruit_db.seed_if_empty(agent._conn)
    app.include_router(agent.router, prefix="/api/recruit")
    with TestClient(app) as c:
        yield c
    agent._conn.close()


@pytest.fixture
def seeded_job_id(client) -> str:
    jobs = client.get("/api/recruit/jobs").json()
    assert jobs, "seed jobs missing"
    return jobs[0]["id"]


def test_fixture_boots_and_seeds_jobs(client, seeded_job_id):
    assert seeded_job_id.startswith("job-")


@pytest.fixture
def applied_app(client, seeded_job_id) -> dict:
    r = client.post(
        "/api/recruit/applications",
        data={"job_id": seeded_job_id, "candidate_email": "ada@example.com",
              "candidate_name": "Ada Lovelace"},
    )
    assert r.status_code == 200, r.text
    return r.json()


def test_apply_starts_in_applied_stage(applied_app):
    assert applied_app["stage"] == "applied"


def test_invalid_stage_rejected(client, applied_app):
    r = client.patch(
        f"/api/recruit/applications/{applied_app['id']}",
        json={"stage": "screening"},
    )
    assert r.status_code == 400
    assert "screening" in r.text.lower() or "stage" in r.text.lower()


def test_legacy_stage_migration_runs_on_init(tmp_path):
    """Rows with legacy stage values get rewritten to 'approved' at init."""
    db_path = tmp_path / "legacy.db"
    conn = recruit_db.init_db(db_path)
    recruit_db.seed_if_empty(conn)
    job_id = recruit_db.list_jobs(conn)[0]["id"]
    a = recruit_db.create_application(conn, {"job_id": job_id, "candidate_email": "x@y"})
    b = recruit_db.create_application(conn, {"job_id": job_id, "candidate_email": "y@z"})
    conn.execute("UPDATE applications SET stage = 'screening' WHERE id = ?", (a["id"],))
    conn.execute("UPDATE applications SET stage = 'interview' WHERE id = ?", (b["id"],))
    conn.close()

    conn2 = recruit_db.init_db(db_path)
    a2 = recruit_db.get_application(conn2, a["id"])
    b2 = recruit_db.get_application(conn2, b["id"])
    conn2.close()
    assert a2["stage"] == "approved"
    assert b2["stage"] == "approved"


def test_approve_sets_stage_and_notifies(client, applied_app):
    r = client.post(f"/api/recruit/applications/{applied_app['id']}/invite-interview")
    assert r.status_code == 200, r.text

    app_after = client.get(f"/api/recruit/applications/{applied_app['id']}").json()
    assert app_after["stage"] == "approved"

    notifs = client.get(
        "/api/recruit/notifications",
        params={"user_role": "candidate", "user_id": "ada@example.com"},
    ).json()
    invite = [n for n in notifs if n["kind"] == "interview_invite"]
    assert len(invite) == 1
    assert invite[0]["link"] == f"/c/interview/{applied_app['id']}"


def test_approve_is_idempotent(client, applied_app):
    client.post(f"/api/recruit/applications/{applied_app['id']}/invite-interview")
    client.post(f"/api/recruit/applications/{applied_app['id']}/invite-interview")

    notifs = client.get(
        "/api/recruit/notifications",
        params={"user_role": "candidate", "user_id": "ada@example.com"},
    ).json()
    invite = [n for n in notifs if n["kind"] == "interview_invite"]
    assert len(invite) == 1, f"expected 1 invite notification, got {len(invite)}"


def test_reject_creates_notification(client, applied_app):
    r = client.patch(
        f"/api/recruit/applications/{applied_app['id']}",
        json={"stage": "rejected"},
    )
    assert r.status_code == 200, r.text

    notifs = client.get(
        "/api/recruit/notifications",
        params={"user_role": "candidate", "user_id": "ada@example.com"},
    ).json()
    rejected = [n for n in notifs if n["kind"] == "application_rejected"]
    assert len(rejected) == 1
    assert rejected[0]["link"] == "/c/applications"


def test_reject_idempotent(client, applied_app):
    client.patch(f"/api/recruit/applications/{applied_app['id']}",
                 json={"stage": "rejected"})
    client.patch(f"/api/recruit/applications/{applied_app['id']}",
                 json={"stage": "rejected"})

    notifs = client.get(
        "/api/recruit/notifications",
        params={"user_role": "candidate", "user_id": "ada@example.com"},
    ).json()
    rejected = [n for n in notifs if n["kind"] == "application_rejected"]
    assert len(rejected) == 1


def test_create_interview_refuses_when_applied(client, applied_app):
    r = client.post(
        "/api/recruit/interviews",
        json={"application_id": applied_app["id"]},
    )
    assert r.status_code == 403
    assert "approved" in r.text.lower() or "stage" in r.text.lower()


def test_create_interview_allowed_when_approved(client, applied_app):
    client.post(f"/api/recruit/applications/{applied_app['id']}/invite-interview")
    r = client.post(
        "/api/recruit/interviews",
        json={"application_id": applied_app["id"]},
    )
    assert r.status_code == 200, r.text


def test_create_interview_allowed_when_interviewed(client, applied_app):
    client.patch(
        f"/api/recruit/applications/{applied_app['id']}",
        json={"stage": "interviewed"},
    )
    r = client.post(
        "/api/recruit/interviews",
        json={"application_id": applied_app["id"]},
    )
    assert r.status_code == 200, r.text
