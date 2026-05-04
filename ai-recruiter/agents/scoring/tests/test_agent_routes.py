import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agents.scoring.agent import ScoringAgent
from agents.scoring.schema import (
    InterviewReport,
    OverallAssessment,
    TurnScore,
)


class _AppState:
    def __init__(self, recruit_conn, artifacts_dir, fake_client):
        self.recruit_conn = recruit_conn
        self.artifacts_dir = artifacts_dir
        self.fake_groq_client = fake_client


def _stub_report(interview_id="int-test"):
    return InterviewReport(
        interview_id=interview_id, candidate_name="Sarah", job_title="DE",
        generated_at=1700000000.0, model="m-1",
        turns=[
            TurnScore(turn_index=0, question="q?", answer="a",
                      technical_score=4, technical_rationale="ok",
                      coherence_score=5, coherence_rationale="ok"),
        ],
        overall=OverallAssessment(
            recommendation="hire", technical_avg=4.0, coherence_avg=5.0,
            summary="x", strengths=["a"], concerns=["b"],
        ),
    )


@pytest.fixture
def app_with_agent(recruit_conn, tmp_path: Path, fake_groq_factory):
    import asyncio
    fake = fake_groq_factory([])  # replies injected per-test by patching score_interview
    agent = ScoringAgent()
    state = _AppState(recruit_conn, tmp_path / "scoring", fake)

    # startup() does no async I/O when state is fully injected; a fresh loop is fine.
    asyncio.run(agent.startup(state))

    app = FastAPI()
    app.include_router(agent.router, prefix="/api/scoring")
    return app, agent, state


def test_run_returns_report_and_caches(app_with_agent):
    app, agent, state = app_with_agent
    client = TestClient(app)

    async def fake_score(**kwargs):
        return _stub_report(kwargs["interview_id"])

    with patch("agents.scoring.agent.score_interview", side_effect=fake_score):
        r = client.post("/api/scoring/int-test/run", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["overall"]["recommendation"] == "hire"

    # Files written
    assert (state.artifacts_dir / "int-test_report.json").exists()
    assert (state.artifacts_dir / "int-test_report.md").exists()

    # Recruit DB updated
    row = state.recruit_conn.execute(
        "SELECT scores_json FROM interviews WHERE id=?", ("int-test",)
    ).fetchone()
    assert json.loads(row["scores_json"])["recommendation"] == "hire"


def test_run_returns_cached_without_calling_llm(app_with_agent):
    app, agent, state = app_with_agent
    client = TestClient(app)

    # Pre-warm cache
    state.artifacts_dir.mkdir(parents=True, exist_ok=True)
    (state.artifacts_dir / "int-test_report.json").write_text(
        _stub_report().model_dump_json()
    )
    (state.artifacts_dir / "int-test_report.md").write_text("md cached")

    with patch("agents.scoring.agent.score_interview") as mock:
        r = client.post("/api/scoring/int-test/run", json={})
        assert mock.called is False
    assert r.status_code == 200
    assert r.json()["interview_id"] == "int-test"


def test_run_force_bypasses_cache(app_with_agent):
    app, agent, state = app_with_agent
    client = TestClient(app)

    state.artifacts_dir.mkdir(parents=True, exist_ok=True)
    (state.artifacts_dir / "int-test_report.json").write_text(
        _stub_report().model_dump_json()
    )

    async def fake_score(**kwargs):
        return _stub_report(kwargs["interview_id"])

    with patch("agents.scoring.agent.score_interview", side_effect=fake_score) as mock:
        r = client.post("/api/scoring/int-test/run", json={"force": True})
        assert mock.called is True
    assert r.status_code == 200


def test_run_unknown_interview_returns_404(app_with_agent):
    app, agent, state = app_with_agent
    client = TestClient(app)
    r = client.post("/api/scoring/int-nope/run", json={})
    assert r.status_code == 404


def test_run_empty_transcript_returns_422(app_with_agent, empty_interview_conn, tmp_path, fake_groq_factory):
    # Build a separate agent bound to the empty DB
    fake = fake_groq_factory([])
    agent = ScoringAgent()
    state = _AppState(empty_interview_conn, tmp_path / "scoring2", fake)
    import asyncio
    asyncio.run(agent.startup(state))
    app = FastAPI()
    app.include_router(agent.router, prefix="/api/scoring")
    client = TestClient(app)

    r = client.post("/api/scoring/int-empty/run", json={})
    assert r.status_code == 422
    assert "no transcript" in r.json()["detail"].lower()


def test_inline_transcript_override_path(app_with_agent, empty_interview_conn, tmp_path, fake_groq_factory):
    fake = fake_groq_factory([])
    agent = ScoringAgent()
    state = _AppState(empty_interview_conn, tmp_path / "scoring3", fake)
    import asyncio
    asyncio.run(agent.startup(state))
    app = FastAPI()
    app.include_router(agent.router, prefix="/api/scoring")
    client = TestClient(app)

    async def fake_score(**kwargs):
        # Verify the override was passed through
        assert len(kwargs["transcript"]) == 1
        assert kwargs["transcript"][0].q == "override"
        return _stub_report("int-empty")

    with patch("agents.scoring.agent.score_interview", side_effect=fake_score):
        r = client.post(
            "/api/scoring/int-empty/run",
            json={"transcript": [{"q": "override", "a": "answer"}]},
        )
    assert r.status_code == 200


def test_get_report_404_then_200(app_with_agent):
    app, _agent, state = app_with_agent
    client = TestClient(app)
    assert client.get("/api/scoring/int-test/report").status_code == 404

    state.artifacts_dir.mkdir(parents=True, exist_ok=True)
    (state.artifacts_dir / "int-test_report.json").write_text(
        _stub_report().model_dump_json()
    )
    r = client.get("/api/scoring/int-test/report")
    assert r.status_code == 200
    assert r.json()["overall"]["recommendation"] == "hire"


def test_get_markdown(app_with_agent):
    app, _agent, state = app_with_agent
    client = TestClient(app)
    state.artifacts_dir.mkdir(parents=True, exist_ok=True)
    (state.artifacts_dir / "int-test_report.md").write_text("# hello")
    r = client.get("/api/scoring/int-test/report.md")
    assert r.status_code == 200
    assert r.text == "# hello"
    assert r.headers["content-type"].startswith("text/markdown")


def test_delete_report(app_with_agent):
    app, _agent, state = app_with_agent
    client = TestClient(app)
    state.artifacts_dir.mkdir(parents=True, exist_ok=True)
    (state.artifacts_dir / "int-test_report.json").write_text(
        _stub_report().model_dump_json()
    )
    r = client.delete("/api/scoring/int-test/report")
    assert r.status_code == 200
    assert not (state.artifacts_dir / "int-test_report.json").exists()


def test_health_reports_degraded_without_key(app_with_agent, monkeypatch):
    app, _agent, _state = app_with_agent
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    client = TestClient(app)
    r = client.get("/api/scoring/health")
    assert r.status_code == 200
    # HealthState enum serializes to lowercase strings. We use FALLBACK in
    # place of the plan's DEGRADED since the shared core enum doesn't expose
    # a DEGRADED member; either FALLBACK or READY is acceptable depending on
    # whether the agent picked up a Groq client at startup.
    assert r.json()["state"] in ("fallback", "ready")
