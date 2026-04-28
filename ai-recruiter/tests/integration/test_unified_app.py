"""Smoke test for the unified FastAPI app.

Imports `main:app` and hits `/api/health`, `/.well-known/agents/{name}`,
and a couple of per-agent `/health` endpoints. ML-heavy startup paths
(vLLM, MediaPipe, torch_geometric) are *not* mocked — agents are
instantiated lazily, so the test only fails if a contract is broken.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    # Imports here so collection of this module is cheap when tests skip.
    from main import app
    with TestClient(app) as c:
        yield c


def test_aggregate_health_returns_200(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["version"] == "0.2.0"
    assert "agents" in body
    assert {"matching", "vision", "nlp", "avatar", "scoring",
            "voice", "orchestrator"} <= set(body["agents"].keys())


def test_overall_state_is_known_value(client):
    body = client.get("/api/health").json()
    assert body["overall"] in {"ready", "loading", "fallback", "error"}


@pytest.mark.parametrize("name", ["matching", "vision", "nlp", "avatar",
                                  "scoring", "voice", "orchestrator"])
def test_agent_card_endpoint(client, name):
    r = client.get(f"/.well-known/agents/{name}")
    assert r.status_code == 200
    card = r.json()
    assert card["name"] == name
    assert isinstance(card.get("capabilities", []), list)


def test_unknown_agent_card_returns_404(client):
    r = client.get("/.well-known/agents/does_not_exist")
    assert r.status_code == 404


def test_orchestrator_lists_agents(client):
    r = client.get("/api/orchestrator/agents")
    assert r.status_code == 200
    cards = r.json()
    assert "matching" in cards
    assert "vision" in cards


def test_scoring_record_and_read(client):
    sid = "test-session-001"
    r = client.post(f"/api/scoring/{sid}/score",
                    json={"source": "test", "kind": "smoke", "value": 0.5})
    assert r.status_code == 200
    r2 = client.get(f"/api/scoring/{sid}/scores")
    assert r2.status_code == 200
    assert any(rec["kind"] == "smoke" for rec in r2.json()["scores"])


def test_orchestrator_dispatch_round_trip(client):
    """Ask scoring to record_score via the orchestrator dispatcher."""
    task = {
        "task_id": "t1",
        "sender": "test",
        "target": "scoring",
        "capability": "record_score",
        "payload": {"session_id": "test-session-002",
                    "score": {"source": "test", "kind": "via-dispatch", "value": 1.0}},
    }
    r = client.post("/api/orchestrator/dispatch", json=task)
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
