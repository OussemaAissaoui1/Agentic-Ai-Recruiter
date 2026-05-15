"""Foundation tests for Phase 1 — graph schema + seed payload integrity.

Unit tests (no external dependencies):
    * Synthetic payload referential integrity (every employee skill_id resolves,
      every team_id exists, every roles_held.role_id exists).
    * Past-JD rejection invariants (3 of 10 are rejected, each with reason text).
    * Schema statements are well-formed Cypher strings.
    * Cohort floor — every (family, level) cohort has at least 3 employees so
      the agent's MIN_COHORT guard rarely triggers.

Integration tests (require live Neo4j, marked @pytest.mark.integration):
    * apply_schema is idempotent.
    * seed() loads the expected counts and Neo4j returns them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

DATA = (
    Path(__file__).resolve().parents[2]
    / "agents"
    / "jd_generation"
    / "data"
    / "synthetic_employees.json"
)


# ---------------------------------------------------------------------------
# Payload loading — module scope so we read the JSON once
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def payload() -> Dict[str, Any]:
    return json.loads(DATA.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Headcounts
# ---------------------------------------------------------------------------
def test_meta_headcounts_match_arrays(payload: Dict[str, Any]) -> None:
    meta = payload["_meta"]
    assert meta["employee_count"] == len(payload["employees"]) == 100
    assert meta["skill_count"] == len(payload["skills"])
    assert meta["role_count"] == len(payload["roles"]) == 15  # 5 families x 3 levels
    assert meta["team_count"] == len(payload["teams"]) == 5
    assert meta["past_jd_count"] == len(payload["past_jds"]) == 10


def test_all_role_families_represented(payload: Dict[str, Any]) -> None:
    families = {r["role_family"] for r in payload["roles"]}
    assert families == {"backend", "frontend", "data_science", "product", "design"}


# ---------------------------------------------------------------------------
# Referential integrity — every edge target exists
# ---------------------------------------------------------------------------
def test_employee_skill_ids_resolve(payload: Dict[str, Any]) -> None:
    skill_ids = {s["id"] for s in payload["skills"]}
    for emp in payload["employees"]:
        for s in emp["skills"]:
            assert s["skill_id"] in skill_ids, f"unknown skill_id in {emp['id']}"


def test_employee_role_ids_resolve(payload: Dict[str, Any]) -> None:
    role_ids = {r["id"] for r in payload["roles"]}
    for emp in payload["employees"]:
        for r in emp["roles_held"]:
            assert r["role_id"] in role_ids, f"unknown role_id in {emp['id']}"


def test_employee_team_ids_resolve(payload: Dict[str, Any]) -> None:
    team_ids = {t["id"] for t in payload["teams"]}
    for emp in payload["employees"]:
        if emp.get("team_id"):
            assert emp["team_id"] in team_ids, f"unknown team_id in {emp['id']}"


def test_past_jd_role_families_resolve(payload: Dict[str, Any]) -> None:
    families = {r["role_family"] for r in payload["roles"]}
    for jd in payload["past_jds"]:
        assert jd["role_family"] in families


# ---------------------------------------------------------------------------
# Rejection-reason invariants — the feedback-loop fuel
# ---------------------------------------------------------------------------
def test_three_rejected_jds(payload: Dict[str, Any]) -> None:
    rejected = [j for j in payload["past_jds"] if j["status"] == "rejected"]
    assert len(rejected) == 3
    for jd in rejected:
        assert "rejection" in jd
        rej = jd["rejection"]
        assert rej["text"].strip(), "rejection text must be non-empty"
        assert rej["categories"], "rejection categories must be non-empty"


def test_jargon_rejection_is_for_backend(payload: Dict[str, Any]) -> None:
    """Acceptance-test setup: backend role family must have at least one
    rejection that mentions jargon. The integration test in
    test_rejection_loop.py asserts the agent acts on this."""
    backend_rejections = [
        j for j in payload["past_jds"]
        if j["status"] == "rejected" and j["role_family"] == "backend"
    ]
    assert backend_rejections, "need at least one backend rejection for the acceptance test"
    assert any("jargon" in j["rejection"]["text"].lower() for j in backend_rejections)


def test_seven_good_hires(payload: Dict[str, Any]) -> None:
    good = [j for j in payload["past_jds"] if j["status"] == "approved"
            and j["hire_outcome"] == "good_hire"]
    assert len(good) == 7


# ---------------------------------------------------------------------------
# Cohort floor — agent's MIN_COHORT=3 rule must rarely trigger
# ---------------------------------------------------------------------------
def test_every_family_level_cohort_has_minimum(payload: Dict[str, Any]) -> None:
    by_family_level: Dict[tuple, int] = {}
    role_by_id = {r["id"]: r for r in payload["roles"]}
    for emp in payload["employees"]:
        for r in emp["roles_held"]:
            if not r.get("current"):
                continue
            role = role_by_id[r["role_id"]]
            key = (role["role_family"], role["level"])
            by_family_level[key] = by_family_level.get(key, 0) + 1

    families = {"backend", "frontend", "data_science", "product", "design"}
    levels = {"junior", "mid", "senior"}
    for family in families:
        for level in levels:
            count = by_family_level.get((family, level), 0)
            assert count >= 3, f"cohort ({family},{level}) has {count} employees (<3)"


# ---------------------------------------------------------------------------
# Schema statements are syntactically well-formed
# ---------------------------------------------------------------------------
def test_schema_statements_well_formed() -> None:
    from agents.jd_generation.graph.schema import _CONSTRAINTS, _INDEXES

    assert len(_CONSTRAINTS) >= 8, "must have a unique-id constraint per node label"
    assert len(_INDEXES) >= 4
    for stmt in _CONSTRAINTS:
        assert stmt.upper().startswith("CREATE CONSTRAINT")
        assert "IF NOT EXISTS" in stmt
    for stmt in _INDEXES:
        assert stmt.upper().startswith("CREATE INDEX")
        assert "IF NOT EXISTS" in stmt


def test_config_loads_without_env() -> None:
    """Config loads in any environment; consumers check is_configured."""
    from agents.jd_generation.config import load_jd_config

    cfg = load_jd_config()
    assert cfg.agent.max_tool_calls == 15
    assert cfg.agent.deterministic_preambles[:2] == [
        "find_role_family", "get_past_rejections"
    ]
    assert cfg.llm.backend == "groq"
    assert cfg.llm.model == "llama-3.3-70b-versatile"


# ---------------------------------------------------------------------------
# Integration — live Neo4j required
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_apply_schema_idempotent(requires_neo4j) -> None:  # noqa: ARG001
    from agents.jd_generation.graph.client import get_client, close_client
    from agents.jd_generation.graph.schema import apply_schema

    try:
        client = get_client()
        r1 = apply_schema(client)
        r2 = apply_schema(client)
        assert r1["statements_applied"] == r2["statements_applied"]
    finally:
        close_client()


@pytest.mark.integration
def test_seed_loads_expected_counts(requires_neo4j) -> None:  # noqa: ARG001
    from agents.jd_generation.graph.client import close_client, get_client
    from agents.jd_generation.graph.seed import seed

    try:
        summary = seed(clear=True, quiet=True)
        assert summary["nodes"]["Employee"] == 100
        assert summary["nodes"]["Role"] == 15
        assert summary["nodes"]["Team"] == 5
        assert summary["nodes"]["RejectionReason"] == 3
        assert summary["edges"]["HOLDS"] == 100

        client = get_client()
        rec = client.read("MATCH (e:Employee) RETURN count(e) AS n")[0]
        assert rec["n"] == 100
    finally:
        close_client()
