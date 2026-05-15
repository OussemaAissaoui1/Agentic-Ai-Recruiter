"""Phase 2 tests — graph tools + dispatcher.

Unit tests use an in-memory FakeClient so we don't need Neo4j. The
cypher_query read-only guard and dispatcher contracts are pure-Python and
fully covered here.

Integration tests (@pytest.mark.integration) seed a real graph and exercise
every retrieval query against it.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest


# ---------------------------------------------------------------------------
# FakeClient — records calls and returns canned rows
# ---------------------------------------------------------------------------
class FakeClient:
    def __init__(self, read_returns: Any = None, write_returns: Any = None) -> None:
        self.read_calls: List[Dict[str, Any]] = []
        self.write_calls: List[Dict[str, Any]] = []
        self._read_returns = read_returns if read_returns is not None else []
        self._write_returns = write_returns if write_returns is not None else []

    def read(self, query: str, **params: Any) -> List[Dict[str, Any]]:
        self.read_calls.append({"q": query, "p": params})
        if isinstance(self._read_returns, list) and self._read_returns \
                and isinstance(self._read_returns[0], dict) \
                and "_when" in self._read_returns[0]:
            # Lookup style: list of {"_when": <substr>, "rows": [...]}
            for r in self._read_returns:
                if r["_when"] in query:
                    return r["rows"]
            return []
        return self._read_returns

    def write(self, query: str, **params: Any) -> List[Dict[str, Any]]:
        self.write_calls.append({"q": query, "p": params})
        return self._write_returns


# ===========================================================================
# cypher_query read-only enforcement
# ===========================================================================
def test_cypher_query_rejects_create() -> None:
    from agents.jd_generation.graph.queries import cypher_query
    res = cypher_query(FakeClient(), "MATCH (n) CREATE (n)-[:R]->(m) RETURN n")
    assert res["ok"] is False
    assert "write" in res["reason"].lower()


@pytest.mark.parametrize("verb", ["CREATE", "MERGE", "DELETE", "SET", "REMOVE", "DROP"])
def test_cypher_query_rejects_each_write_verb(verb: str) -> None:
    from agents.jd_generation.graph.queries import cypher_query
    res = cypher_query(FakeClient(), f"MATCH (n) {verb} n.x = 1 RETURN n")
    assert res["ok"] is False, f"verb {verb} should be rejected"


def test_cypher_query_rejects_detach_delete() -> None:
    from agents.jd_generation.graph.queries import cypher_query
    res = cypher_query(FakeClient(), "MATCH (n) DETACH DELETE n")
    assert res["ok"] is False


def test_cypher_query_rejects_multiple_statements() -> None:
    from agents.jd_generation.graph.queries import cypher_query
    res = cypher_query(FakeClient(), "MATCH (n) RETURN n; MATCH (m) RETURN m;")
    assert res["ok"] is False
    assert "multiple" in res["reason"].lower()


def test_cypher_query_rejects_empty() -> None:
    from agents.jd_generation.graph.queries import cypher_query
    res = cypher_query(FakeClient(), "   ")
    assert res["ok"] is False
    assert "empty" in res["reason"].lower()


def test_cypher_query_allows_read() -> None:
    from agents.jd_generation.graph.queries import cypher_query
    rows = [{"id": "e1"}, {"id": "e2"}]
    res = cypher_query(FakeClient(read_returns=rows), "MATCH (e:Employee) RETURN e.id AS id")
    assert res["ok"] is True
    assert res["rows"] == rows
    assert res["row_count"] == 2


def test_cypher_query_tolerates_comments() -> None:
    """// CREATE in a comment shouldn't trigger the write guard."""
    from agents.jd_generation.graph.queries import cypher_query
    res = cypher_query(
        FakeClient(read_returns=[]),
        "MATCH (n) // here we used to CREATE before refactoring\nRETURN n",
    )
    assert res["ok"] is True


def test_cypher_query_surfaces_db_errors() -> None:
    from agents.jd_generation.graph.queries import cypher_query

    class BoomClient(FakeClient):
        def read(self, *a, **k):  # noqa: ANN001
            raise RuntimeError("syntax error in query")

    res = cypher_query(BoomClient(), "MATCH (n) RETURN n")
    assert res["ok"] is False
    assert "syntax error" in res["reason"]


# ===========================================================================
# Tool descriptors — OpenAI/Groq shape
# ===========================================================================
def test_retrieval_tools_complete() -> None:
    from agents.jd_generation.tools.retrieval import RETRIEVAL_TOOLS
    names = {t.name for t in RETRIEVAL_TOOLS}
    expected = {
        "find_role_family", "get_cohort", "get_skill_distribution",
        "get_career_paths_into", "get_education_distribution",
        "get_prior_industries", "get_team_signals", "get_past_jds",
        "cypher_query",
    }
    assert names == expected


def test_rejection_tools_exposes_get_past_rejections() -> None:
    from agents.jd_generation.tools.rejection import REJECTION_TOOLS
    names = {t.name for t in REJECTION_TOOLS}
    assert names == {"get_past_rejections"}


def test_tool_descriptors_are_openai_shaped() -> None:
    from agents.jd_generation.tools.retrieval import RETRIEVAL_TOOLS
    from agents.jd_generation.tools.rejection import REJECTION_TOOLS
    for t in RETRIEVAL_TOOLS + REJECTION_TOOLS:
        d = t.as_openai_tool()
        assert d["type"] == "function"
        assert "name" in d["function"]
        assert "description" in d["function"]
        assert "parameters" in d["function"]
        assert d["function"]["parameters"]["type"] == "object"
        assert "required" in d["function"]["parameters"]


# ===========================================================================
# dispatch_tool — routing + error handling
# ===========================================================================
def test_dispatch_unknown_tool_returns_error_result() -> None:
    from agents.jd_generation.tools.retrieval import dispatch_tool
    res = dispatch_tool("does_not_exist", {}, FakeClient())
    assert res.ok is False
    assert "unknown tool" in (res.error or "").lower()


def test_dispatch_missing_required_arg() -> None:
    from agents.jd_generation.tools.retrieval import dispatch_tool
    res = dispatch_tool("get_cohort", {}, FakeClient())
    assert res.ok is False
    assert "role_family" in (res.error or "")


def test_dispatch_known_tool_returns_query_result() -> None:
    from agents.jd_generation.tools.retrieval import dispatch_tool
    cohort_rows = [
        {"employee_id": "e1", "tenure_years": 3.4, "hire_date": "2023-01-15",
         "team_id": "team_platform", "team_name": "Platform"},
        {"employee_id": "e2", "tenure_years": 2.1, "hire_date": "2024-02-02",
         "team_id": "team_platform", "team_name": "Platform"},
    ]
    res = dispatch_tool(
        "get_cohort",
        {"role_family": "backend", "level": "senior", "k": 20},
        FakeClient(read_returns=cohort_rows),
    )
    assert res.ok is True
    assert res.data["role_family"] == "backend"
    assert res.data["level"] == "senior"
    assert res.data["size"] == 2
    assert res.data["employee_ids"] == ["e1", "e2"]


def test_dispatch_extra_tools_param() -> None:
    """Rejection tools are passed in via the `extra` kwarg of dispatch_tool."""
    from agents.jd_generation.tools.rejection import REJECTION_TOOLS
    from agents.jd_generation.tools.retrieval import dispatch_tool
    extra = {t.name: t for t in REJECTION_TOOLS}
    res = dispatch_tool(
        "get_past_rejections",
        {"role_family": "backend", "k": 10},
        FakeClient(read_returns=[]),
        extra=extra,
    )
    assert res.ok is True
    assert res.data["role_family"] == "backend"


# ===========================================================================
# Skill distribution bucketing
# ===========================================================================
def test_skill_distribution_buckets_correctly() -> None:
    """8 employees; skill seen by 7/8 = 87.5% universal; 4/8 = 50% distinctive;
    2/8 = 25% long_tail."""
    from agents.jd_generation.graph.queries import get_skill_distribution
    ids = [f"e{i}" for i in range(8)]
    rows = [
        {"skill_id": "sk_universal", "name": "Python", "category": "language_python",
         "holders": 7, "avg_prof": 4.2},
        {"skill_id": "sk_distinct",  "name": "Kafka",  "category": "backend_messaging",
         "holders": 4, "avg_prof": 3.5},
        {"skill_id": "sk_long",      "name": "Rust",   "category": "language_systems",
         "holders": 2, "avg_prof": 3.0},
    ]
    result = get_skill_distribution(FakeClient(read_returns=rows), ids)
    assert result["cohort_size"] == 8
    assert [s["skill_id"] for s in result["universal"]] == ["sk_universal"]
    assert [s["skill_id"] for s in result["distinctive"]] == ["sk_distinct"]
    assert [s["skill_id"] for s in result["long_tail"]] == ["sk_long"]
    assert result["universal"][0]["frequency"] == 0.875


def test_skill_distribution_handles_empty_cohort() -> None:
    from agents.jd_generation.graph.queries import get_skill_distribution
    result = get_skill_distribution(FakeClient(), [])
    assert result == {"cohort_size": 0, "universal": [], "distinctive": [], "long_tail": []}


# ===========================================================================
# Rejection tool — sanitation + classifier hook
# ===========================================================================
def test_store_rejection_raises_on_empty_text() -> None:
    from agents.jd_generation.tools.rejection import store_rejection
    from agents.jd_generation.tools.retrieval import ToolError
    with pytest.raises(ToolError):
        store_rejection(FakeClient(write_returns=[]), jd_id="jd1", reason_text="   ")


def test_store_rejection_defaults_categories_to_other_when_no_classifier() -> None:
    from agents.jd_generation.tools.rejection import store_rejection
    client = FakeClient(write_returns=[{"jd_id": "jd1", "reason_id": "rej_x"}])
    store_rejection(client, jd_id="jd1", reason_text="too vague")
    assert client.write_calls
    p = client.write_calls[0]["p"]
    assert p["categories"] == ["other"]


def test_store_rejection_invokes_classifier_when_categories_none() -> None:
    from agents.jd_generation.tools.rejection import store_rejection
    seen = {}

    def classifier(text: str) -> list[str]:
        seen["text"] = text
        return ["tone", "made_up_category"]

    client = FakeClient(write_returns=[{"jd_id": "jd1", "reason_id": "rej_x"}])
    store_rejection(client, jd_id="jd1", reason_text="jargon-heavy", classifier=classifier)
    assert seen["text"] == "jargon-heavy"
    p = client.write_calls[0]["p"]
    # Unknown category gets filtered out by the taxonomy guard
    assert p["categories"] == ["tone"]


def test_store_rejection_sanitizes_unknown_categories() -> None:
    from agents.jd_generation.tools.rejection import store_rejection
    client = FakeClient(write_returns=[{"jd_id": "jd1", "reason_id": "rej_x"}])
    store_rejection(client, jd_id="jd1", reason_text="r", categories=["banana", "tone"])
    assert client.write_calls[0]["p"]["categories"] == ["tone"]


def test_store_rejection_falls_back_to_other_when_all_unknown() -> None:
    from agents.jd_generation.tools.rejection import store_rejection
    client = FakeClient(write_returns=[{"jd_id": "jd1", "reason_id": "rej_x"}])
    store_rejection(client, jd_id="jd1", reason_text="r", categories=["foo", "bar"])
    assert client.write_calls[0]["p"]["categories"] == ["other"]


def test_rejection_categories_match_spec() -> None:
    """Section 11 of the design doc."""
    from agents.jd_generation.tools.rejection import REJECTION_CATEGORIES
    assert set(REJECTION_CATEGORIES) == {
        "tone", "requirements", "bias", "culture-fit",
        "accuracy", "structure", "other",
    }


# ===========================================================================
# Integration — exercise every retrieval tool against a live, seeded Neo4j
# ===========================================================================
@pytest.fixture
def seeded_graph(requires_neo4j) -> None:  # noqa: ARG001
    """Wipe + seed the live Neo4j once per session for integration tests."""
    from agents.jd_generation.graph.client import close_client
    from agents.jd_generation.graph.seed import seed
    try:
        seed(clear=True, quiet=True)
        yield
    finally:
        close_client()


@pytest.mark.integration
def test_find_role_family_resolves_backend(seeded_graph) -> None:  # noqa: ARG001
    from agents.jd_generation.graph.client import get_client
    from agents.jd_generation.graph.queries import find_role_family

    r = find_role_family(get_client(), role_title="Senior Backend Engineer", level="senior")
    assert r["matched"] is not None
    assert r["matched"]["role_family"] == "backend"
    assert r["matched"]["level"] == "senior"


@pytest.mark.integration
def test_get_cohort_returns_seeded_count(seeded_graph) -> None:  # noqa: ARG001
    from agents.jd_generation.graph.client import get_client
    from agents.jd_generation.graph.queries import get_cohort

    r = get_cohort(get_client(), role_family="backend", level="senior", k=50)
    assert r["size"] >= 3
    assert all(e.startswith("emp_backend_") for e in r["employee_ids"])


@pytest.mark.integration
def test_skill_distribution_on_real_cohort(seeded_graph) -> None:  # noqa: ARG001
    from agents.jd_generation.graph.client import get_client
    from agents.jd_generation.graph.queries import get_cohort, get_skill_distribution

    cohort = get_cohort(get_client(), role_family="backend", level="senior", k=50)
    dist = get_skill_distribution(get_client(), cohort["employee_ids"])
    assert dist["cohort_size"] == cohort["size"]
    assert isinstance(dist["universal"], list)
    assert isinstance(dist["distinctive"], list)


@pytest.mark.integration
def test_get_past_rejections_returns_three(seeded_graph) -> None:  # noqa: ARG001
    from agents.jd_generation.graph.client import get_client
    from agents.jd_generation.graph.queries import get_past_rejections

    backend = get_past_rejections(get_client(), role_family="backend")
    frontend = get_past_rejections(get_client(), role_family="frontend")
    ds = get_past_rejections(get_client(), role_family="data_science")
    assert backend["count"] >= 1
    assert frontend["count"] >= 1
    assert ds["count"] >= 1
    assert "jargon" in backend["reasons"][0]["reason"].lower()


@pytest.mark.integration
def test_get_past_jds_only_returns_good_hires_by_default(seeded_graph) -> None:  # noqa: ARG001
    from agents.jd_generation.graph.client import get_client
    from agents.jd_generation.graph.queries import get_past_jds

    good = get_past_jds(get_client(), role_family="backend", outcome="good_hire", k=5)
    for jd in good["jds"]:
        assert jd["hire_outcome"] == "good_hire"
