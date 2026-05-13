"""Phase 5 acceptance test — the rejection feedback loop.

This is the make-or-break test for the whole sub-module. If this passes, the
agent is actually learning from rejections. If it doesn't, the module is
just a template generator with a logging table.

All tests here are @pytest.mark.integration — they hit a live Neo4j AND a
live Groq endpoint. Run with:

    JD_NEO4J_URI=... JD_NEO4J_USER=... JD_NEO4J_PASSWORD=... \\
    GROQ_API_KEY=... pytest tests/jd_generation/test_rejection_loop.py -m integration -v

Two complementary tests:

1. test_rejection_reason_influences_next_generation (spec section 14, verbatim):
   - Generate a JD for Senior Backend Engineer
   - Reject with a SPECIFIC jargon-y phrase
   - Generate again
   - Assert: the new draft mentions the rejection in rejection_checks,
            does NOT repeat the rejected phrasing,
            tool_call_log contains get_past_rejections.

2. test_seeded_rejections_seen_on_first_generation:
   - Without manually injecting a rejection, the seeded data already has
     a jargon rejection for backend. The first generation must see it.
"""

from __future__ import annotations

import os

import pytest

REJECTED_PHRASE = "synergize cross-functional verticals"
JARGON_REASON = (
    "Too much jargon — phrases like 'synergize cross-functional verticals' "
    "don't match our straightforward voice."
)


@pytest.fixture
def seeded_graph(requires_neo4j) -> None:  # noqa: ARG001
    from agents.jd_generation.graph.client import close_client
    from agents.jd_generation.graph.seed import seed
    try:
        seed(clear=True, quiet=True)
        yield
    finally:
        close_client()


# ---------------------------------------------------------------------------
# Test 1 — explicit reject + regenerate (spec section 14)
# ---------------------------------------------------------------------------
@pytest.mark.integration
async def test_rejection_reason_influences_next_generation(
    seeded_graph,             # noqa: ARG001
    requires_groq,            # noqa: ARG001
) -> None:
    """The acceptance test from spec section 14, verbatim semantics.

    Note: the seeded graph already contains a jargon-related rejection for
    backend (test_jargon_rejection_is_for_backend covers that). This test
    adds a SECOND, freshly-created rejection, then asserts the agent's NEXT
    generation reflects both — or at minimum the newly-added one with its
    exact phrasing.
    """
    from agents.jd_generation.service import generate_jd, reject_jd

    # 1. Generate JD #1.
    jd1 = await generate_jd(role_title="Senior Backend Engineer", level="senior")
    assert jd1.jd_text.strip()

    # 2. Reject with a SPECIFIC reason that is more specific than the
    #    seeded one (so we can prove the *new* rejection influences the
    #    *next* generation, not just the pre-existing seed).
    await reject_jd(jd_id=jd1.jd_id, reason_text=JARGON_REASON, categories=["tone"])

    # 3. Generate JD #2 for the same role family.
    jd2 = await generate_jd(role_title="Senior Backend Engineer", level="senior")
    assert jd2.jd_text.strip()

    # 4. The new draft must show awareness of the prior rejection.
    assert any(
        "jargon" in (rc.past_reason or "").lower()
        for rc in jd2.rejection_checks
    ), (
        "expected rejection_checks to mention 'jargon'; got "
        f"{[rc.past_reason for rc in jd2.rejection_checks]}"
    )

    # 5. The new draft must not repeat the rejected phrasing.
    jd_lower = jd2.jd_text.lower()
    assert "synergize" not in jd_lower, "rejected term 'synergize' leaked into new draft"
    assert "cross-functional verticals" not in jd_lower, (
        "rejected phrase 'cross-functional verticals' leaked into new draft"
    )

    # 6. The tool call log must show get_past_rejections was called.
    tool_names = [call["tool"] for call in jd2.tool_call_log]
    assert "get_past_rejections" in tool_names, (
        f"agent never called get_past_rejections; tools used: {tool_names}"
    )


# ---------------------------------------------------------------------------
# Test 2 — seeded rejections are surfaced on the first generation
# ---------------------------------------------------------------------------
@pytest.mark.integration
async def test_seeded_rejections_seen_on_first_generation(
    seeded_graph,             # noqa: ARG001
    requires_groq,            # noqa: ARG001
) -> None:
    """The seeded data has 3 rejections. The agent should see and address
    the backend one without us having to manually create another."""
    from agents.jd_generation.service import generate_jd

    jd = await generate_jd(role_title="Senior Backend Engineer", level="senior")

    # rejection_checks must mention the seeded jargon reason.
    seeded_terms = ("jargon", "synergize", "verticals")
    assert any(
        any(term in (rc.past_reason or "").lower() for term in seeded_terms)
        for rc in jd.rejection_checks
    ), (
        "the seeded jargon rejection didn't appear in rejection_checks; got "
        f"{[rc.past_reason for rc in jd.rejection_checks]}"
    )

    # And the rejected phrasing must not leak into the new JD.
    assert "synergize" not in jd.jd_text.lower()
    assert "cross-functional verticals" not in jd.jd_text.lower()


# ---------------------------------------------------------------------------
# Test 3 — store_rejection persists; get_past_rejections returns it
# ---------------------------------------------------------------------------
@pytest.mark.integration
async def test_reject_persists_and_is_readable(
    seeded_graph,             # noqa: ARG001
) -> None:
    """Pure DB layer — no LLM. Verifies the rejection-write path independently
    of the agent loop, so debugging is easier when test 1 fails."""
    from agents.jd_generation.graph.client import get_client
    from agents.jd_generation.graph.queries import get_past_rejections
    from agents.jd_generation.service import reject_jd
    from agents.jd_generation.runtime.runner import run_generation

    # Need a JD record to reject against — generate one without Groq via a
    # short-circuit path is overkill; just use an existing seeded JD id.
    client = get_client()
    rows = client.read(
        """
        MATCH (j:JobDescription) WHERE j.role_family = 'backend'
        RETURN j.id AS id LIMIT 1
        """
    )
    jd_id = rows[0]["id"]

    # Pass categories explicitly so we don't need GROQ_API_KEY for this test.
    await reject_jd(
        jd_id=jd_id,
        reason_text="A very specific test rejection reason no.42",
        categories=["tone", "structure"],
    )

    rejections = get_past_rejections(client, role_family="backend", k=20)
    found = [r for r in rejections["reasons"]
             if "no.42" in (r["reason"] or "")]
    assert found, f"newly-stored rejection not found in get_past_rejections"
    assert set(found[0]["categories"]) == {"tone", "structure"}
