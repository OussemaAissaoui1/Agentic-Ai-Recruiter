"""Phase 3 tests — runtime (llm_client + runner + bias_check).

Unit tests:
    * llm_client._parse_chat_response: parses tool_calls / content;
      lenient on malformed JSON tool arguments.
    * runner._parse_final_json: strict / fenced / trailing-object recovery.
    * runner._build_user_intro: includes resolved role + past rejections.
    * runner._summarize and _extract_node_ids per tool.
    * bias_check: wordlist + unrealistic-requirements pattern.

Integration tests (live Groq + Neo4j):
    * Full run_generation: agent loop drives a real model, JD persists,
      tool_call_log includes get_past_rejections, bias warnings populated.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest


# ===========================================================================
# llm_client._parse_chat_response
# ===========================================================================
def _wire(content: Any, tool_calls: List[Dict[str, Any]] = None,
          finish_reason: str = "stop") -> Dict[str, Any]:
    return {
        "choices": [{
            "message": {
                "content":     content,
                "tool_calls":  tool_calls,
            },
            "finish_reason": finish_reason,
        }],
        "usage": {"total_tokens": 42},
    }


def test_parse_response_with_final_content() -> None:
    from agents.jd_generation.runtime.llm_client import _parse_chat_response
    res = _parse_chat_response(_wire("hello world"))
    assert res.content == "hello world"
    assert res.tool_calls == []
    assert res.finish_reason == "stop"
    assert res.usage == {"total_tokens": 42}


def test_parse_response_with_tool_calls() -> None:
    from agents.jd_generation.runtime.llm_client import _parse_chat_response
    res = _parse_chat_response(_wire(
        None,
        tool_calls=[{
            "id":   "call_abc",
            "type": "function",
            "function": {"name": "get_cohort",
                         "arguments": '{"role_family":"backend","level":"senior","k":20}'},
        }],
        finish_reason="tool_calls",
    ))
    assert res.content is None
    assert len(res.tool_calls) == 1
    tc = res.tool_calls[0]
    assert tc.id == "call_abc"
    assert tc.name == "get_cohort"
    assert tc.arguments == {"role_family": "backend", "level": "senior", "k": 20}


def test_parse_response_lenient_on_malformed_tool_arguments() -> None:
    """Llama models occasionally emit malformed JSON. We log + treat as
    empty args, never crash."""
    from agents.jd_generation.runtime.llm_client import _parse_chat_response
    res = _parse_chat_response(_wire(
        None,
        tool_calls=[{
            "id":   "call_zzz",
            "type": "function",
            "function": {"name": "get_cohort",
                         "arguments": "{not valid json"},
        }],
        finish_reason="tool_calls",
    ))
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0].arguments == {}
    assert res.tool_calls[0].raw_arguments == "{not valid json"


def test_parse_response_skips_non_function_tool_calls() -> None:
    from agents.jd_generation.runtime.llm_client import _parse_chat_response
    res = _parse_chat_response(_wire(
        None,
        tool_calls=[
            {"id": "call_x", "type": "code_interpreter", "function": {"name": "x", "arguments": "{}"}},
            {"id": "call_y", "type": "function",
             "function": {"name": "get_cohort", "arguments": "{}"}},
        ],
        finish_reason="tool_calls",
    ))
    assert [tc.name for tc in res.tool_calls] == ["get_cohort"]


def test_parse_response_raises_on_unexpected_shape() -> None:
    from agents.jd_generation.runtime.llm_client import GroqError, _parse_chat_response
    with pytest.raises(GroqError):
        _parse_chat_response({"choices": []})


def test_parse_response_treats_non_dict_arguments_as_empty() -> None:
    """If the model returns a JSON array as `arguments`, we don't pass it
    on as if it were a kwargs dict."""
    from agents.jd_generation.runtime.llm_client import _parse_chat_response
    res = _parse_chat_response(_wire(
        None,
        tool_calls=[{
            "id":   "c1",
            "type": "function",
            "function": {"name": "x", "arguments": "[1,2,3]"},
        }],
        finish_reason="tool_calls",
    ))
    assert res.tool_calls[0].arguments == {}


# ===========================================================================
# runner._parse_final_json
# ===========================================================================
def test_parse_final_json_strict() -> None:
    from agents.jd_generation.runtime.runner import _parse_final_json
    obj = {"jd_text": "...", "must_have": [], "nice_to_have": [],
           "evidence": {}, "rejection_checks": []}
    assert _parse_final_json(json.dumps(obj)) == obj


def test_parse_final_json_fenced() -> None:
    from agents.jd_generation.runtime.runner import _parse_final_json
    payload = '{"jd_text":"hello","must_have":[],"nice_to_have":[],"evidence":{},"rejection_checks":[]}'
    text = f"Sure! Here's the JSON:\n```json\n{payload}\n```\nLet me know if you need changes."
    assert _parse_final_json(text)["jd_text"] == "hello"


def test_parse_final_json_trailing_object() -> None:
    from agents.jd_generation.runtime.runner import _parse_final_json
    text = ('Some preamble text the model decided to add.\n'
            '{"jd_text":"x","must_have":[],"nice_to_have":[],"evidence":{},"rejection_checks":[]}')
    assert _parse_final_json(text)["jd_text"] == "x"


def test_parse_final_json_raises_on_garbage() -> None:
    from agents.jd_generation.runtime.runner import RunnerError, _parse_final_json
    with pytest.raises(RunnerError):
        _parse_final_json("definitely not json here")


def test_parse_final_json_raises_on_empty() -> None:
    from agents.jd_generation.runtime.runner import RunnerError, _parse_final_json
    with pytest.raises(RunnerError):
        _parse_final_json("   ")


# ===========================================================================
# runner._build_user_intro
# ===========================================================================
def test_user_intro_includes_role_and_rejections() -> None:
    from agents.jd_generation.runtime.runner import _build_user_intro
    find_result = {
        "matched": {"id": "role_backend_senior", "title": "Senior Backend Engineer",
                    "level": "senior", "role_family": "backend", "department": "engineering"},
        "alternatives": [],
    }
    rejections = {
        "role_family": "backend",
        "count": 1,
        "reasons": [{
            "reason":     "Too much jargon.",
            "categories": ["tone"],
            "jd_snippet": "We need a 10x ninja...",
        }],
    }
    intro = _build_user_intro(role_title="Senior Backend Engineer", level="senior",
                              team_id="team_platform", find_result=find_result,
                              rejections=rejections)
    assert "backend" in intro
    assert "team_platform" in intro
    assert "Too much jargon." in intro
    assert "10x ninja" in intro
    assert "MUST address" in intro


def test_user_intro_no_rejections_branch() -> None:
    from agents.jd_generation.runtime.runner import _build_user_intro
    find_result = {
        "matched": {"id": "role_design_junior", "title": "Junior Product Designer",
                    "level": "junior", "role_family": "design", "department": "design"},
        "alternatives": [],
    }
    rejections = {"role_family": "design", "count": 0, "reasons": []}
    intro = _build_user_intro(role_title="Junior Product Designer", level="junior",
                              team_id=None, find_result=find_result, rejections=rejections)
    assert "(none — this is the first JD" in intro
    assert "MUST address" not in intro  # nothing to address


# ===========================================================================
# runner._summarize / _extract_node_ids
# ===========================================================================
def test_summarize_cohort() -> None:
    from agents.jd_generation.runtime.runner import _summarize
    s = _summarize("get_cohort", {"size": 17, "employee_ids": []})
    assert "17" in s


def test_summarize_skill_distribution_counts() -> None:
    from agents.jd_generation.runtime.runner import _summarize
    s = _summarize("get_skill_distribution", {
        "universal": [1, 2, 3], "distinctive": [1, 2], "long_tail": [1],
    })
    assert "universal=3" in s
    assert "distinctive=2" in s


def test_extract_node_ids_from_cohort() -> None:
    from agents.jd_generation.runtime.runner import _extract_node_ids
    ids = _extract_node_ids("get_cohort", {"employee_ids": ["e1", "e2", "e3"]})
    assert ids == ["e1", "e2", "e3"]


def test_extract_node_ids_caps_long_lists() -> None:
    from agents.jd_generation.runtime.runner import _extract_node_ids
    ids = _extract_node_ids("get_cohort", {"employee_ids": [f"e{i}" for i in range(200)]})
    assert len(ids) == 50


def test_extract_node_ids_skill_distribution_takes_universal() -> None:
    from agents.jd_generation.runtime.runner import _extract_node_ids
    ids = _extract_node_ids("get_skill_distribution", {
        "universal":   [{"skill_id": "sk_1"}, {"skill_id": "sk_2"}],
        "distinctive": [{"skill_id": "sk_3"}],
    })
    assert ids == ["sk_1", "sk_2"]


# ===========================================================================
# bias_check
# ===========================================================================
def test_bias_catches_aggressive_dominate() -> None:
    from agents.jd_generation.runtime.bias_check import scan_bias
    text = "Looking for an aggressive engineer who can dominate the JavaScript ecosystem."
    warnings = scan_bias(text)
    terms = {w.term.lower() for w in warnings}
    assert any("aggressive" in t for t in terms)
    assert any("dominate" in t for t in terms)


def test_bias_catches_age_coded() -> None:
    from agents.jd_generation.runtime.bias_check import scan_bias
    text = "We want a young, energetic digital native."
    warnings = scan_bias(text)
    cats = {w.category for w in warnings}
    assert "age_coded" in cats


def test_bias_catches_unrealistic_requirements() -> None:
    from agents.jd_generation.runtime.bias_check import scan_bias
    text = "Required: 15+ years of PyTorch experience and 10+ years TypeScript."
    warnings = scan_bias(text)
    cats = {w.category for w in warnings}
    assert "unrealistic_requirements" in cats


def test_bias_clean_text_returns_empty() -> None:
    from agents.jd_generation.runtime.bias_check import scan_bias
    clean = ("We're hiring a backend engineer to own our event-ingestion pipeline "
             "end-to-end. You'll partner with the data team on schema evolution.")
    assert scan_bias(clean) == []


def test_bias_returns_snippets() -> None:
    from agents.jd_generation.runtime.bias_check import scan_bias
    text = "We need a competitive engineer."
    warnings = scan_bias(text)
    assert all(w.snippet for w in warnings)


# ===========================================================================
# Integration — full agent run against live Groq + Neo4j
# ===========================================================================
@pytest.fixture
def seeded_graph(requires_neo4j) -> None:  # noqa: ARG001
    from agents.jd_generation.graph.client import close_client
    from agents.jd_generation.graph.seed import seed
    try:
        seed(clear=True, quiet=True)
        yield
    finally:
        close_client()


@pytest.mark.integration
async def test_run_generation_end_to_end(seeded_graph, requires_groq) -> None:  # noqa: ARG001
    from agents.jd_generation.runtime.runner import run_generation
    result = await run_generation(
        role_title="Senior Backend Engineer",
        level="senior",
        team_id="team_platform",
    )
    assert result.jd_text.strip()
    assert result.role_family == "backend"
    assert result.level == "senior"
    assert len(result.must_have) >= 3
    tool_names = [c["tool"] for c in result.tool_call_log]
    assert "find_role_family" in tool_names
    assert "get_past_rejections" in tool_names
    # The seeded data has a 'jargon' rejection for backend — the model
    # must address it.
    assert any("jargon" in (rc.past_reason or "").lower() for rc in result.rejection_checks)
