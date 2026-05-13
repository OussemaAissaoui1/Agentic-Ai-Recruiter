"""Agent runner.

`run_generation` orchestrates a single JD draft:

    1. Deterministic preamble: find_role_family
    2. Deterministic preamble: get_past_rejections
    3. Build initial messages
    4. Agent loop until the model returns a final answer or max_tool_calls
    5. Force final JSON output
    6. Parse, bias-check, store, return

The two preambles are NOT delegated to the model — the runner always calls
them. The model is told "the runner has already done these" so it can pick
up from step 3.

The runner is async; tool handlers are sync (Neo4j driver is sync), so
each dispatch runs via `asyncio.to_thread`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from agents.jd_generation.config import JDConfig, load_jd_config
from agents.jd_generation.graph.client import Neo4jClient, get_client
from agents.jd_generation.graph.queries import create_jd, find_role_family, get_past_rejections
from agents.jd_generation.runtime.bias_check import BiasWarning, scan_bias
from agents.jd_generation.runtime.llm_client import (
    ChatResult,
    GroqClient,
    GroqError,
    ToolCall,
)
from agents.jd_generation.runtime.prompts import FINAL_JSON_INSTRUCTIONS, SYSTEM_PROMPT
from agents.jd_generation.tools import REJECTION_TOOLS, RETRIEVAL_TOOLS
from agents.jd_generation.tools.retrieval import (
    ToolDescriptor,
    ToolResult,
    all_tool_specs,
    dispatch_tool,
)

log = logging.getLogger("agents.jd_generation.runtime.runner")

# Combined tool registry (LLM-visible) — retrieval + rejection (get_past_rejections only)
_LLM_TOOLS: List[ToolDescriptor] = list(RETRIEVAL_TOOLS) + list(REJECTION_TOOLS)
_LLM_TOOL_SPECS: List[Dict[str, Any]] = all_tool_specs(extra=list(REJECTION_TOOLS))
_LLM_TOOLS_BY_NAME: Dict[str, ToolDescriptor] = {t.name: t for t in _LLM_TOOLS}

MIN_COHORT_HARD_FLOOR = 3


# ---------------------------------------------------------------------------
# Errors + result shapes
# ---------------------------------------------------------------------------
class RunnerError(RuntimeError):
    """Surfaced when generation cannot complete (cohort too small, JSON
    malformed past retries, etc.)."""


@dataclass
class RejectionCheck:
    past_reason:   str
    how_addressed: str
    categories:    List[str] = field(default_factory=list)


@dataclass
class GenerateResult:
    jd_id:            str
    role_family:      str
    level:            str
    team_id:          Optional[str]
    jd_text:          str
    must_have:        List[str]
    nice_to_have:     List[str]
    evidence:         Dict[str, Any]
    rejection_checks: List[RejectionCheck]
    tool_call_log:    List[Dict[str, Any]]
    bias_warnings:    List[Dict[str, Any]]
    created_at:       str

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "rejection_checks": [asdict(rc) for rc in self.rejection_checks],
        }


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------
async def run_generation(
    *,
    role_title:   str,
    level:        str,
    team_id:      Optional[str] = None,
    client:       Optional[Neo4jClient] = None,
    llm:          Optional[GroqClient] = None,
    cfg:          Optional[JDConfig] = None,
    event_sink:   Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
) -> GenerateResult:
    """End-to-end JD generation.

    `event_sink`, if provided, receives `{type, ...}` events as the agent
    progresses. This is how the SSE wrapper streams progress to the
    frontend; CLI / tests pass `None` and just receive the final result.
    """
    cfg = cfg or load_jd_config()
    if client is None:
        client = get_client(cfg)
    own_llm = False
    if llm is None:
        if not cfg.llm.is_configured:
            raise RunnerError("GROQ_API_KEY is not configured")
        llm = GroqClient(api_key=cfg.llm.api_key, timeout=cfg.llm.request_timeout_s)
        own_llm = True

    tool_call_log: List[Dict[str, Any]] = []

    async def _emit(event: Dict[str, Any]) -> None:
        if event_sink is not None:
            await event_sink(event)

    try:
        # ---- Preamble 1: find_role_family --------------------------------
        await _emit({"type": "tool_call", "tool": "find_role_family",
                     "args": {"role_title": role_title, "level": level}})
        rf_result = await asyncio.to_thread(
            find_role_family, client, role_title, level,
        )
        tool_call_log.append({"tool": "find_role_family", "ok": True, "data": rf_result})
        await _emit({"type": "tool_result", "tool": "find_role_family",
                     "ok": True, "summary": _summarize("find_role_family", rf_result)})

        if not rf_result.get("matched"):
            raise RunnerError(f"could not resolve role title '{role_title}' to a known family")

        matched = rf_result["matched"]
        role_family: str = matched["role_family"]
        resolved_level: str = matched["level"] or level

        # ---- Preamble 2: get_past_rejections -----------------------------
        await _emit({"type": "tool_call", "tool": "get_past_rejections",
                     "args": {"role_family": role_family, "k": cfg.agent.rejection_lookback}})
        pr_result = await asyncio.to_thread(
            get_past_rejections, client, role_family, cfg.agent.rejection_lookback,
        )
        tool_call_log.append({"tool": "get_past_rejections", "ok": True, "data": pr_result})
        await _emit({"type": "tool_result", "tool": "get_past_rejections",
                     "ok": True, "summary": _summarize("get_past_rejections", pr_result)})

        # ---- Build messages ----------------------------------------------
        messages: List[Dict[str, Any]] = [
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": _build_user_intro(
                role_title=role_title, level=resolved_level, team_id=team_id,
                find_result=rf_result, rejections=pr_result,
            )},
        ]

        # ---- Agent loop --------------------------------------------------
        max_calls = cfg.agent.max_tool_calls
        for turn in range(max_calls + 1):  # +1 guard so we never infinite-loop
            if len(tool_call_log) >= max_calls:
                # We've burned the tool-call budget; force the model to finalize.
                messages.append({"role": "user", "content": FINAL_JSON_INSTRUCTIONS})
                final = await llm.chat_with_tools(
                    model=cfg.llm.model,
                    messages=messages,
                    tools=[],
                    tool_choice="none",
                    temperature=cfg.llm.temperature_synthesis,
                    timeout=cfg.llm.request_timeout_s,
                )
                return await _finalize(
                    final=final, role_family=role_family, level=resolved_level,
                    team_id=team_id, client=client, cfg=cfg,
                    rejections=pr_result, tool_call_log=tool_call_log, emit=_emit,
                )

            try:
                resp = await llm.chat_with_tools(
                    model=cfg.llm.model,
                    messages=messages,
                    tools=_LLM_TOOL_SPECS,
                    tool_choice="auto",
                    temperature=cfg.llm.temperature_tool_select,
                    timeout=cfg.llm.request_timeout_s,
                )
            except GroqError as e:
                raise RunnerError(f"groq failed turn={turn}: {e}") from e

            if resp.tool_calls:
                # Append model's assistant message with the tool calls
                messages.append({
                    "role": "assistant",
                    "content": resp.content,
                    "tool_calls": [
                        {
                            "id":   tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": tc.raw_arguments or "{}"},
                        }
                        for tc in resp.tool_calls
                    ],
                })
                # Dispatch each, append each tool result as a `tool` message
                for tc in resp.tool_calls:
                    await _emit({"type": "tool_call", "tool": tc.name, "args": tc.arguments})
                    tool_res: ToolResult = await asyncio.to_thread(
                        dispatch_tool, tc.name, tc.arguments, client,
                        {n: t for n, t in _LLM_TOOLS_BY_NAME.items() if n not in {"find_role_family"}},
                    )
                    tool_call_log.append({
                        "tool":  tc.name,
                        "ok":    tool_res.ok,
                        "data":  tool_res.data if tool_res.ok else {"error": tool_res.error},
                    })
                    await _emit({"type": "tool_result", "tool": tc.name, "ok": tool_res.ok,
                                 "summary": _summarize(tc.name, tool_res.data)
                                            if tool_res.ok else tool_res.error,
                                 "node_ids": _extract_node_ids(tc.name, tool_res.data)})
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(
                            _compact_tool_result(tc.name, tool_res.data)
                            if tool_res.ok else {"error": tool_res.error}
                        ),
                    })
                continue

            # No tool calls — the model is ready to finalize
            messages.append({"role": "assistant", "content": resp.content or ""})
            messages.append({"role": "user",      "content": FINAL_JSON_INSTRUCTIONS})

            try:
                final = await llm.chat_with_tools(
                    model=cfg.llm.model,
                    messages=messages,
                    tools=[],
                    tool_choice="none",
                    temperature=cfg.llm.temperature_synthesis,
                    timeout=cfg.llm.request_timeout_s,
                )
            except GroqError as e:
                raise RunnerError(f"groq failed during finalize: {e}") from e
            return await _finalize(
                final=final, role_family=role_family, level=resolved_level,
                team_id=team_id, client=client, cfg=cfg,
                rejections=pr_result, tool_call_log=tool_call_log, emit=_emit,
            )

        # Defensive: exhausted the for-loop without returning
        raise RunnerError(f"agent loop exhausted after {max_calls} tool calls")

    finally:
        if own_llm and llm is not None:
            await llm.aclose()


# ---------------------------------------------------------------------------
# Finalization — parse JSON, store, return
# ---------------------------------------------------------------------------
async def _finalize(
    *,
    final:         ChatResult,
    role_family:   str,
    level:         str,
    team_id:       Optional[str],
    client:        Neo4jClient,
    cfg:           JDConfig,
    rejections:    Dict[str, Any],
    tool_call_log: List[Dict[str, Any]],
    emit:          Callable[[Dict[str, Any]], Awaitable[None]],
) -> GenerateResult:
    text = (final.content or "").strip()
    parsed = _parse_final_json(text)

    jd_text = (parsed.get("jd_text") or "").strip()
    must_have = list(parsed.get("must_have") or [])
    nice_to_have = list(parsed.get("nice_to_have") or [])
    evidence = parsed.get("evidence") or {}
    rejection_checks_raw = parsed.get("rejection_checks") or []

    if not jd_text:
        raise RunnerError("final response missing jd_text")

    rejection_checks = [
        RejectionCheck(
            past_reason=str(rc.get("past_reason", "")),
            how_addressed=str(rc.get("how_addressed", "")),
            categories=list(rc.get("categories") or []),
        )
        for rc in rejection_checks_raw
    ]

    # Backfill rejection_checks if the model omitted them and rejections exist
    if not rejection_checks and rejections.get("count", 0) > 0:
        for r in rejections["reasons"]:
            rejection_checks.append(RejectionCheck(
                past_reason=str(r.get("reason", "")),
                how_addressed="(model did not provide an explanation — please verify manually)",
                categories=list(r.get("categories") or []),
            ))

    warnings = scan_bias(jd_text)
    bias_warnings = [
        {"category": w.category, "term": w.term, "snippet": w.snippet}
        for w in warnings
    ]

    jd_id = f"jd_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    await asyncio.to_thread(
        create_jd, client, jd_id=jd_id, text=jd_text,
        role_family=role_family, created_at=created_at,
    )

    result = GenerateResult(
        jd_id=jd_id,
        role_family=role_family,
        level=level,
        team_id=team_id,
        jd_text=jd_text,
        must_have=must_have,
        nice_to_have=nice_to_have,
        evidence=evidence,
        rejection_checks=rejection_checks,
        tool_call_log=tool_call_log,
        bias_warnings=bias_warnings,
        created_at=created_at,
    )
    await emit({"type": "final", "jd_id": jd_id})
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_user_intro(
    *,
    role_title:  str,
    level:       str,
    team_id:     Optional[str],
    find_result: Dict[str, Any],
    rejections:  Dict[str, Any],
) -> str:
    """The initial user turn — packed with preamble results."""
    matched = find_result["matched"]
    parts: List[str] = [
        f"Recruiter input: role_title={role_title!r}, level={level!r}"
        f"{', team_id=' + repr(team_id) if team_id else ''}",
        "",
        "Resolved role family (from find_role_family):",
        f"  role_family: {matched['role_family']}",
        f"  level:       {matched['level']}",
        f"  canonical:   {matched['title']}",
        f"  department:  {matched['department']}",
    ]
    if find_result.get("alternatives"):
        alts = ", ".join(a["title"] for a in find_result["alternatives"][:3])
        parts.append(f"  alternatives: {alts}")

    parts.extend(["", f"Past rejections for this role family (count={rejections.get('count', 0)}):"])
    if not rejections.get("reasons"):
        parts.append("  (none — this is the first JD for this role family)")
    else:
        for i, r in enumerate(rejections["reasons"], 1):
            cats = ", ".join(r.get("categories") or [])
            parts.append(f"  {i}. [{cats}] {r.get('reason')}")
            snip = (r.get("jd_snippet") or "")[:120].replace("\n", " ")
            parts.append(f"     prior JD snippet: {snip}…")
        parts.append("")
        parts.append("You MUST address each past rejection in your final rejection_checks "
                     "array, and MUST NOT repeat any rejected phrasing in jd_text.")

    parts.extend([
        "",
        "Now proceed with the remaining steps (3-9). Start by calling get_cohort.",
    ])
    return "\n".join(parts)


_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_TRAILING_JSON_RE = re.compile(r"(\{.*\})\s*$", re.DOTALL)


def _parse_final_json(text: str) -> Dict[str, Any]:
    """Best-effort JSON extraction.

    Llama-class models occasionally wrap their JSON in ```json fences or
    add a stray line of prose before the object. We try strict parse first,
    then unwrap a code fence, then fall back to "last {...} in the string".
    """
    text = text.strip()
    if not text:
        raise RunnerError("model returned empty final response")
    # Strict
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fenced
    m = _FENCED_JSON_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Trailing object — match by counting braces (DOTALL greedy)
    m = _TRAILING_JSON_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    raise RunnerError(f"could not parse model output as JSON: {text[:300]}")


def _summarize(tool: str, data: Dict[str, Any]) -> str:
    """Tiny human-readable line for the event stream / log."""
    if not isinstance(data, dict):
        return str(data)[:120]
    if tool == "find_role_family":
        m = data.get("matched") or {}
        return f"matched {m.get('title')} ({m.get('role_family')}, {m.get('level')})"
    if tool == "get_past_rejections":
        return f"{data.get('count', 0)} past rejections"
    if tool == "get_cohort":
        return f"{data.get('size', 0)} employees in cohort"
    if tool == "get_skill_distribution":
        return (f"universal={len(data.get('universal', []))} "
                f"distinctive={len(data.get('distinctive', []))} "
                f"long_tail={len(data.get('long_tail', []))}")
    if tool == "get_past_jds":
        return f"{len(data.get('jds', []))} JDs"
    if tool == "cypher_query":
        return f"{data.get('row_count', 0)} rows"
    return str(data)[:120]


def _compact_tool_result(tool: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Trim a tool result before re-sending it in the model's context.

    Groq's free-tier TPM is tight; full skill_distribution payloads (294 skills
    × 4 fields) can blow the budget after one or two turns. We keep the
    salient signal (names, frequencies, top entries) and drop secondary fields
    plus over-long tails.

    The TRUE result still gets logged into tool_call_log for the evidence
    panel — only the message-history copy is trimmed.
    """
    if not isinstance(data, dict):
        return data
    if tool == "get_skill_distribution":
        def _slim(rows: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
            return [{"name": r.get("name"), "frequency": r.get("frequency")}
                    for r in rows[:k]]
        return {
            "cohort_size": data.get("cohort_size"),
            "universal":   _slim(data.get("universal", []),   10),
            "distinctive": _slim(data.get("distinctive", []), 12),
            "long_tail":   _slim(data.get("long_tail", []),    8),
        }
    if tool == "get_cohort":
        return {
            "role_family": data.get("role_family"),
            "level":       data.get("level"),
            "size":        data.get("size"),
            "employee_ids": data.get("employee_ids", [])[:20],
        }
    if tool == "get_career_paths_into":
        return {
            "role_family": data.get("role_family"),
            "paths":       data.get("paths", [])[:5],
            "immediate_predecessors": data.get("immediate_predecessors", [])[:6],
        }
    if tool == "get_education_distribution":
        return {
            "cohort_size":     data.get("cohort_size"),
            "by_field":        data.get("by_field", [])[:6],
            "by_degree":       data.get("by_degree", [])[:6],
            "self_taught_pct": data.get("self_taught_pct"),
        }
    if tool == "get_prior_industries":
        return {
            "cohort_size":    data.get("cohort_size"),
            "by_industry":    data.get("by_industry", [])[:6],
            "by_size_bucket": data.get("by_size_bucket", []),
        }
    if tool == "get_past_jds":
        return {
            "role_family": data.get("role_family"),
            "outcome":     data.get("outcome"),
            "jds": [
                {"id": j.get("id"), "text": (j.get("text") or "")[:500]}
                for j in data.get("jds", [])[:3]
            ],
        }
    if tool == "get_past_rejections":
        # Past-rejection reasons are load-bearing — keep them, just trim the
        # JD snippet that contextualizes each.
        return {
            "role_family": data.get("role_family"),
            "count":       data.get("count"),
            "reasons": [
                {
                    "reason":     r.get("reason"),
                    "categories": r.get("categories"),
                    "jd_snippet": (r.get("jd_snippet") or "")[:220],
                }
                for r in data.get("reasons", [])
            ],
        }
    if tool == "get_team_signals":
        return {
            "team_id":   data.get("team_id"),
            "name":      data.get("name"),
            "size":      data.get("size"),
            "by_level":  data.get("by_level"),
            "by_family": data.get("by_family"),
        }
    if tool == "cypher_query":
        rows = data.get("rows", [])[:20]
        return {"ok": data.get("ok"), "row_count": data.get("row_count"), "rows": rows}
    return data


def _extract_node_ids(tool: str, data: Dict[str, Any]) -> List[str]:
    """Pull node ids out of a tool result so the 3D viz can highlight them."""
    if not isinstance(data, dict):
        return []
    if tool == "get_cohort":
        return list(data.get("employee_ids", []))[:50]
    if tool == "get_skill_distribution":
        return [s["skill_id"] for s in data.get("universal", [])][:30]
    if tool == "get_team_signals":
        return [data.get("team_id")] if data.get("team_id") else []
    if tool == "get_past_jds":
        return [j["id"] for j in data.get("jds", [])]
    if tool == "get_past_rejections":
        return [r["reason_id"] for r in data.get("reasons", []) if r.get("reason_id")]
    return []
