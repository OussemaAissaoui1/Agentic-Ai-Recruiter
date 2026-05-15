"""Retrieval tools — the 8 graph tools plus a read-only Cypher escape hatch.

Each tool is a plain function (sync) that takes a `Neo4jClient` plus
arguments and returns a dict. The `RETRIEVAL_TOOLS` list bundles the
function + an OpenAI-style JSON-schema descriptor for the LLM.

`dispatch_tool` resolves a tool name and arguments — used by the agent
runner so the runner doesn't need to switch over every tool.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from agents.jd_generation.graph.client import Neo4jClient
from agents.jd_generation.graph.queries import (
    cypher_query,
    find_role_family,
    get_career_paths_into,
    get_cohort,
    get_education_distribution,
    get_past_jds,
    get_prior_industries,
    get_skill_distribution,
    get_team_signals,
)

log = logging.getLogger("agents.jd_generation.tools")


# ---------------------------------------------------------------------------
# Tool result + error envelopes
# ---------------------------------------------------------------------------
@dataclass
class ToolResult:
    """What every tool returns to the agent runner."""
    tool:    str
    ok:      bool
    data:    Dict[str, Any]
    error:   Optional[str] = None

    def to_event(self) -> Dict[str, Any]:
        """Shape suitable for an SSE `tool_result` event."""
        return {
            "type": "tool_result",
            "tool": self.tool,
            "ok":   self.ok,
            "data": self.data,
            "error": self.error,
        }


class ToolError(RuntimeError):
    """Raised when a tool's arguments are malformed."""


# ---------------------------------------------------------------------------
# Tool descriptor (OpenAI tools-API shape — Groq accepts the same)
# ---------------------------------------------------------------------------
@dataclass
class ToolDescriptor:
    name:        str
    description: str
    parameters:  Dict[str, Any]
    handler:     Callable[..., Dict[str, Any]]

    def as_openai_tool(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name":        self.name,
                "description": self.description,
                "parameters":  self.parameters,
            },
        }


# ---------------------------------------------------------------------------
# Argument coercion
# ---------------------------------------------------------------------------
def _coerce_str(args: Dict[str, Any], key: str, default: Optional[str] = None) -> str:
    val = args.get(key, default)
    if val is None:
        raise ToolError(f"missing argument: {key}")
    return str(val).strip()


def _coerce_int(args: Dict[str, Any], key: str, default: int) -> int:
    val = args.get(key, default)
    try:
        return int(val) if val is not None else default
    except (TypeError, ValueError) as exc:
        raise ToolError(f"argument {key} must be int") from exc


def _coerce_str_list(args: Dict[str, Any], key: str) -> List[str]:
    val = args.get(key)
    if val is None:
        raise ToolError(f"missing argument: {key}")
    if not isinstance(val, list):
        raise ToolError(f"argument {key} must be a list of strings")
    return [str(x) for x in val]


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------
def _h_find_role_family(client: Neo4jClient, args: Dict[str, Any]) -> Dict[str, Any]:
    return find_role_family(client, _coerce_str(args, "role_title"),
                            level=args.get("level"))


def _h_get_cohort(client: Neo4jClient, args: Dict[str, Any]) -> Dict[str, Any]:
    return get_cohort(
        client,
        role_family=_coerce_str(args, "role_family"),
        level=_coerce_str(args, "level"),
        k=_coerce_int(args, "k", 20),
    )


def _h_get_skill_distribution(client: Neo4jClient, args: Dict[str, Any]) -> Dict[str, Any]:
    return get_skill_distribution(client, _coerce_str_list(args, "employee_ids"))


def _h_get_career_paths_into(client: Neo4jClient, args: Dict[str, Any]) -> Dict[str, Any]:
    return get_career_paths_into(
        client,
        role_family=_coerce_str(args, "role_family"),
        depth=_coerce_int(args, "depth", 3),
    )


def _h_get_education_distribution(client: Neo4jClient, args: Dict[str, Any]) -> Dict[str, Any]:
    return get_education_distribution(client, _coerce_str_list(args, "employee_ids"))


def _h_get_prior_industries(client: Neo4jClient, args: Dict[str, Any]) -> Dict[str, Any]:
    return get_prior_industries(client, _coerce_str_list(args, "employee_ids"))


def _h_get_team_signals(client: Neo4jClient, args: Dict[str, Any]) -> Dict[str, Any]:
    return get_team_signals(client, _coerce_str(args, "team_id"))


def _h_get_past_jds(client: Neo4jClient, args: Dict[str, Any]) -> Dict[str, Any]:
    return get_past_jds(
        client,
        role_family=_coerce_str(args, "role_family"),
        outcome=_coerce_str(args, "outcome", "good_hire"),
        k=_coerce_int(args, "k", 3),
    )


def _h_cypher_query(client: Neo4jClient, args: Dict[str, Any]) -> Dict[str, Any]:
    return cypher_query(client, _coerce_str(args, "query"))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
RETRIEVAL_TOOLS: List[ToolDescriptor] = [
    ToolDescriptor(
        name="find_role_family",
        description=(
            "Resolve a free-text role title (with optional level) to the company's canonical "
            "Role node. Call this first. Returns the matched role plus alternatives."
        ),
        parameters={
            "type": "object",
            "properties": {
                "role_title": {"type": "string", "description": "Free-text role title from the recruiter input."},
                "level":      {"type": "string", "description": "Optional level hint: junior, mid, senior.",
                               "enum": ["junior", "mid", "senior"]},
            },
            "required": ["role_title"],
        },
        handler=_h_find_role_family,
    ),
    ToolDescriptor(
        name="get_cohort",
        description=(
            "Return the active employees currently holding a role in the given family + level. "
            "Use to bootstrap analysis. If cohort < 5, broaden the role family before drafting."
        ),
        parameters={
            "type": "object",
            "properties": {
                "role_family": {"type": "string"},
                "level":       {"type": "string", "enum": ["junior", "mid", "senior"]},
                "k":           {"type": "integer", "default": 20, "minimum": 1, "maximum": 100},
            },
            "required": ["role_family", "level"],
        },
        handler=_h_get_cohort,
    ),
    ToolDescriptor(
        name="get_skill_distribution",
        description=(
            "For a cohort of employees, return skills bucketed by frequency: "
            "universal (>80%), distinctive (40-80%), long_tail (<40%). "
            "Use to identify must-have vs nice-to-have skills."
        ),
        parameters={
            "type": "object",
            "properties": {
                "employee_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["employee_ids"],
        },
        handler=_h_get_skill_distribution,
    ),
    ToolDescriptor(
        name="get_career_paths_into",
        description=(
            "Show the most-common career paths leading INTO this role family. "
            "Useful for the 'who thrives in this role' part of the JD."
        ),
        parameters={
            "type": "object",
            "properties": {
                "role_family": {"type": "string"},
                "depth":       {"type": "integer", "default": 3, "minimum": 1, "maximum": 5},
            },
            "required": ["role_family"],
        },
        handler=_h_get_career_paths_into,
    ),
    ToolDescriptor(
        name="get_education_distribution",
        description=(
            "Distribution of degrees, fields, and institution tiers across a cohort. "
            "Use to phrase education expectations honestly (e.g. 'self-taught welcome')."
        ),
        parameters={
            "type": "object",
            "properties": {
                "employee_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["employee_ids"],
        },
        handler=_h_get_education_distribution,
    ),
    ToolDescriptor(
        name="get_prior_industries",
        description=(
            "What industries did current employees in this cohort come from? "
            "Use to surface non-obvious background sources the recruiter should signal."
        ),
        parameters={
            "type": "object",
            "properties": {
                "employee_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["employee_ids"],
        },
        handler=_h_get_prior_industries,
    ),
    ToolDescriptor(
        name="get_team_signals",
        description=(
            "Team-level structural signals: size, department, level distribution, role-family mix. "
            "Only call if the recruiter provided a team_id."
        ),
        parameters={
            "type": "object",
            "properties": {"team_id": {"type": "string"}},
            "required": ["team_id"],
        },
        handler=_h_get_team_signals,
    ),
    ToolDescriptor(
        name="get_past_jds",
        description=(
            "Past JD texts for this role family — by default only those that led to "
            "good hires. Use to capture the company's voice."
        ),
        parameters={
            "type": "object",
            "properties": {
                "role_family": {"type": "string"},
                "outcome":     {"type": "string", "enum": ["good_hire", "bad_hire", "not_filled", "any"],
                                "default": "good_hire"},
                "k":           {"type": "integer", "default": 3, "minimum": 1, "maximum": 10},
            },
            "required": ["role_family"],
        },
        handler=_h_get_past_jds,
    ),
    ToolDescriptor(
        name="cypher_query",
        description=(
            "READ-ONLY Cypher escape hatch for ad-hoc questions about the graph. "
            "Write keywords (CREATE/MERGE/DELETE/SET/REMOVE/DROP) are rejected."
        ),
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        handler=_h_cypher_query,
    ),
]


# ---------------------------------------------------------------------------
# Dispatcher — used by the runner
# ---------------------------------------------------------------------------
_BY_NAME = {t.name: t for t in RETRIEVAL_TOOLS}


def dispatch_tool(
    name: str, args: Dict[str, Any], client: Neo4jClient,
    extra: Optional[Dict[str, ToolDescriptor]] = None,
) -> ToolResult:
    """Resolve a tool by name and invoke it. Wraps argument + execution errors."""
    pool = dict(_BY_NAME)
    if extra:
        pool.update(extra)
    tool = pool.get(name)
    if tool is None:
        return ToolResult(tool=name, ok=False, data={}, error=f"unknown tool: {name}")
    try:
        data = tool.handler(client, args or {})
    except ToolError as e:
        return ToolResult(tool=name, ok=False, data={}, error=str(e))
    except Exception as e:  # noqa: BLE001 — surface to the agent loop, not the user
        log.exception("tool %s crashed", name)
        return ToolResult(tool=name, ok=False, data={}, error=f"{type(e).__name__}: {e}")
    return ToolResult(tool=name, ok=True, data=data)


def all_tool_specs(extra: Optional[List[ToolDescriptor]] = None) -> List[Dict[str, Any]]:
    """Return the OpenAI-shaped tool list. Includes rejection tools if passed in."""
    tools = list(RETRIEVAL_TOOLS)
    if extra:
        tools.extend(extra)
    return [t.as_openai_tool() for t in tools]
