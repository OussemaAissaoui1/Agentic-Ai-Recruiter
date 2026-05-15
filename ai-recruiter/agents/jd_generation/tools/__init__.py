"""Agent-facing tools — thin wrappers over `graph/queries.py`.

`retrieval.py` exposes the 8 graph tools plus the read-only `cypher_query`
escape hatch. `rejection.py` exposes the 2 rejection tools. Each tool comes
with a JSON-schema descriptor that the LLM client uses to advertise it.
"""

from __future__ import annotations

from .rejection import REJECTION_TOOLS, get_past_rejections, store_rejection
from .retrieval import RETRIEVAL_TOOLS, ToolError, ToolResult, dispatch_tool

__all__ = [
    "REJECTION_TOOLS",
    "RETRIEVAL_TOOLS",
    "ToolError",
    "ToolResult",
    "dispatch_tool",
    "get_past_rejections",
    "store_rejection",
]
