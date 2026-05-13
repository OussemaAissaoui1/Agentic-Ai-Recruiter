"""Rejection-loop tools — the proof-of-learning surface.

`get_past_rejections` is called DETERMINISTICALLY by the agent runner before
the model takes over (see runtime/runner.py). It's also registered as an
LLM tool so the model can re-fetch with different k if it wants more context.

`store_rejection` is called from the `reject_jd` service function; the LLM
never invokes it directly.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from agents.jd_generation.graph.client import Neo4jClient
from agents.jd_generation.graph.queries import (
    get_past_rejections as _q_get_past_rejections,
    store_rejection as _q_store_rejection,
)
from agents.jd_generation.tools.retrieval import ToolDescriptor, ToolError

log = logging.getLogger("agents.jd_generation.tools.rejection")

# Authoritative list — kept in sync with the spec's section 11 taxonomy.
REJECTION_CATEGORIES = [
    "tone", "requirements", "bias", "culture-fit", "accuracy", "structure", "other",
]


def get_past_rejections(
    client: Neo4jClient, role_family: str, k: int = 10
) -> Dict[str, Any]:
    """Fetch the k most-recent rejections for a role family."""
    return _q_get_past_rejections(client, role_family=role_family, k=k)


def store_rejection(
    client: Neo4jClient,
    jd_id: str,
    reason_text: str,
    categories: Optional[List[str]] = None,
    classifier: Optional[Any] = None,
) -> Dict[str, Any]:
    """Persist a rejection. If `categories` is None and a classifier is
    provided, classify the free-text reason first.

    The classifier callable shape: `classifier(reason_text) -> list[str]`.
    Keeping it pluggable here means tests can pass a deterministic stub
    without going through Groq.
    """
    text = (reason_text or "").strip()
    if not text:
        raise ToolError("reason_text must be non-empty")

    if categories is None:
        if classifier is not None:
            categories = classifier(text)
        else:
            categories = ["other"]

    # Sanitize categories against the taxonomy; unknown values become "other".
    cleaned = [c for c in (categories or []) if c in REJECTION_CATEGORIES]
    if not cleaned:
        cleaned = ["other"]

    reason_id = f"rej_{uuid.uuid4().hex[:12]}"
    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return _q_store_rejection(
        client,
        jd_id=jd_id,
        reason_id=reason_id,
        reason_text=text,
        categories=cleaned,
        created_at=created_at,
    )


# ---------------------------------------------------------------------------
# Tool descriptors — only `get_past_rejections` is LLM-visible.
# `store_rejection` is invoked from service.py, not by the model.
# ---------------------------------------------------------------------------
def _h_get_past_rejections(client: Neo4jClient, args: Dict[str, Any]) -> Dict[str, Any]:
    return get_past_rejections(
        client,
        role_family=str(args.get("role_family", "")).strip(),
        k=int(args.get("k", 10)),
    )


REJECTION_TOOLS: List[ToolDescriptor] = [
    ToolDescriptor(
        name="get_past_rejections",
        description=(
            "Return the most-recent JD rejections for this role family, with the recruiter's "
            "free-text reason and tagged categories. The agent runner calls this DETERMINISTICALLY "
            "before drafting — you should consult the results before writing anything. Calling "
            "this again with a larger k is fine if you want more historical context."
        ),
        parameters={
            "type": "object",
            "properties": {
                "role_family": {"type": "string"},
                "k":           {"type": "integer", "default": 10, "minimum": 1, "maximum": 50},
            },
            "required": ["role_family"],
        },
        handler=_h_get_past_rejections,
    ),
]
