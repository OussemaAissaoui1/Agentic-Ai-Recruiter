"""Public Python API for the JD generation module.

Other parts of the parent project (and tests) call these functions directly
without going through HTTP. The FastAPI router in `api.py` is a thin wrapper
over these.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import asdict
from typing import Any, AsyncIterator, Dict, List, Optional

from agents.jd_generation.config import JDConfig, load_jd_config
from agents.jd_generation.graph.client import Neo4jClient, get_client
from agents.jd_generation.graph.queries import (
    approve_jd_node,
    get_jd_node,
)
from agents.jd_generation.runtime.llm_client import GroqClient
from agents.jd_generation.runtime.runner import (
    GenerateResult,
    RunnerError,
    run_generation,
)
from agents.jd_generation.tools.rejection import (
    REJECTION_CATEGORIES,
    store_rejection as _tool_store_rejection,
)

log = logging.getLogger("agents.jd_generation.service")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _client(cfg: JDConfig) -> Neo4jClient:
    return get_client(cfg)


# ---------------------------------------------------------------------------
# generate_jd — non-streaming
# ---------------------------------------------------------------------------
async def generate_jd(
    *,
    role_title: str,
    level:      str,
    team_id:    Optional[str] = None,
    cfg:        Optional[JDConfig] = None,
) -> GenerateResult:
    """End-to-end synchronous generation. SSE callers use `generate_jd_stream`."""
    cfg = cfg or load_jd_config()
    client = _client(cfg)
    llm = GroqClient(api_key=cfg.llm.api_key, timeout=cfg.llm.request_timeout_s)
    try:
        return await run_generation(
            role_title=role_title, level=level, team_id=team_id,
            client=client, llm=llm, cfg=cfg,
        )
    finally:
        await llm.aclose()


# ---------------------------------------------------------------------------
# generate_jd_stream — async event generator for SSE
# ---------------------------------------------------------------------------
async def generate_jd_stream(
    *,
    role_title: str,
    level:      str,
    team_id:    Optional[str] = None,
    cfg:        Optional[JDConfig] = None,
) -> AsyncIterator[Dict[str, Any]]:
    """Stream `tool_call`, `tool_result`, `final`, `done` events.

    The agent runner emits events to an asyncio.Queue we drain in parallel
    with the run task. When the run task finishes (or raises) we drain
    any remaining events and yield a terminal `done` (or `error`).
    """
    cfg = cfg or load_jd_config()
    client = _client(cfg)
    llm = GroqClient(api_key=cfg.llm.api_key, timeout=cfg.llm.request_timeout_s)

    queue: asyncio.Queue = asyncio.Queue()

    async def _sink(event: Dict[str, Any]) -> None:
        await queue.put(event)

    async def _runner() -> Optional[GenerateResult]:
        try:
            return await run_generation(
                role_title=role_title, level=level, team_id=team_id,
                client=client, llm=llm, cfg=cfg, event_sink=_sink,
            )
        except Exception as e:  # noqa: BLE001 — surface as event, not raise
            await queue.put({"type": "error", "error": f"{type(e).__name__}: {e}"})
            return None

    task = asyncio.create_task(_runner())

    try:
        while True:
            # Wait on either: (a) the runner task finishes, or (b) a new event
            # arrives in the queue. asyncio.wait accepts Tasks directly — the
            # earlier version wrapped `task` in another create_task() which
            # raised "a coroutine was expected, got <Task ...>" because Tasks
            # are Futures, not coroutines.
            get_task = asyncio.create_task(queue.get())
            done, _ = await asyncio.wait(
                {task, get_task}, return_when=asyncio.FIRST_COMPLETED,
            )
            if get_task in done:
                event = get_task.result()
                yield event
                if event.get("type") == "error":
                    yield {"type": "done"}
                    return
                # Loop again — runner task is shared across iterations.
            else:
                # Runner finished; flush remaining queue then exit.
                get_task.cancel()
                while not queue.empty():
                    yield queue.get_nowait()
                result = task.result() if task.done() else None
                if result is not None:
                    yield {"type": "final_full", "result": result.to_dict()}
                yield {"type": "done"}
                return
    finally:
        if not task.done():
            task.cancel()
        await llm.aclose()


# ---------------------------------------------------------------------------
# reject_jd
# ---------------------------------------------------------------------------
class RejectionError(RuntimeError):
    pass


async def reject_jd(
    *,
    jd_id:       str,
    reason_text: str,
    categories:  Optional[List[str]] = None,
    cfg:         Optional[JDConfig] = None,
) -> Dict[str, Any]:
    """Persist a rejection. If `categories` is None and Groq is configured,
    classify the free-text via a one-shot JSON-mode call before writing."""
    cfg = cfg or load_jd_config()
    if not reason_text or not reason_text.strip():
        raise RejectionError("reason_text must be non-empty")

    # Resolve categories before the write — keeps the store_rejection tool
    # synchronous and side-effect-free with respect to the LLM.
    if categories is None and cfg.llm.is_configured:
        categories = await _classify_rejection(reason_text, cfg=cfg)

    client = _client(cfg)
    return await asyncio.to_thread(
        _tool_store_rejection, client,
        jd_id=jd_id, reason_text=reason_text,
        categories=categories, classifier=None,
    )


async def _classify_rejection(text: str, *, cfg: JDConfig) -> List[str]:
    """One-shot JSON classifier returning categories from the taxonomy."""
    llm = GroqClient(api_key=cfg.llm.api_key, timeout=cfg.llm.request_timeout_s)
    try:
        cats = ", ".join(REJECTION_CATEGORIES)
        result = await llm.chat_json(
            model=cfg.llm.model,
            temperature=cfg.llm.temperature_tool_select,
            system=(
                "You classify free-text JD-rejection reasons into a small "
                f"taxonomy. Allowed categories: {cats}. "
                "Output JSON: {\"categories\": [string, ...]}. Pick 1-3 categories. "
                "If unsure, output [\"other\"]."
            ),
            user=text,
        )
        out = [c for c in (result.get("categories") or []) if c in REJECTION_CATEGORIES]
        return out or ["other"]
    finally:
        await llm.aclose()


# ---------------------------------------------------------------------------
# approve_jd / get_jd
# ---------------------------------------------------------------------------
async def approve_jd(*, jd_id: str, cfg: Optional[JDConfig] = None) -> Dict[str, Any]:
    cfg = cfg or load_jd_config()
    client = _client(cfg)
    return await asyncio.to_thread(approve_jd_node, client, jd_id)


async def get_jd(*, jd_id: str, cfg: Optional[JDConfig] = None) -> Optional[Dict[str, Any]]:
    cfg = cfg or load_jd_config()
    client = _client(cfg)
    return await asyncio.to_thread(get_jd_node, client, jd_id)


# ---------------------------------------------------------------------------
# Graph dump + upload — stubs for Phase 6; api.py wires the routes.
# ---------------------------------------------------------------------------
async def dump_graph(*, filter: Optional[str] = None,
                     include_rejections: bool = True,
                     cfg: Optional[JDConfig] = None) -> Dict[str, Any]:
    """D3-shaped {nodes, links} dump.

    Phase 6 implements the full filtering. For Phase 4 we return a small
    payload so the /explorer page can render against /api/jd/graph/dump
    even before Phase 6 lands.
    """
    from agents.jd_generation.graph.dump import dump as _dump  # local import; module added in Phase 6
    cfg = cfg or load_jd_config()
    client = _client(cfg)
    return await asyncio.to_thread(_dump, client, filter, include_rejections)


async def upload_graph(*, file_bytes: bytes, filename: str, mode: str,
                       cfg: Optional[JDConfig] = None) -> Dict[str, Any]:
    """Multipart upload — JSON, CSV, or .zip bundle.

    Implementation lives in `upload/parsers.py` (Phase 6).
    """
    from agents.jd_generation.upload import handle_upload  # Phase 6
    cfg = cfg or load_jd_config()
    client = _client(cfg)
    return await asyncio.to_thread(handle_upload, client, file_bytes, filename, mode, cfg)
