"""FastAPI router for the JD Generation agent.

Mounted at /api/jd/* by main.py. Each route is a thin wrapper over the
public Python API in `service.py` — keep business logic out of this file
so other modules can call the Python API directly without going through
HTTP.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .config import load_jd_config
from .graph.client import Neo4jNotConfigured
from .runtime.llm_client import GroqError
from .runtime.runner import RunnerError
from .service import (
    RejectionError,
    approve_jd,
    generate_jd,
    generate_jd_stream,
    get_jd,
    reject_jd,
)

if TYPE_CHECKING:
    from .agent import JDGenerationAgent

log = logging.getLogger("agents.jd_generation.api")


# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------
class GenerateIn(BaseModel):
    role_title: str = Field(..., min_length=1, max_length=200)
    level:      str = Field(..., pattern=r"^(junior|mid|senior)$")
    team_id:    Optional[str] = Field(None, max_length=64)
    stream:     bool = True


class RejectIn(BaseModel):
    reason_text: str = Field(..., min_length=1, max_length=2000)
    categories:  Optional[List[str]] = None


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
def build_router(agent: "JDGenerationAgent") -> APIRouter:
    router = APIRouter()

    # ---------------------------------------------------------------- health
    @router.get("/health")
    async def health():
        status = await agent.health()
        return JSONResponse(json.loads(status.model_dump_json()))

    # ---------------------------------------------------------------- generate
    @router.post("/generate")
    async def generate(body: GenerateIn):
        _require_ready(agent)
        if body.stream:
            return StreamingResponse(
                _sse(generate_jd_stream(
                    role_title=body.role_title, level=body.level, team_id=body.team_id,
                )),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
            )
        try:
            result = await generate_jd(
                role_title=body.role_title, level=body.level, team_id=body.team_id,
            )
        except RunnerError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except GroqError as e:
            raise HTTPException(status_code=502, detail=f"groq: {e}")
        return result.to_dict()

    # ---------------------------------------------------------------- get / approve / reject
    @router.get("/{jd_id}")
    async def get_one(jd_id: str):
        _require_ready(agent)
        jd = await get_jd(jd_id=jd_id)
        if jd is None:
            raise HTTPException(status_code=404, detail=f"jd {jd_id} not found")
        return jd

    @router.post("/{jd_id}/approve")
    async def approve(jd_id: str):
        _require_ready(agent)
        result = await approve_jd(jd_id=jd_id)
        if not result.get("ok"):
            raise HTTPException(status_code=404, detail=f"jd {jd_id} not found")
        return result

    @router.post("/{jd_id}/reject")
    async def reject(jd_id: str, body: RejectIn):
        _require_ready(agent)
        try:
            result = await reject_jd(
                jd_id=jd_id, reason_text=body.reason_text, categories=body.categories,
            )
        except RejectionError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except GroqError as e:
            # Classifier failed — fall back to ["other"] rather than 5xx,
            # so the rejection is still recorded.
            log.warning("rejection classifier failed (%s); falling back to ['other']", e)
            result = await reject_jd(
                jd_id=jd_id, reason_text=body.reason_text, categories=["other"],
            )
        if not result.get("ok"):
            raise HTTPException(status_code=404, detail=f"jd {jd_id} not found")
        return result

    # ---------------------------------------------------------------- graph
    @router.get("/graph/dump")
    async def graph_dump(
        filter: Optional[str] = Query(default=None),
        include_rejections: bool = Query(default=True),
    ):
        _require_ready(agent)
        try:
            from .service import dump_graph
            return await dump_graph(filter=filter, include_rejections=include_rejections)
        except (ImportError, ModuleNotFoundError):
            # Phase 6 hasn't landed yet — return an honest 501 rather than
            # a confusing 500.
            raise HTTPException(status_code=501, detail="graph dump not implemented yet")

    @router.post("/graph/upload")
    async def graph_upload(
        file: UploadFile = File(...),
        mode: str = Form("augment"),
    ):
        _require_ready(agent)
        try:
            from .service import upload_graph
            cfg = load_jd_config()
            max_bytes = cfg.upload.max_file_size_mb * 1024 * 1024
            body = await file.read()
            if len(body) > max_bytes:
                raise HTTPException(status_code=413,
                                    detail=f"upload exceeds {cfg.upload.max_file_size_mb} MB")
            ext = (file.filename or "").rsplit(".", 1)[-1].lower()
            if ext not in cfg.upload.allowed_extensions:
                raise HTTPException(status_code=400,
                                    detail=f"unsupported extension .{ext}")
            if mode not in ("augment", "replace"):
                raise HTTPException(status_code=400, detail="mode must be augment|replace")
            if mode == "replace" and cfg.upload.prod_guard:
                raise HTTPException(status_code=403,
                                    detail="JD_PROD_GUARD is set — replace uploads refused")
            return await upload_graph(
                file_bytes=body, filename=file.filename or "upload",
                mode=mode,
            )
        except (ImportError, ModuleNotFoundError):
            raise HTTPException(status_code=501, detail="graph upload not implemented yet")

    @router.post("/graph/seed-default")
    async def graph_seed_default():
        _require_ready(agent)
        try:
            from .graph.seed import seed
            import asyncio as _aio
            summary = await _aio.to_thread(seed, True, True)  # clear=True, quiet=True
            return {"ok": True, "summary": summary}
        except Neo4jNotConfigured as e:
            raise HTTPException(status_code=503, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=403, detail=str(e))

    return router


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _require_ready(agent: "JDGenerationAgent") -> None:
    """Refuse non-health routes when startup failed."""
    if agent.startup_error:
        raise HTTPException(status_code=503,
                            detail=f"agent unavailable: {agent.startup_error}")


async def _sse(events):
    """Convert an async iterator of dicts into SSE wire format."""
    try:
        async for ev in events:
            yield f"data: {json.dumps(ev)}\n\n"
    except Exception as e:  # noqa: BLE001 — surface as an SSE error frame
        yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        yield "data: {\"type\":\"done\"}\n\n"
