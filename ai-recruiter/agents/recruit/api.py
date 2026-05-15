"""HTTP router for the recruit agent. Mounted at /api/recruit/* by main.py."""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import (
    APIRouter,
    Body,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from configs import resolve as resolve_path

from . import db as recruit_db
from .jd_gen import JDInputs, stream_jd
from .taste.capture import DEFAULT_RECRUITER_ID, capture_decision

# Persistent CV storage. The original file the candidate uploaded lives
# here so the recruiter can preview the source PDF/DOCX/image — not the
# OCR'd text.
CV_STORE = Path(resolve_path("artifacts/cvs"))
CV_STORE.mkdir(parents=True, exist_ok=True)


def _safe_filename_part(name: str) -> str:
    """Strip path separators + collapse weird chars so we never escape CV_STORE."""
    name = os.path.basename(name or "")
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return name[:80] or "cv"

log = logging.getLogger("agents.recruit")


def _resolve_recruiter_id(header_value: Optional[str]) -> str:
    """Map the X-Recruiter-Id request header to a recruiter_id string.

    No auth in this codebase yet — the header is the simplest forward-
    compatible identity surface. If unset, fall back to the singleton
    "hr-default" so the POC works out-of-the-box for a one-recruiter
    install.
    """
    return header_value.strip() if header_value and header_value.strip() else DEFAULT_RECRUITER_ID


# ---------------------------------------------------------------------------
# Pydantic schemas (module scope — Pydantic 2.12 chokes on closures)
# ---------------------------------------------------------------------------
class JobIn(BaseModel):
    title: str
    team: Optional[str] = None
    location: Optional[str] = None
    work_mode: Optional[str] = None
    level: Optional[str] = None
    employment_type: Optional[str] = None
    salary_min: Optional[int] = None
    salary_max: Optional[int] = None
    currency: Optional[str] = "USD"
    description: Optional[str] = None
    must_have: List[str] = Field(default_factory=list)
    nice_to_have: List[str] = Field(default_factory=list)
    status: Optional[str] = "open"


class JobPatch(BaseModel):
    title: Optional[str] = None
    team: Optional[str] = None
    location: Optional[str] = None
    work_mode: Optional[str] = None
    level: Optional[str] = None
    employment_type: Optional[str] = None
    salary_min: Optional[int] = None
    salary_max: Optional[int] = None
    currency: Optional[str] = None
    description: Optional[str] = None
    must_have: Optional[List[str]] = None
    nice_to_have: Optional[List[str]] = None
    status: Optional[str] = None


class JDGenerateIn(BaseModel):
    title: str
    team: Optional[str] = None
    level: Optional[str] = None
    location: Optional[str] = None
    work_mode: Optional[str] = None
    seed_text: Optional[str] = None
    sample_employees: List[Dict[str, Any]] = Field(default_factory=list)


class ApplicationPatch(BaseModel):
    stage: Optional[str] = None
    notes: Optional[str] = None
    candidate_name: Optional[str] = None
    candidate_email: Optional[str] = None


class NotificationIn(BaseModel):
    user_role: Optional[str] = None
    user_id: Optional[str] = None
    kind: Optional[str] = None
    title: Optional[str] = None
    body: Optional[str] = None
    link: Optional[str] = None


class NotificationPatch(BaseModel):
    read: bool = True


class InterviewIn(BaseModel):
    application_id: str
    status: Optional[str] = "scheduled"
    transcript: Optional[List[Dict[str, Any]]] = None
    scores: Optional[Dict[str, Any]] = None
    behavioral: Optional[Dict[str, Any]] = None
    started_at: Optional[float] = None
    ended_at: Optional[float] = None


class InterviewPatch(BaseModel):
    status: Optional[str] = None
    transcript: Optional[List[Dict[str, Any]]] = None
    scores: Optional[Dict[str, Any]] = None
    behavioral: Optional[Dict[str, Any]] = None
    started_at: Optional[float] = None
    ended_at: Optional[float] = None


VALID_STAGES = {"applied", "approved", "interviewed", "offer", "hired", "rejected"}


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
def build_router(agent) -> APIRouter:
    router = APIRouter(tags=["recruit"])

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------
    @router.get("/health")
    async def health():
        return (await agent.health()).model_dump()

    # ------------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------------
    @router.get("/jobs")
    async def list_jobs(
        status: Optional[str] = Query(None),
        q: Optional[str] = Query(None),
    ):
        rows = await asyncio.to_thread(
            recruit_db.list_jobs, agent.conn, status, q
        )
        return rows

    @router.post("/jobs")
    async def create_job(body: JobIn):
        return await asyncio.to_thread(
            recruit_db.create_job, agent.conn, body.model_dump()
        )

    @router.get("/jobs/{job_id}")
    async def get_job(job_id: str):
        job = await asyncio.to_thread(recruit_db.get_job, agent.conn, job_id)
        if job is None:
            raise HTTPException(404, f"job not found: {job_id}")
        return job

    @router.patch("/jobs/{job_id}")
    async def update_job(job_id: str, body: JobPatch):
        job = await asyncio.to_thread(
            recruit_db.update_job,
            agent.conn,
            job_id,
            body.model_dump(exclude_unset=True),
        )
        if job is None:
            raise HTTPException(404, f"job not found: {job_id}")
        return job

    @router.delete("/jobs/{job_id}")
    async def delete_job(job_id: str):
        ok = await asyncio.to_thread(recruit_db.delete_job, agent.conn, job_id)
        if not ok:
            raise HTTPException(404, f"job not found: {job_id}")
        return {"ok": True, "id": job_id}

    # ------------------------------------------------------------------
    # JD generation (SSE)
    # ------------------------------------------------------------------
    @router.post("/jobs/generate")
    async def generate_jd(body: JDGenerateIn):
        nlp_agent = None
        try:
            registry = agent.registry
            if registry is not None:
                nlp_agent = registry.get("nlp")
        except Exception:
            nlp_agent = None

        inputs = JDInputs(
            title=body.title,
            team=body.team,
            level=body.level,
            location=body.location,
            work_mode=body.work_mode,
            seed_text=body.seed_text,
            sample_employees=body.sample_employees,
        )

        async def event_source():
            try:
                async for ev in stream_jd(nlp_agent, inputs):
                    yield f"data: {json.dumps(ev)}\n\n"
            except Exception as e:
                log.exception("jd_gen failed: %s", e)
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

        return StreamingResponse(
            event_source(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # ------------------------------------------------------------------
    # Applications
    # ------------------------------------------------------------------
    @router.get("/applications")
    async def list_applications(
        job_id: Optional[str] = Query(None),
        candidate_email: Optional[str] = Query(None),
        stage: Optional[str] = Query(None),
    ):
        rows = await asyncio.to_thread(
            recruit_db.list_applications,
            agent.conn,
            job_id,
            candidate_email,
            stage,
        )
        return rows

    @router.post("/applications")
    async def create_application(
        job_id: str = Form(...),
        candidate_name: Optional[str] = Form(None),
        candidate_email: Optional[str] = Form(None),
        cv: Optional[UploadFile] = File(None),
    ):
        # Validate job
        job = await asyncio.to_thread(recruit_db.get_job, agent.conn, job_id)
        if job is None:
            raise HTTPException(404, f"job not found: {job_id}")

        # Pre-generate the application id so we can name the persisted CV
        # deterministically — the GET /applications/{id}/cv route uses it.
        aid = f"app-{uuid.uuid4().hex[:12]}"

        cv_text = ""
        cv_filename: Optional[str] = None
        cv_path: Optional[str] = None
        if cv is not None and cv.filename:
            cv_filename = cv.filename
            cv_bytes = await cv.read()
            # Persist the original file to artifacts/cvs/{aid}__{safe_name}.
            safe_name = _safe_filename_part(cv.filename)
            persisted = CV_STORE / f"{aid}__{safe_name}"
            try:
                persisted.write_bytes(cv_bytes)
                cv_path = str(persisted)
            except Exception as e:
                log.warning("cv persist failed for %s: %s", cv.filename, e)
                cv_path = None
            # OCR / text extraction for matching + scoring downstream.
            try:
                from agents.matching.backend.utils.pdf_reader import read_any
                with tempfile.TemporaryDirectory() as tmp:
                    dest = os.path.join(tmp, cv.filename)
                    with open(dest, "wb") as fh:
                        fh.write(cv_bytes)
                    try:
                        cv_text = (read_any(dest) or "").strip()
                    except Exception as e:
                        log.warning("cv parse failed for %s: %s", cv.filename, e)
                        cv_text = ""
            except Exception as e:
                log.warning("cv handling failed: %s", e)

        # Score against the job description via the matching agent (best effort).
        fit_score: Optional[float] = None
        matched_skills: List[str] = []
        missing_skills: List[str] = []
        if cv_text and job.get("description"):
            try:
                ranked = await _rank_with_matching(
                    agent.registry, job["description"], cv_text, candidate_name or cv_filename or "candidate"
                )
                if ranked:
                    top = ranked[0]
                    fit_score = float(top.get("fit_score") or 0.0)
                    me = top.get("matching_entities") or {}
                    re_ = top.get("resume_entities") or {}
                    je = top.get("job_entities") or {}
                    matched_skills = _flatten_entities(me)
                    missing_skills = _diff_entities(je, re_)
            except Exception as e:
                log.warning("matching rank failed: %s", e)

        payload = {
            "id": aid,
            "job_id": job_id,
            "candidate_name": candidate_name,
            "candidate_email": candidate_email,
            "cv_text": cv_text or None,
            "cv_filename": cv_filename,
            "cv_path": cv_path,
            "fit_score": fit_score,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
        }
        app_row = await asyncio.to_thread(
            recruit_db.create_application, agent.conn, payload
        )

        # Notify the recruiter that a new application has arrived.
        try:
            display_name = (
                candidate_name
                or candidate_email
                or cv_filename
                or "Anonymous candidate"
            )
            await asyncio.to_thread(
                recruit_db.create_notification,
                agent.conn,
                {
                    "user_role": "recruiter",
                    "user_id": None,
                    "kind": "new_application",
                    "title": f"New application: {display_name}",
                    "body": f"For role: {job.get('title')}",
                    "link": f"/applications/{app_row['id']}",
                },
            )
        except Exception as e:
            log.warning("recruiter notification failed: %s", e)

        return app_row

    @router.get("/applications/{app_id}")
    async def get_application(app_id: str):
        row = await asyncio.to_thread(
            recruit_db.get_application, agent.conn, app_id
        )
        if row is None:
            raise HTTPException(404, f"application not found: {app_id}")
        return row

    @router.api_route(
        "/applications/{app_id}/cv", methods=["GET", "HEAD"]
    )
    async def get_application_cv(app_id: str):
        """Stream the original uploaded CV file (PDF / DOCX / image / txt).

        Served `inline` so the browser previews it in an iframe rather than
        triggering a download. Falls back to 404 if the application has no
        persisted file (e.g. legacy rows that pre-date the cv_path column).

        Also responds to HEAD so the frontend can probe existence cheaply
        before mounting the iframe — Starlette's FileResponse skips the
        body for HEAD requests automatically.
        """
        row = await asyncio.to_thread(
            recruit_db.get_application, agent.conn, app_id
        )
        if row is None:
            raise HTTPException(404, f"application not found: {app_id}")
        cv_path = await asyncio.to_thread(
            recruit_db.get_application_cv_path, agent.conn, app_id
        )
        if not cv_path:
            raise HTTPException(404, "no CV on file for this application")
        path = Path(cv_path)
        if not path.is_file():
            raise HTTPException(404, "CV file is missing on disk")
        # Defence-in-depth: never serve anything outside CV_STORE.
        try:
            path.resolve().relative_to(CV_STORE.resolve())
        except ValueError:
            raise HTTPException(403, "CV path is outside the allowed store")
        media_type, _ = mimetypes.guess_type(str(path))
        if not media_type:
            media_type = "application/octet-stream"
        download_name = row.get("cv_filename") or path.name
        # FastAPI's FileResponse defaults to attachment disposition. Force
        # `inline` so PDFs/images preview directly in the iframe.
        response = FileResponse(
            str(path), media_type=media_type, filename=download_name
        )
        response.headers["Content-Disposition"] = (
            f'inline; filename="{download_name}"'
        )
        return response

    @router.patch("/applications/{app_id}")
    async def update_application(
        app_id: str,
        body: ApplicationPatch,
        x_recruiter_id: Optional[str] = Header(default=None, alias="X-Recruiter-Id"),
    ):
        if body.stage is not None and body.stage not in VALID_STAGES:
            raise HTTPException(
                400,
                f"invalid stage '{body.stage}'. Allowed: {sorted(VALID_STAGES)}",
            )

        # Capture the previous stage *before* the update so we can detect
        # transitions into 'rejected' and dispatch the candidate notification
        # exactly once (idempotent).
        prev = await asyncio.to_thread(
            recruit_db.get_application, agent.conn, app_id
        )
        if prev is None:
            raise HTTPException(404, f"application not found: {app_id}")

        row = await asyncio.to_thread(
            recruit_db.update_application,
            agent.conn,
            app_id,
            body.model_dump(exclude_unset=True),
        )
        if row is None:
            raise HTTPException(404, f"application not found: {app_id}")

        if (
            body.stage == "rejected"
            and prev.get("stage") != "rejected"
        ):
            job = await asyncio.to_thread(
                recruit_db.get_job, agent.conn, row["job_id"]
            )
            job_title = (job or {}).get("title", "the role")
            try:
                await asyncio.to_thread(
                    recruit_db.create_notification,
                    agent.conn,
                    {
                        "user_role": "candidate",
                        "user_id": row.get("candidate_email"),
                        "kind": "application_rejected",
                        "title": f"Update on your application for {job_title}",
                        "body": (
                            f"Hi {row.get('candidate_name') or 'there'} — your "
                            f"application for {job_title} has been closed. "
                            f"Thanks for your interest."
                        ),
                        "link": "/c/applications",
                    },
                )
            except Exception as e:
                log.warning("rejection notification failed: %s", e)

            # Recruiter-taste capture (additive — never blocks the transition).
            try:
                await asyncio.to_thread(
                    capture_decision,
                    agent.conn,
                    _resolve_recruiter_id(x_recruiter_id),
                    app_id,
                    "rejected",
                )
            except Exception as e:                       # pragma: no cover
                log.warning("taste capture (rejected) failed: %s", e)

        return row

    @router.post("/applications/{app_id}/invite-interview")
    async def invite_interview(
        app_id: str,
        x_recruiter_id: Optional[str] = Header(default=None, alias="X-Recruiter-Id"),
    ):
        app_row = await asyncio.to_thread(
            recruit_db.get_application, agent.conn, app_id
        )
        if app_row is None:
            raise HTTPException(404, f"application not found: {app_id}")
        job = await asyncio.to_thread(
            recruit_db.get_job, agent.conn, app_row["job_id"]
        )
        job_title = (job or {}).get("title", "the role")

        title = f"You've been approved for an interview: {job_title}"
        body = (
            f"Hi {app_row.get('candidate_name') or 'there'} — a recruiter has "
            f"approved your application for {job_title}. Click the link to "
            f"start the AI interview."
        )
        link = f"/c/interview/{app_id}"

        # Idempotency: if the application is already approved, return the
        # existing row + most recent invite notification without dispatching a
        # second one. Guards against double-clicks and accidental retries.
        if app_row.get("stage") == "approved":
            existing = await asyncio.to_thread(
                recruit_db.list_notifications,
                agent.conn,
                "candidate",
                app_row.get("candidate_email"),
                False,
            )
            existing_invite = next(
                (n for n in existing
                 if n.get("kind") == "interview_invite" and n.get("link") == link),
                None,
            )
            return {
                "ok": True,
                "application_id": app_id,
                "notification": existing_invite,
                "invite": {"title": title, "body": body, "link": link},
            }

        notif = await asyncio.to_thread(
            recruit_db.create_notification,
            agent.conn,
            {
                "user_role": "candidate",
                "user_id": app_row.get("candidate_email"),
                "kind": "interview_invite",
                "title": title,
                "body": body,
                "link": link,
            },
        )
        await asyncio.to_thread(
            recruit_db.update_application,
            agent.conn,
            app_id,
            {"stage": "approved"},
        )

        # Recruiter-taste capture (additive — never blocks the transition).
        try:
            await asyncio.to_thread(
                capture_decision,
                agent.conn,
                _resolve_recruiter_id(x_recruiter_id),
                app_id,
                "approved",
            )
        except Exception as e:                           # pragma: no cover
            log.warning("taste capture (approved) failed: %s", e)

        return {
            "ok": True,
            "application_id": app_id,
            "notification": notif,
            "invite": {"title": title, "body": body, "link": link},
        }

    # ------------------------------------------------------------------
    # Notifications
    # ------------------------------------------------------------------
    @router.get("/notifications")
    async def list_notifications(
        user_role: Optional[str] = Query(None),
        user_id: Optional[str] = Query(None),
        unread_only: bool = Query(False),
    ):
        rows = await asyncio.to_thread(
            recruit_db.list_notifications,
            agent.conn,
            user_role,
            user_id,
            unread_only,
        )
        return rows

    @router.post("/notifications")
    async def create_notification(body: NotificationIn):
        return await asyncio.to_thread(
            recruit_db.create_notification,
            agent.conn,
            body.model_dump(exclude_unset=True),
        )

    @router.patch("/notifications/{notif_id}")
    async def patch_notification(notif_id: str, body: NotificationPatch):
        row = await asyncio.to_thread(
            recruit_db.mark_notification_read, agent.conn, notif_id, body.read
        )
        if row is None:
            raise HTTPException(404, f"notification not found: {notif_id}")
        return row

    # ------------------------------------------------------------------
    # Interviews (minimal scaffolding)
    # ------------------------------------------------------------------
    @router.get("/interviews")
    async def list_interviews(
        application_id: Optional[str] = Query(None),
    ):
        rows = await asyncio.to_thread(
            recruit_db.list_interviews, agent.conn, application_id
        )
        return rows

    @router.post("/interviews")
    async def create_interview(body: InterviewIn):
        # Confirm application exists
        app_row = await asyncio.to_thread(
            recruit_db.get_application, agent.conn, body.application_id
        )
        if app_row is None:
            raise HTTPException(
                404, f"application not found: {body.application_id}"
            )
        # Gate: an interview can only be created once the recruiter has
        # approved the application (stage='approved') or after a prior
        # interview has already been run (stage='interviewed', re-entry).
        if app_row.get("stage") not in {"approved", "interviewed"}:
            raise HTTPException(
                403,
                f"application stage '{app_row.get('stage')}' is not eligible "
                f"for an interview — recruiter approval required.",
            )
        return await asyncio.to_thread(
            recruit_db.create_interview, agent.conn, body.model_dump()
        )

    @router.get("/interviews/{iid}")
    async def get_interview(iid: str):
        row = await asyncio.to_thread(recruit_db.get_interview, agent.conn, iid)
        if row is None:
            raise HTTPException(404, f"interview not found: {iid}")
        return row

    @router.patch("/interviews/{iid}")
    async def patch_interview(iid: str, body: InterviewPatch):
        row = await asyncio.to_thread(
            recruit_db.update_interview,
            agent.conn,
            iid,
            body.model_dump(exclude_unset=True),
        )
        if row is None:
            raise HTTPException(404, f"interview not found: {iid}")
        return row

    # ------------------------------------------------------------------
    # Recruiter-taste (POC). Mounted under /api/recruit/taste/* — the
    # whole feature is additive and lives in agents/recruit/taste/.
    # ------------------------------------------------------------------
    from .taste.features import (
        FEATURE_LABELS as _FEATURE_LABELS,
        get_or_compute_features as _get_or_compute_features,
    )
    from .taste.learner import (
        fit_recruiter_weights as _fit_recruiter_weights,
        load_profile as _load_profile,
        reset_profile as _reset_profile,
        score_candidate as _score_candidate,
    )

    def _profile_to_payload(profile) -> Dict[str, Any]:
        """Render a RecruiterProfile into the JSON shape the UI consumes.

        Top-positive / top-negative are computed against the training mean
        (i.e. against a "median candidate"), so the labels reflect what the
        recruiter weights regardless of which candidate is being scored.
        """
        # For the profile view, surface the bare standardised weights —
        # they're already on a comparable scale across features.
        ranked = sorted(
            profile.weights.items(), key=lambda kv: -kv[1],
        )
        positives = [
            {"label": _FEATURE_LABELS[name], "weight": round(w, 3)}
            for name, w in ranked if w > 0
        ][:5]
        negatives = [
            {"label": _FEATURE_LABELS[name], "weight": round(w, 3)}
            for name, w in reversed(ranked) if w < 0
        ][:3]
        return {
            "recruiter_id": profile.recruiter_id,
            "n_decisions": profile.n_decisions,
            "confidence": profile.confidence.value,
            "version": profile.version,
            "updated_at": profile.updated_at,
            "top_positive": positives,
            "top_negative": negatives,
        }

    @router.get("/taste/me/profile")
    async def taste_get_profile(
        x_recruiter_id: Optional[str] = Header(default=None, alias="X-Recruiter-Id"),
    ):
        rid = _resolve_recruiter_id(x_recruiter_id)
        profile = await asyncio.to_thread(_load_profile, agent.conn, rid)
        if profile is None:
            return {
                "recruiter_id": rid,
                "n_decisions": 0,
                "confidence": "cold",
                "version": 0,
                "updated_at": None,
                "top_positive": [],
                "top_negative": [],
            }
        return _profile_to_payload(profile)

    @router.get("/taste/score/{application_id}")
    async def taste_score(
        application_id: str,
        x_recruiter_id: Optional[str] = Header(default=None, alias="X-Recruiter-Id"),
    ):
        rid = _resolve_recruiter_id(x_recruiter_id)

        try:
            features = await asyncio.to_thread(
                _get_or_compute_features, agent.conn, application_id,
            )
        except LookupError:
            raise HTTPException(404, f"application not found: {application_id}")

        score = await asyncio.to_thread(
            _score_candidate, agent.conn, rid, features,
        )
        return {
            "application_id": application_id,
            "recruiter_id": rid,
            "score": round(score.score, 4),
            "confidence": score.confidence.value,
            "top_positive": [
                {"label": label, "contribution": round(c, 3)}
                for label, c in score.top_positive
            ],
            "top_negative": [
                {"label": label, "contribution": round(c, 3)}
                for label, c in score.top_negative
            ],
        }

    @router.post("/taste/me/refit")
    async def taste_refit(
        x_recruiter_id: Optional[str] = Header(default=None, alias="X-Recruiter-Id"),
    ):
        rid = _resolve_recruiter_id(x_recruiter_id)
        profile = await asyncio.to_thread(_fit_recruiter_weights, agent.conn, rid)
        return _profile_to_payload(profile)

    @router.post("/taste/me/reset")
    async def taste_reset(
        x_recruiter_id: Optional[str] = Header(default=None, alias="X-Recruiter-Id"),
    ):
        rid = _resolve_recruiter_id(x_recruiter_id)
        existed = await asyncio.to_thread(_reset_profile, agent.conn, rid)
        return {"recruiter_id": rid, "reset": existed}

    return router


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
async def _rank_with_matching(
    registry, jd_text: str, cv_text: str, cv_id: str
) -> List[Dict[str, Any]]:
    """Call the matching agent's `rank` if it's registered, else return []."""
    if registry is None:
        return []
    try:
        matching = registry.get("matching")
    except Exception:
        return []
    try:
        return await matching.rank(jd_text, [{"id": cv_id, "text": cv_text}])
    except Exception as e:
        log.warning("matching.rank raised: %s", e)
        return []


def _flatten_entities(ent: Dict[str, Any]) -> List[str]:
    """Best-effort flatten of `{category: [skills]}` → unique list."""
    out: List[str] = []
    if isinstance(ent, dict):
        for v in ent.values():
            if isinstance(v, (list, tuple)):
                for s in v:
                    if s and str(s).strip():
                        out.append(str(s).strip())
            elif isinstance(v, str) and v.strip():
                out.append(v.strip())
    # de-duplicate while preserving order
    seen: set[str] = set()
    uniq: List[str] = []
    for s in out:
        k = s.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(s)
    return uniq


def _diff_entities(
    jd_entities: Dict[str, Any], cv_entities: Dict[str, Any]
) -> List[str]:
    jd = {s.lower(): s for s in _flatten_entities(jd_entities)}
    cv = {s.lower() for s in _flatten_entities(cv_entities)}
    return [orig for k, orig in jd.items() if k not in cv]
