"""HTTP router for the matching agent. Mounted under /api/matching by main.py."""

from __future__ import annotations

import json
import os
import tempfile
from typing import Dict, List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel


def build_router(agent) -> APIRouter:
    """Build a FastAPI router bound to a MatchingAgent instance."""
    router = APIRouter(tags=["matching"])

    # ---------------------------------------------------------------- pydantic
    class GAConstraintsIn(BaseModel):
        budget: float = 0.0
        min_female_ratio: float = 0.0
        min_fit_threshold: float = 0.0
        role_requirements: Dict[str, int] = {}

    class CandidateMetaIn(BaseModel):
        id: str
        cost: float = 0.0
        gender_female: int = 0
        interview_score: float = 0.0
        cultural_fit: float = 0.0
        experience_score: float = 0.0
        salary_alignment: float = 0.0

    # ---------------------------------------------------------------- routes
    @router.get("/health")
    async def health():
        h = await agent.health()
        return h.model_dump()

    @router.post("/extract-signals")
    async def extract_signals(
        job_description: str = Form(...),
        files: List[UploadFile] = File(...),
    ):
        from .backend.utils.pdf_reader import read_any
        from .backend.agents.cv_extractor import get_extractor

        if not files:
            raise HTTPException(400, "No resumes uploaded.")
        if not (job_description or "").strip():
            raise HTTPException(400, "Job description is required for extraction.")

        upload_dir = str(agent.upload_dir)
        resumes: List[Dict[str, str]] = []
        with tempfile.TemporaryDirectory(dir=upload_dir) as tmp:
            for f in files:
                dest = os.path.join(tmp, f.filename)
                with open(dest, "wb") as fh:
                    fh.write(await f.read())
                text = read_any(dest)
                if text:
                    resumes.append({"id": f.filename, "text": text})

        if not resumes:
            raise HTTPException(400, "Could not extract text from any uploaded file.")

        try:
            out = await agent.extract_signals(job_description, resumes)
        except RuntimeError as e:
            extractor = get_extractor()
            raise HTTPException(
                503,
                f"LLM unavailable: {e}. Set HF_TOKEN and accept the license at "
                f"https://huggingface.co/{extractor.model_name}.",
            )
        return {"signals": out, "model": get_extractor().model_name}

    @router.post("/rank")
    async def rank(
        job_description: str = Form(...),
        apply_ga_flag: bool = Form(False),
        metadata_json: str = Form("[]"),
        constraints_json: str = Form("{}"),
        files: List[UploadFile] = File(...),
    ):
        from .backend.utils.pdf_reader import read_any

        if not files:
            raise HTTPException(400, "No resumes uploaded.")
        try:
            meta_list = [CandidateMetaIn(**m) for m in json.loads(metadata_json)]
        except Exception as e:
            raise HTTPException(400, f"Bad metadata_json: {e}")
        try:
            constraints = GAConstraintsIn(**json.loads(constraints_json))
        except Exception as e:
            raise HTTPException(400, f"Bad constraints_json: {e}")

        upload_dir = str(agent.upload_dir)
        resumes: List[Dict[str, str]] = []
        with tempfile.TemporaryDirectory(dir=upload_dir) as tmp:
            for f in files:
                dest = os.path.join(tmp, f.filename)
                with open(dest, "wb") as fh:
                    fh.write(await f.read())
                text = read_any(dest)
                if text:
                    resumes.append({"id": f.filename, "text": text})

        if not resumes:
            raise HTTPException(400, "Could not extract text from any uploaded file.")

        ranked = await agent.rank(job_description, resumes)
        response = {"ranked": ranked, "model_info": {
            "tapjfnn_loaded": (await agent.health()).detail.get("tapjfnn_loaded", False),
            "gnn_loaded": (await agent.health()).detail.get("gnn_loaded", False),
        }}

        if apply_ga_flag:
            ga_inputs = {m.id: m.model_dump() for m in meta_list}
            missing = [r["id"] for r in ranked if r["id"] not in ga_inputs]
            if missing:
                raise HTTPException(400, f"Missing GA metadata for: {missing}")
            response["ga"] = await agent.apply_ga(
                ranked, ga_inputs, constraints.model_dump()
            )

        return response

    return router
