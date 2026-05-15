"""LLM-backed extraction of per-candidate GA signals.
Given a job description and a CV, runs a locally loaded HuggingFace
Llama-3.2-3B-Instruct and returns a structured object that fills the fields
the Constraint-Aware GA consumes:

    cost              <- LLM-estimated salary expectation (USD)
    gender_female     <- LLM guess from the candidate's first name
    cultural_fit      <- LLM inference from soft-skills / values signals
    experience_score  <- LLM score vs the JD's experience requirement
    salary_alignment  <- LLM assessment vs market for the role

`interview_score` stays 0 by design: it cannot be inferred from a CV.
The hiring team fills it in the UI after the actual interview.
The gated model is loaded on first use; pass HF_TOKEN via env.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import traceback
from typing import Dict, List, Optional

import torch
from pydantic import BaseModel, ValidationError, field_validator
from transformers import AutoModelForCausalLM, AutoTokenizer


HF_MODEL = os.environ.get("LLM_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
# IMPORTANT: HF_TOKEN must come from the environment. The original repo embedded
# a token in source — rotate that token in your HF account; this fork reads from
# the env only. See docs/runbooks/fallback_modes.md for what happens when unset.
HF_TOKEN = os.environ.get("HF_TOKEN", "")


SYSTEM_PROMPT = """You are an expert technical recruiter. You extract structured hiring signals from a CV given a job description, for use in a downstream optimizer.
Return ONLY a single valid JSON object, no prose, no markdown fences. Schema:

{
  "years_experience": number,          // relevant years of work experience (0-40). 0 if unclear.
  "experience_score": number,          // 0.0-1.0. Match of CV's experience to the JD's stated requirement. 1.0 clearly exceeds, 0.0 clearly below, 0.5 uncertain.
  "cultural_fit": number,              // 0.0-1.0. Inferred from soft skills, leadership, teamwork, communication cues vs JD's culture signals. 0.5 if unclear.
  "salary_expectation_usd": number,    // Expected annual salary in USD. Prefer a value stated in the CV. Otherwise estimate from role, seniority, and typical market for the JD. Return 0 only if truly impossible to estimate.
  "salary_alignment": number,          // 0.0-1.0. How aligned salary expectation is with typical market pay for this role. 0.5 if unclear.
  "gender_female": integer,            // 1 if the candidate's first name is typically female, else 0. Use 0 when ambiguous.
  "confidence": number,                // 0.0-1.0. Your overall confidence in this extraction.
  "notes": string                      // Max 20 words: what you inferred and why.
}

Output must be valid JSON only."""


class CandidateSignals(BaseModel):
    """Pydantic-validated output of one LLM extraction call."""

    years_experience: float = 0.0
    experience_score: float = 0.5
    cultural_fit: float = 0.5
    salary_expectation_usd: float = 0.0
    salary_alignment: float = 0.5
    gender_female: int = 0
    confidence: float = 0.0
    notes: str = ""

    @field_validator("experience_score", "cultural_fit", "salary_alignment", "confidence")
    @classmethod
    def _clamp_unit(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    @field_validator("gender_female")
    @classmethod
    def _as_bit(cls, v: int) -> int:
        return 1 if int(v) == 1 else 0

    @field_validator("years_experience", "salary_expectation_usd")
    @classmethod
    def _non_negative(cls, v: float) -> float:
        return max(0.0, float(v))

    def to_ga_metadata(self, candidate_id: str) -> Dict[str, float]:
        """Project into the exact shape CandidateMetaIn / GACandidate want."""
        return {
            "id": candidate_id,
            "cost": float(self.salary_expectation_usd),
            "gender_female": int(self.gender_female),
            "interview_score": 0.0,  # reserved for the hiring team
            "cultural_fit": float(self.cultural_fit),
            "experience_score": float(self.experience_score),
            "salary_alignment": float(self.salary_alignment),
        }


class CVSignalExtractor:
    """CV -> CandidateSignals extractor over a local HuggingFace model.

    Generation is single-slot (one forward pass at a time) to avoid VRAM
    blow-up; `extract_batch` runs sequentially but asynchronously so the
    FastAPI worker stays responsive.
    """

    def __init__(
        self,
        model_name: str = HF_MODEL,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self.model_name = model_name
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = dtype or (
            torch.bfloat16 if self.device.type == "cuda" else torch.float32
        )
        self._load_lock = asyncio.Lock()
        self._gen_lock = asyncio.Lock()
        self._tokenizer = None
        self._model = None
        self._load_error: Optional[str] = None

    async def _ensure_loaded(self) -> None:
        if self._model is not None or self._load_error is not None:
            return
        async with self._load_lock:
            if self._model is not None or self._load_error is not None:
                return
            try:
                if not HF_TOKEN:
                    raise RuntimeError(
                        "HF_TOKEN env var not set; required for gated model "
                        f"{self.model_name}. Falling back to entity-only signals."
                    )
                self._tokenizer = await asyncio.to_thread(
                    AutoTokenizer.from_pretrained, self.model_name, token=HF_TOKEN
                )
                model = await asyncio.to_thread(
                    AutoModelForCausalLM.from_pretrained,
                    self.model_name,
                    torch_dtype=self.dtype,
                    token=HF_TOKEN,
                )
                model = model.to(self.device)
                model.eval()
                if self._tokenizer.pad_token_id is None:
                    self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
                self._model = model
            except Exception as e:
                self._load_error = f"{type(e).__name__}: {e}"

    async def health(self) -> Dict[str, object]:
        """Return LLM status. Does NOT force a load — probes current state."""
        return {
            "backend": "huggingface",
            "model": self.model_name,
            "device": str(self.device),
            "dtype": str(self.dtype).replace("torch.", ""),
            "loaded": self._model is not None,
            "has_token": bool(HF_TOKEN),
            "error": self._load_error,
        }

    def _build_messages(self, jd_text: str, cv_text: str, cv_filename: str):
        # Trim inputs to keep context usage predictable.
        jd_trim = (jd_text or "").strip()[:4000]
        cv_trim = (cv_text or "").strip()[:8000]
        fname = cv_filename or "cv.txt"
        user = (
            f"JOB DESCRIPTION:\n{jd_trim}\n\n"
            f"CV (filename: {fname}):\n{cv_trim}\n\n"
            "Return the JSON now."
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]

    def _parse(self, raw: str) -> CandidateSignals:
        raw = (raw or "").strip()
        # Strip ```json ... ``` fences if the model wrapped the JSON.
        fence = re.search(r"```(?:json)?\s*([\s\S]+?)```", raw)
        body = fence.group(1).strip() if fence else raw
        try:
            obj = json.loads(body)
        except json.JSONDecodeError:
            m = re.search(r"\{[\s\S]*\}", body)
            if not m:
                raise ValueError(f"Model output was not JSON: {raw[:300]}")
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON parse failed: {e}; raw={raw[:300]}")
        try:
            return CandidateSignals(**obj)
        except ValidationError as e:
            raise ValueError(f"Schema mismatch: {e}; raw={raw[:300]}")

    def _generate_sync(self, messages) -> str:
        assert self._model is not None and self._tokenizer is not None
        encoded = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        input_len = encoded["input_ids"].shape[-1]
        with torch.inference_mode():
            out = self._model.generate(
                **encoded,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        gen = out[0, input_len:]
        return self._tokenizer.decode(gen, skip_special_tokens=True)
    async def extract(
        self,
        jd_text: str,
        cv_text: str,
        cv_filename: str = "",
    ) -> CandidateSignals:
        await self._ensure_loaded()
        if self._model is None:
            raise RuntimeError(
                f"LLM failed to load: {self._load_error or 'unknown error'}"
            )
        msgs = self._build_messages(jd_text, cv_text, cv_filename)
        try:
            async with self._gen_lock:
                raw = await asyncio.to_thread(self._generate_sync, msgs)
        except Exception:
            print(
                f"[cv_extractor] {cv_filename!r} GENERATE failed:\n"
                + traceback.format_exc(),
                flush=True,
            )
            raise
        print(f"[cv_extractor] {cv_filename!r} raw output:\n{raw}\n---", flush=True)
        try:
            return self._parse(raw)
        except Exception:
            print(
                f"[cv_extractor] {cv_filename!r} PARSE failed:\n"
                + traceback.format_exc(),
                flush=True,
            )
            raise

    async def extract_batch(
        self,
        jd_text: str,
        cvs: List[Dict[str, str]],
    ) -> Dict[str, Dict[str, object]]:
        """Return {candidate_id: {"signals": CandidateSignals dump, "error": str|None}}."""
        async def _one(c: Dict[str, str]):
            try:
                sig = await self.extract(jd_text, c["text"], c["id"])
                return c["id"], sig, None
            except Exception as e:
                return c["id"], None, f"{type(e).__name__}: {e}"

        results = await asyncio.gather(*[_one(c) for c in cvs])
        out: Dict[str, Dict[str, object]] = {}
        for cid, sig, err in results:
            if sig is None:
                out[cid] = {"signals": CandidateSignals().model_dump(), "error": err}
            else:
                out[cid] = {"signals": sig.model_dump(), "error": None}
        return out
_singleton: Optional[CVSignalExtractor] = None
def get_extractor() -> CVSignalExtractor:
    global _singleton
    if _singleton is None:
        _singleton = CVSignalExtractor()
    return _singleton
