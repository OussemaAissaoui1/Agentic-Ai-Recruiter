"""Orchestrates LLM calls to produce an InterviewReport from context.

Public entry point: score_interview(...). Per-turn calls run with bounded
concurrency. Each per-turn failure is isolated (None scores + error rationale).
The overall pass also fails-soft into a deterministic average-based rule.
"""
from __future__ import annotations

import asyncio
import time
from typing import Optional, Protocol

from .llm_client import GroqError
from .prompts import per_turn_prompt, overall_prompt
from .report import build_overall_fallback, compute_averages
from .schema import (
    InterviewReport,
    OverallAssessment,
    TranscriptTurn,
    TurnScore,
)


class _ChatJSON(Protocol):
    async def chat_json(
        self, *, system: str, user: str, model: str, temperature: float, timeout: float,
        max_retries: int,
    ) -> dict: ...


CONTEXT_WINDOW = 3   # how many prior turns to include in each per-turn prompt


async def score_interview(
    *,
    client,                 # GroqClient | FakeGroq — anything with chat_json
    model: str,
    interview_id: str,
    candidate_name: Optional[str],
    job_title: Optional[str],
    cv_text: str,
    jd_text: str,
    transcript: list[TranscriptTurn],
    behavior_summary: Optional[dict] = None,
    concurrency: int = 4,
    timeout: float = 30.0,
) -> InterviewReport:
    if not transcript:
        raise ValueError("transcript is empty; nothing to score")

    sem = asyncio.Semaphore(concurrency)

    async def _score_one(idx: int, turn: TranscriptTurn) -> TurnScore:
        async with sem:
            prior = transcript[max(0, idx - CONTEXT_WINDOW): idx]
            system, user = per_turn_prompt(
                job_title=job_title or "",
                jd_text=jd_text,
                cv_text=cv_text,
                prior_context=prior,
                question=turn.q,
                answer=turn.a,
            )
            try:
                data = await client.chat_json(
                    system=system, user=user, model=model,
                    temperature=0.2, timeout=timeout, max_retries=1,
                )
                return TurnScore(
                    turn_index=idx,
                    question=turn.q,
                    answer=turn.a,
                    technical_score=_clamp(data.get("technical_score")),
                    technical_rationale=str(data.get("technical_rationale", "")),
                    coherence_score=_clamp(data.get("coherence_score")),
                    coherence_rationale=str(data.get("coherence_rationale", "")),
                )
            except GroqError as exc:
                msg = f"scoring failed: {exc}"
                return TurnScore(
                    turn_index=idx,
                    question=turn.q,
                    answer=turn.a,
                    technical_score=None,
                    technical_rationale=msg,
                    coherence_score=None,
                    coherence_rationale=msg,
                )

    per_turn = await asyncio.gather(
        *(_score_one(i, t) for i, t in enumerate(transcript))
    )

    overall = await _score_overall(
        client=client, model=model, timeout=timeout,
        job_title=job_title or "", jd_text=jd_text, cv_text=cv_text,
        transcript=transcript, per_turn=list(per_turn),
        behavior_summary=behavior_summary,
    )

    return InterviewReport(
        interview_id=interview_id,
        candidate_name=candidate_name,
        job_title=job_title,
        generated_at=time.time(),
        model=model,
        turns=list(per_turn),
        overall=overall,
    )


async def _score_overall(
    *,
    client,
    model: str,
    timeout: float,
    job_title: str,
    jd_text: str,
    cv_text: str,
    transcript: list[TranscriptTurn],
    per_turn: list[TurnScore],
    behavior_summary: Optional[dict] = None,
) -> OverallAssessment:
    tech_avg, coh_avg = compute_averages(per_turn)
    system, user = overall_prompt(
        job_title=job_title, jd_text=jd_text, cv_text=cv_text,
        transcript=transcript, per_turn=per_turn,
        behavior_summary=behavior_summary,
    )
    try:
        data = await client.chat_json(
            system=system, user=user, model=model,
            temperature=0.3, timeout=timeout, max_retries=1,
        )
        rec = data.get("recommendation")
        if not isinstance(rec, str) or rec not in {"strong_hire", "hire", "lean_hire", "no_hire"}:
            rec = build_overall_fallback(tech_avg, coh_avg).recommendation
        return OverallAssessment(
            recommendation=rec,
            technical_avg=tech_avg,
            coherence_avg=coh_avg,
            summary=str(data.get("summary", "")) or "no summary returned",
            strengths=[str(s) for s in (data.get("strengths") or [])][:6],
            concerns=[str(s) for s in (data.get("concerns") or [])][:6],
        )
    except (GroqError, TypeError, ValueError, KeyError) as exc:
        fb = build_overall_fallback(tech_avg, coh_avg)
        return OverallAssessment(
            recommendation=fb.recommendation,
            technical_avg=tech_avg,
            coherence_avg=coh_avg,
            summary=f"overall pass failed; using deterministic fallback: {exc}",
            strengths=fb.strengths,
            concerns=fb.concerns,
        )


def _clamp(value) -> Optional[int]:
    """Coerce LLM output into an int in [0, 5] or None."""
    if value is None:
        return None
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return max(0, min(5, n))
