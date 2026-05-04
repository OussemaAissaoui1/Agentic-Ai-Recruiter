"""Prompt templates for per-turn and overall scoring passes.

Two functions, each returning (system, user). Keep prompts boring and
specific — the JSON shape is the contract; the wording can be tuned.
"""
from __future__ import annotations

from .schema import TranscriptTurn, TurnScore


_PER_TURN_SYSTEM = (
    "You are a senior technical interviewer calibrating answers against a "
    "specific role. Score the candidate's answer on two independent axes "
    "(technical correctness and coherence). Return ONLY valid JSON matching "
    "the requested schema. Do not include any text outside the JSON."
)

_RUBRIC = """
Technical (0-5):
  0 = no signal, refusal, or nonsense
  1 = clearly wrong or fabricated
  2 = vague / hand-wavy; little evidence of hands-on knowledge
  3 = generic correctness; lacks concrete detail
  4 = correct with concrete detail and a believable trade-off
  5 = expert-level; specific, accurate, mentions edge cases or trade-offs

Coherence (0-5):
  0 = incoherent / unintelligible
  1 = fragmented; hard to follow
  2 = present but disorganised
  3 = clear enough; some redundancy
  4 = well-structured, easy to follow
  5 = articulate, well-paced, no wasted words
"""


def per_turn_prompt(
    *,
    job_title: str,
    jd_text: str,
    cv_text: str,
    prior_context: list[TranscriptTurn],
    question: str,
    answer: str,
) -> tuple[str, str]:
    """Render the per-turn scoring prompt.

    *prior_context* is the trailing window of turns before this one (already
    truncated by the caller).
    """
    ctx_lines = [f"Q: {t.q}\nA: {t.a}" for t in prior_context]
    context_block = "\n\n".join(ctx_lines) if ctx_lines else "(no prior context)"

    user = f"""Role: {job_title or "(unspecified)"}

Job description:
{jd_text[:1500]}

Candidate CV (excerpt):
{cv_text[:600]}

{_RUBRIC}

Recent conversation context:
{context_block}

== Score this exchange ==
Question:
{question}

Answer:
{answer}

Return JSON exactly in this shape:
{{
  "technical_score": <0-5 integer>,
  "technical_rationale": "<one or two sentences referencing the answer>",
  "coherence_score": <0-5 integer>,
  "coherence_rationale": "<one or two sentences>"
}}
"""
    return _PER_TURN_SYSTEM, user


_OVERALL_SYSTEM = (
    "You are the lead technical interviewer summarising a completed "
    "interview into a hiring recommendation. Be calibrated and specific. "
    "Return ONLY valid JSON matching the requested schema."
)


def overall_prompt(
    *,
    job_title: str,
    jd_text: str,
    cv_text: str,
    transcript: list[TranscriptTurn],
    per_turn: list[TurnScore],
) -> tuple[str, str]:
    transcript_block = "\n\n".join(
        f"Turn {i}\nQ: {t.q}\nA: {t.a}" for i, t in enumerate(transcript)
    ) or "(no transcript)"

    scores_block = "\n".join(
        f"Turn {s.turn_index}: technical={s.technical_score} ({s.technical_rationale})"
        f" | coherence={s.coherence_score} ({s.coherence_rationale})"
        for s in per_turn
    ) or "(no per-turn scores)"

    user = f"""Role: {job_title or "(unspecified)"}

Job description:
{jd_text[:1500]}

Candidate CV (excerpt):
{cv_text[:800]}

Transcript:
{transcript_block}

Per-turn scoring already produced:
{scores_block}

Produce a hiring recommendation. Pick exactly one value for "recommendation"
from: "strong_hire", "hire", "lean_hire", "no_hire".

Return JSON exactly in this shape:
{{
  "recommendation": "<one of the four values above>",
  "summary": "<3-5 sentence narrative tying the per-turn scores to the role>",
  "strengths": ["<bullet>", "<bullet>"],
  "concerns": ["<bullet>", "<bullet>"]
}}

Calibration:
  - strong_hire: clear technical excellence, no significant concerns
  - hire: solid match; minor reservations
  - lean_hire: split signal; would need a second opinion
  - no_hire: insufficient evidence, fabrications, or major concerns
"""
    return _OVERALL_SYSTEM, user
