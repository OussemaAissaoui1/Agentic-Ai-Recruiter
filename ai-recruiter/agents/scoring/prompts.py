"""Prompt templates for per-turn and overall scoring passes.

Two functions, each returning (system, user). Keep prompts boring and
specific — the JSON shape is the contract; the wording can be tuned.
"""
from __future__ import annotations

from typing import Optional

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
    behavior_summary: Optional[dict] = None,
) -> tuple[str, str]:
    transcript_block = "\n\n".join(
        f"Turn {i}\nQ: {t.q}\nA: {t.a}" for i, t in enumerate(transcript)
    ) or "(no transcript)"

    scores_block = "\n".join(
        f"Turn {s.turn_index}: technical={s.technical_score} ({s.technical_rationale})"
        f" | coherence={s.coherence_score} ({s.coherence_rationale})"
        for s in per_turn
    ) or "(no per-turn scores)"

    behavior_block = _render_behavior_block(behavior_summary)

    user = f"""Role: {job_title or "(unspecified)"}

Job description:
{jd_text[:1500]}

Candidate CV (excerpt):
{cv_text[:800]}

Transcript:
{transcript_block}

Per-turn scoring already produced:
{scores_block}
{behavior_block}
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

Treat behavioral signals as advisory only: they refine but never override
technical correctness. Low engagement/confidence on its own is not grounds
for rejection if technical answers are strong.
"""
    return _OVERALL_SYSTEM, user

def _render_behavior_block(summary: Optional[dict]) -> str:
    """Render the camera/voice session summary into a compact prompt block.

    Returns an empty string if no summary is available so the prompt stays
    clean. Voice signals are included only when the audio detector produced
    data; otherwise we explicitly say "voice analysis: not available" so the
    LLM doesn't hallucinate vocal observations.
    """
    if not summary or not isinstance(summary, dict):
        return ""
    
    per_dim = summary.get("per_dimension") or {}
    if not per_dim and not summary.get("notable_moments"):
        return ""

    def _fmt(dim: str) -> str:
        d = per_dim.get(dim) or {}
        if not d:
            return f"{dim}: n/a"
        return (
            f"{dim}: mean={d.get('mean', 'n/a')} "
            f"std={d.get('std', 'n/a')} "
            f"range=[{d.get('min', 'n/a')},{d.get('max', 'n/a')}] "
            f"trend={d.get('trend', 'n/a')}"
        )

    notable = summary.get("notable_moments") or []
    notable_lines = [
        f"  - {m.get('dimension', 'unknown')}: magnitude {m.get('magnitude', 0):.2f}"
        for m in notable[:6]
    ]
    notable_block = "\n".join(notable_lines) if notable_lines else "  (none)"

    qb = summary.get("question_breakdowns") or {}
    qb_lines = []
    for label, vals in list(qb.items())[:6]:
        qb_lines.append(
            f"  - \"{label[:80]}\": engagement={vals.get('engagement_level', 'n/a')} "
            f"confidence={vals.get('confidence_level', 'n/a')} "
            f"cognitive_load={vals.get('cognitive_load', 'n/a')}"
        )
    qb_block = "\n".join(qb_lines) if qb_lines else "  (no per-question breakdown)"

    modalities = []
    # The aggregator does not expose modalities at the summary level, so we
    # infer presence from the dimension data + any audio_stress_prob slot.
    if per_dim:
        modalities.append("vision")
    audio_present = any(
        ("audio" in str(m.get("dimension", "")).lower())
        for m in notable
    )
    voice_line = "voice analysis: present" if audio_present else "voice analysis: not available (audio not routed to vision pipeline)"

    return f"""

Behavioral signals (camera/voice during the live interview):
- duration_seconds: {summary.get('duration_seconds', 'n/a')}
- baseline_calibrated: {summary.get('baseline_calibrated', False)}
- frames_analyzed: {summary.get('frames_analyzed', 0)}
- overall_arc: {summary.get('overall_arc', 'n/a')}
- {voice_line}
- per-dimension stats (0..1 scale; cognitive_load = stress proxy):
  - {_fmt('engagement_level')}
  - {_fmt('confidence_level')}
  - {_fmt('cognitive_load')}
  - {_fmt('emotional_arousal')}
- notable moments (>=2σ deviation):
{notable_block}
- per-question breakdowns:
{qb_block}
"""
