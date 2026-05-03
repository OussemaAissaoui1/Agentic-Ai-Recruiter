"""JD generation: tries the NLP agent's vLLM engine when available, falls
back to a deterministic templated JD that always works.

Both paths yield SSE-shaped event dicts. The router serializes them.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, AsyncIterator, Dict, List, Optional

log = logging.getLogger("agents.recruit")


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
class JDInputs:
    """Plain container for JD-generation inputs."""

    def __init__(
        self,
        title: str,
        team: Optional[str] = None,
        level: Optional[str] = None,
        location: Optional[str] = None,
        work_mode: Optional[str] = None,
        seed_text: Optional[str] = None,
        sample_employees: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.title = (title or "Untitled Role").strip()
        self.team = (team or "").strip() or None
        self.level = (level or "").strip() or None
        self.location = (location or "").strip() or None
        self.work_mode = (work_mode or "").strip() or None
        self.seed_text = (seed_text or "").strip() or None
        self.sample_employees = sample_employees or []

    def sample_titles(self) -> List[str]:
        out: List[str] = []
        for s in self.sample_employees:
            t = (s.get("title") or s.get("role") or "").strip() if isinstance(s, dict) else str(s).strip()
            if t:
                out.append(t)
        return out


# ---------------------------------------------------------------------------
# Templated fallback
# ---------------------------------------------------------------------------
LEVEL_HEADLINE = {
    "junior": "early-career",
    "mid": "experienced",
    "senior": "senior",
    "staff": "staff-level",
    "principal": "principal-level",
    "lead": "lead",
}


def _templated_jd(inputs: JDInputs) -> Dict[str, Any]:
    """Build a structured JD from inputs. Always succeeds."""
    title = inputs.title
    team_clause = f" on the {inputs.team} team" if inputs.team else ""
    level_word = LEVEL_HEADLINE.get((inputs.level or "").lower(), inputs.level or "")
    level_clause = f" {level_word}" if level_word else ""
    where_clause = ""
    if inputs.location and inputs.work_mode:
        where_clause = f" Based in {inputs.location} ({inputs.work_mode})."
    elif inputs.location:
        where_clause = f" Based in {inputs.location}."
    elif inputs.work_mode:
        where_clause = f" {inputs.work_mode.capitalize()}-friendly."

    seed = inputs.seed_text or ""
    samples = inputs.sample_titles()
    peer_clause = ""
    if samples:
        peer_clause = (
            "You'll work alongside teammates with backgrounds like "
            + ", ".join(samples[:5])
            + "."
        )

    p1 = (
        f"We're hiring a{level_clause} {title}{team_clause}. "
        f"You'll own meaningful product surfaces end-to-end and partner "
        f"closely with cross-functional teammates to ship quickly and "
        f"thoughtfully.{where_clause}"
    ).strip()

    p2 = (
        f"In your first 90 days you'll ramp up on our codebase and stack, "
        f"take ownership of an initial scope, and start shipping changes "
        f"that move customer metrics. {peer_clause}"
    ).strip()

    p3_lead = (
        seed if seed
        else (
            "We care about taste, judgement, and the ability to grow. We "
            "value strong written communication, async collaboration, and a "
            "bias for shipping. You'll be trusted to make calls and supported "
            "when you need help."
        )
    )

    description = "\n\n".join([p1, p2, p3_lead])

    must_have = [
        f"Proven experience in a similar {title} role",
        f"Strong fundamentals relevant to {inputs.team or 'the team'}",
        "Excellent written communication",
        "Comfort owning ambiguous, cross-functional work",
        "Bias toward shipping iteratively",
    ]
    nice_to_have = [
        "Prior startup or scale-up experience",
        "Open-source or community contributions",
        "Mentorship or technical leadership track record",
        "Domain expertise in HR-tech or AI tooling",
    ]
    return {
        "description": description,
        "must_have": must_have,
        "nice_to_have": nice_to_have,
    }


def _chunk_words(text: str, size: int = 4) -> List[str]:
    """Split text into space-preserving chunks for streaming."""
    parts = re.findall(r"\S+\s*|\s+", text)
    out: List[str] = []
    buf = ""
    for i, p in enumerate(parts):
        buf += p
        if (i + 1) % size == 0:
            out.append(buf)
            buf = ""
    if buf:
        out.append(buf)
    return out


async def stream_templated(inputs: JDInputs) -> AsyncIterator[Dict[str, Any]]:
    jd = _templated_jd(inputs)
    text = jd["description"]
    for chunk in _chunk_words(text, size=3):
        yield {"type": "token", "text": chunk}
        await asyncio.sleep(0.01)
    yield {"type": "final", "jd": jd}
    yield {"type": "done"}


# ---------------------------------------------------------------------------
# vLLM-backed path (best-effort)
# ---------------------------------------------------------------------------
def _build_prompt(inputs: JDInputs) -> str:
    parts: List[str] = []
    parts.append(f"Role title: {inputs.title}")
    if inputs.team:
        parts.append(f"Team: {inputs.team}")
    if inputs.level:
        parts.append(f"Level: {inputs.level}")
    if inputs.location:
        parts.append(f"Location: {inputs.location}")
    if inputs.work_mode:
        parts.append(f"Work mode: {inputs.work_mode}")
    if inputs.sample_titles():
        parts.append("Sample existing teammates: " + ", ".join(inputs.sample_titles()[:6]))
    if inputs.seed_text:
        parts.append(f"Recruiter notes: {inputs.seed_text}")
    parts.append(
        "Write a compelling 3-paragraph job description. "
        "Be concrete about scope and impact. Avoid clichés. "
        "Do NOT include a list of requirements — just the prose."
    )
    return "\n".join(parts)


async def stream_vllm(
    nlp_agent: Any, inputs: JDInputs
) -> AsyncIterator[Dict[str, Any]]:
    """Stream via the NLP agent's vLLM engine.

    Falls back to templated streaming on any failure.
    """
    inner = getattr(nlp_agent, "_inner", None)
    if inner is None or not inner.is_ready:
        log.info("recruit.jd_gen: nlp not ready, using template")
        async for ev in stream_templated(inputs):
            yield ev
        return

    try:
        engine = inner.engine
    except Exception as e:
        log.warning("recruit.jd_gen: engine unavailable (%s); using template", e)
        async for ev in stream_templated(inputs):
            yield ev
        return

    system_prompt = (
        "You are a thoughtful technical writer helping recruiters draft "
        "job descriptions. Write in a clear, candidate-respectful voice. "
        "Avoid corporate clichés and rocket-ship language."
    )
    candidate_input = _build_prompt(inputs)

    accumulated: List[str] = []
    try:
        async for piece in engine.generate_streaming(
            system_prompt=system_prompt,
            history=[],
            candidate_input=candidate_input,
            max_tokens=420,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
        ):
            if not piece:
                continue
            accumulated.append(piece)
            yield {"type": "token", "text": piece}
    except Exception as e:
        log.warning("recruit.jd_gen: vLLM streaming failed (%s); falling back", e)
        async for ev in stream_templated(inputs):
            yield ev
        return

    description = "".join(accumulated).strip()
    if not description:
        async for ev in stream_templated(inputs):
            yield ev
        return

    # Reuse the templated requirements arrays — the LLM stays focused on prose.
    tpl = _templated_jd(inputs)
    jd = {
        "description": description,
        "must_have": tpl["must_have"],
        "nice_to_have": tpl["nice_to_have"],
    }
    yield {"type": "final", "jd": jd}
    yield {"type": "done"}


async def stream_jd(
    nlp_agent: Optional[Any], inputs: JDInputs
) -> AsyncIterator[Dict[str, Any]]:
    """Public entrypoint: vLLM if possible, otherwise templated."""
    if nlp_agent is not None:
        async for ev in stream_vllm(nlp_agent, inputs):
            yield ev
        return
    async for ev in stream_templated(inputs):
        yield ev
