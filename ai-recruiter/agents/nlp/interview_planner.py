"""
Interview Planner — LLM-driven control-flow agent for the interview loop.

Replaces the deterministic `InterviewStateTracker.should_follow_up()` rule
engine with a small LLM that reasons about state, recent answers, and the
scorer's structured analysis, then outputs a typed `PlannerDecision`
(action + chain-of-thought reasoning + optional target topic).

This is the file that turns the interview from "state machine with prose
slots" into a real ReAct-style agent loop:
    scorer (perceive)  →  planner (decide)  →  recruiter LLM (act)
                            ↑                       ↓
                            └── state observation ──┘

If the planner fails (model unavailable, JSON malformed, timeout) the
caller falls back to `state.should_follow_up()` so reliability is
preserved. The planner is *additive*: every existing rule still acts as
a safety net.
"""

from __future__ import annotations

import os
import json
import asyncio
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Set HF Hub timeout BEFORE the transformers import — same pattern used by
# response_scorer.py. Without this a slow CDN can hang the process.
os.environ.setdefault("HUGGINGFACE_HUB_HTTP_TIMEOUT", "30")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")


# Action vocabulary the planner is allowed to choose from. Keep this list
# closed — `_pick_one()` snaps any free-form output back to the closest
# legal value so a misbehaving small LLM cannot derail the pipeline.
ACTION_FOLLOW_UP = "follow_up_same_topic"
ACTION_NEW_TOPIC = "move_to_new_topic"
ACTION_ASK_SIMPLER = "ask_simpler"
ACTION_ASK_EXAMPLE = "ask_for_example"
ACTION_WRAP_THEN_MOVE = "wrap_topic_then_move"
ACTION_CLARIFY = "clarify_previous"
ACTION_ACK_SKIP = "acknowledge_skip"

VALID_ACTIONS = {
    ACTION_FOLLOW_UP,
    ACTION_NEW_TOPIC,
    ACTION_ASK_SIMPLER,
    ACTION_ASK_EXAMPLE,
    ACTION_WRAP_THEN_MOVE,
    ACTION_CLARIFY,
    ACTION_ACK_SKIP,
}

# Actions that mean "stay on the current topic" (counts as a follow-up).
FOLLOW_UP_ACTIONS = {
    ACTION_FOLLOW_UP,
    ACTION_ASK_SIMPLER,
    ACTION_ASK_EXAMPLE,
    ACTION_CLARIFY,
}


@dataclass(frozen=True)
class PlannerDecision:
    """Structured decision produced by the planner agent for one turn."""
    action: str
    reasoning: str                # the agent's chain-of-thought
    target_topic: Optional[str]   # populated when action == new_topic
    confidence: float             # planner self-rated 0–1
    latency_ms: float
    raw_text: str = ""            # raw model output, for debugging

    @property
    def should_follow_up(self) -> bool:
        return self.action in FOLLOW_UP_ACTIONS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "reasoning": self.reasoning,
            "target_topic": self.target_topic,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
        }


@dataclass(frozen=True)
class StateSnapshot:
    """The slice of `InterviewStateTracker` the planner reasons over.

    Passed into `plan()` so the planner does not depend on the tracker's
    full surface area. This is the contract — adding a new state variable
    means extending this snapshot, which is intentional.
    """
    turn_count: int
    current_topic: Optional[str]
    topic_depth: int
    topics_covered: List[str]
    topics_remaining: List[str]
    consecutive_vague: int
    max_followups_per_topic: int


# Default fallback when planner is unavailable. Mirrors the scorer's
# default action so behavior degrades gracefully toward "ask a follow-up".
_DEFAULT_DECISION = PlannerDecision(
    action=ACTION_FOLLOW_UP,
    reasoning="planner_unavailable_default",
    target_topic=None,
    confidence=0.0,
    latency_ms=0.0,
)


def _run_with_timeout(func, timeout_sec: int, error_msg: str = "Operation timed out"):
    """Same thread-with-deadline pattern as response_scorer.py."""
    result_holder: list = [None]
    exc_holder: list = [None]

    def _wrapper():
        try:
            result_holder[0] = func()
        except Exception as exc:
            exc_holder[0] = exc

    t = threading.Thread(target=_wrapper, daemon=True)
    t.start()
    t.join(timeout=timeout_sec)
    if t.is_alive():
        raise TimeoutError(error_msg)
    if exc_holder[0] is not None:
        raise exc_holder[0]
    return result_holder[0]


class InterviewPlanner:
    """LLM-driven planner that decides the next interview action.

    Loads its own Qwen2.5-1.5B by default, but `attach_shared_model()`
    lets it reuse a tokenizer+model already loaded by ResponseScorer to
    avoid duplicating ~3 GB of weights on the same device.
    """

    PLANNER_PROMPT = """You are an interview strategy agent. The recruiter (Alex) is conducting a live technical interview. Your job is to decide what Alex should do NEXT — not to write the question, just to choose the strategy.

## Interview state:
- Turn number: {turn_count}
- Current topic: {current_topic}
- Times we have already followed up on this topic: {topic_depth} of {max_followups} max
- Consecutive low-quality answers: {consecutive_vague}
- Topics already covered: {topics_covered}
- Topics still on the queue: {topics_remaining}

## Scorer's analysis of the candidate's last answer:
- Quality: {quality}
- Engagement: {engagement}
- Knowledge level: {knowledge_level}
- Suggested focus: {question_focus}
- Knowledge gaps: {knowledge_gaps}
- Scorer's recommended action: {scorer_action}

## Recent conversation (most recent last):
{recent_history}

## Candidate's latest answer:
"{latest_answer}"

## Your decision space — pick exactly ONE:
- "follow_up_same_topic": dig deeper on the SAME topic — answer was promising but incomplete.
- "ask_for_example": answer was abstract — request a concrete example or anecdote.
- "ask_simpler": candidate is struggling — drop to a more accessible angle on the same topic.
- "clarify_previous": candidate asked for clarification — rephrase the prior question, do not advance.
- "move_to_new_topic": this topic is exhausted — pick a fresh topic from `topics_remaining`.
- "wrap_topic_then_move": acknowledge what they said, then move on (good for partial-but-acceptable answers).
- "acknowledge_skip": candidate explicitly asked to skip — respect it and move to a new topic.

## Reasoning rules:
- Respect `max_followups`: if topic_depth >= max_followups, you MUST move on (new_topic / wrap_topic_then_move / acknowledge_skip).
- If quality is "detailed" or "strong", prefer move_to_new_topic.
- If quality is "vague" and topic_depth < max_followups, prefer ask_simpler or ask_for_example.
- If engagement is "evasive" or "frustrated" and topic_depth >= 1, move on rather than push.
- If `topics_remaining` is empty, you must stay on the current topic (any follow_up flavor) — do NOT pick move_to_new_topic.
- target_topic is REQUIRED when action is move_to_new_topic or wrap_topic_then_move; pick from topics_remaining.

## Output — ONLY this JSON, nothing else:
{{
  "reasoning": "2-3 sentences of step-by-step reasoning citing the state and analysis fields above",
  "action": "<one of: follow_up_same_topic | ask_for_example | ask_simpler | clarify_previous | move_to_new_topic | wrap_topic_then_move | acknowledge_skip>",
  "target_topic": "<topic from topics_remaining if moving on, else empty string>",
  "confidence": <number between 0 and 1>
}}"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "cpu",
        load_timeout_sec: int = 300,
        autoload: bool = True,
    ) -> None:
        self.device = device
        self.model_name = model_name
        self._ready = False
        self._tokenizer = None
        self._model = None
        self._owns_model = True

        if not autoload:
            return

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM

            print(f"[planner] Loading {model_name} on {device} "
                  f"(timeout={load_timeout_sec}s)...")
            t0 = time.time()

            def _load_models():
                tok = AutoTokenizer.from_pretrained(
                    model_name, trust_remote_code=True
                )
                mdl = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    device_map=device,
                    trust_remote_code=True,
                )
                mdl.eval()
                return tok, mdl

            try:
                self._tokenizer, self._model = _run_with_timeout(
                    _load_models,
                    timeout_sec=load_timeout_sec,
                    error_msg=(
                        f"Planner model load timed out after {load_timeout_sec}s. "
                        "Check your internet connection or use a cached model."
                    ),
                )
            except TimeoutError as te:
                print(f"[planner] WARNING: {te}")
                self._ready = False
                return

            param_count = sum(p.numel() for p in self._model.parameters()) / 1e6
            load_time = time.time() - t0
            print(f"[planner] Ready: {param_count:.0f}M params, loaded in {load_time:.1f}s")
            self._ready = True

        except Exception as e:
            print(f"[planner] WARNING: Failed to load planner model: {e}")
            print("[planner] Falling back to deterministic state machine")
            self._ready = False

    def attach_shared_model(self, tokenizer, model, device: str) -> None:
        """Reuse a tokenizer+model already loaded by another component
        (typically ResponseScorer). Saves ~3 GB of duplicate weights."""
        self._tokenizer = tokenizer
        self._model = model
        self.device = device
        self._owns_model = False
        self._ready = tokenizer is not None and model is not None
        if self._ready:
            print(f"[planner] Attached to shared model on {device}")

    @property
    def is_ready(self) -> bool:
        return self._ready

    @staticmethod
    def _format_history(history: List[Tuple[str, str]], max_turns: int = 4) -> str:
        if not history:
            return "(opening turn — no prior conversation)"
        parts: List[str] = []
        for cand, rec in history[-max_turns:]:
            if rec:
                parts.append(f"  Recruiter: {rec}")
            if cand:
                parts.append(f"  Candidate: {cand}")
        return "\n".join(parts) if parts else "(opening turn)"

    @staticmethod
    def _format_list(items: List[str], empty: str = "(none)") -> str:
        if not items:
            return empty
        return ", ".join(items[:8])

    async def plan(
        self,
        scorer_analysis: Dict[str, Any],
        state: StateSnapshot,
        latest_answer: str,
        conversation_history: List[Tuple[str, str]],
    ) -> PlannerDecision:
        """Decide the next interview action. Async-safe; runs CPU/GPU
        inference in a thread pool so it never blocks the event loop."""
        if not self._ready:
            return _DEFAULT_DECISION

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return self._plan_sync(scorer_analysis, state, latest_answer, conversation_history)

        try:
            return await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    self._plan_sync,
                    scorer_analysis,
                    state,
                    latest_answer,
                    conversation_history,
                ),
                timeout=4.0,
            )
        except asyncio.TimeoutError:
            print("[planner] Timed out (>4s), using default")
            return _DEFAULT_DECISION
        except Exception as e:
            print(f"[planner] Error: {e}")
            return _DEFAULT_DECISION

    def _plan_sync(
        self,
        scorer_analysis: Dict[str, Any],
        state: StateSnapshot,
        latest_answer: str,
        conversation_history: List[Tuple[str, str]],
    ) -> PlannerDecision:
        import torch

        t0 = time.perf_counter()

        prompt_text = self.PLANNER_PROMPT.format(
            turn_count=state.turn_count,
            current_topic=state.current_topic or "(none yet)",
            topic_depth=state.topic_depth,
            max_followups=state.max_followups_per_topic,
            consecutive_vague=state.consecutive_vague,
            topics_covered=self._format_list(state.topics_covered),
            topics_remaining=self._format_list(state.topics_remaining),
            quality=scorer_analysis.get("quality", "unknown"),
            engagement=scorer_analysis.get("engagement", "neutral"),
            knowledge_level=scorer_analysis.get("knowledge_level", "uncertain"),
            question_focus=scorer_analysis.get("question_focus") or "(none)",
            knowledge_gaps=self._format_list(list(scorer_analysis.get("knowledge_gaps", ()))),
            scorer_action=scorer_analysis.get("suggested_action", "follow_up_same_topic"),
            recent_history=self._format_history(conversation_history),
            latest_answer=(latest_answer or "")[:600],
        )

        messages = [
            {"role": "system", "content": "You are a JSON-only decision agent. You output exactly one JSON object and nothing else."},
            {"role": "user", "content": prompt_text},
        ]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=180,
                temperature=0.2,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        generated = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        return self._parse_output(generated, state, latency_ms)

    @staticmethod
    def _repair_json(raw: str) -> str:
        import re
        s = raw
        s = re.sub(r",\s*([\]}])", r"\1", s)
        s = re.sub(r"'([^'\n]*)'", r'"\1"', s)
        s = re.sub(r"\bTrue\b", "true", s)
        s = re.sub(r"\bFalse\b", "false", s)
        s = re.sub(r"\bNone\b", "null", s)
        return s

    def _parse_output(
        self, text: str, state: StateSnapshot, latency_ms: float,
    ) -> PlannerDecision:
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        if json_start == -1 or json_end <= json_start:
            return PlannerDecision(
                action=ACTION_FOLLOW_UP,
                reasoning=f"parse_failed_no_json: {text[:120]}",
                target_topic=None,
                confidence=0.0,
                latency_ms=latency_ms,
                raw_text=text,
            )

        raw_json = text[json_start:json_end]
        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError:
            try:
                data = json.loads(self._repair_json(raw_json))
            except json.JSONDecodeError:
                return PlannerDecision(
                    action=ACTION_FOLLOW_UP,
                    reasoning=f"parse_failed_invalid_json: {raw_json[:120]}",
                    target_topic=None,
                    confidence=0.0,
                    latency_ms=latency_ms,
                    raw_text=text,
                )

        action = data.get("action", "")
        if isinstance(action, str) and action in VALID_ACTIONS:
            chosen_action = action
        else:
            chosen_action = ACTION_FOLLOW_UP

        # Safety net: respect max_followups even if the LLM ignored it.
        if (chosen_action in FOLLOW_UP_ACTIONS
                and state.topic_depth >= state.max_followups_per_topic
                and state.topics_remaining):
            chosen_action = ACTION_NEW_TOPIC

        # Safety net: don't move on if there's nowhere to move to.
        if chosen_action in (ACTION_NEW_TOPIC, ACTION_WRAP_THEN_MOVE) and not state.topics_remaining:
            chosen_action = ACTION_FOLLOW_UP

        target_topic = data.get("target_topic") or None
        if isinstance(target_topic, str):
            target_topic = target_topic.strip() or None
        if target_topic and target_topic not in state.topics_remaining:
            # The LLM hallucinated a topic. Snap to the first remaining one.
            target_topic = state.topics_remaining[0] if state.topics_remaining else None

        try:
            confidence = float(data.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.5

        reasoning = data.get("reasoning", "")
        if not isinstance(reasoning, str):
            reasoning = str(reasoning)

        return PlannerDecision(
            action=chosen_action,
            reasoning=reasoning[:500],
            target_topic=target_topic,
            confidence=confidence,
            latency_ms=latency_ms,
            raw_text=text,
        )
