"""
Response Scorer — Mini Model for Interview Answer Assessment

Replaces the rule-based keyword analysis with a small LLM (0.5B-1.5B)
running on CPU. Produces structured JSON assessment of candidate answers.

Runs in PARALLEL with prompt building on the GPU, so latency impact
is minimal (~300-500ms on CPU, hidden behind GPU work).

Architecture:
    CPU Thread Pool → Small LLM → JSON Assessment → Injected into recruiter prompt

Bug fixes (v2):
- Added HuggingFace Hub HTTP timeout via env var to prevent indefinite hangs
  when the Hub is slow or when downloading large model files for the first time.
- Wrapped from_pretrained calls in a thread with a hard deadline (load_timeout_sec)
  so a stuck download raises TimeoutError instead of blocking forever.
- Increased score() timeout from 2s → 3s to accommodate Qwen2.5-1.5B on slow CPUs.
"""

from __future__ import annotations

import os
import json
import asyncio
import threading
import time
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, field

# -----------------------------------------------------------------
# Set HuggingFace Hub HTTP timeout BEFORE any HF imports.
# The default is effectively infinite which causes silent hangs
# when the Hub CDN is unreachable or the connection drops mid-download.
# 30 s per HTTP request is generous for a chunk download; the retry
# logic in huggingface_hub will re-attempt on transient failures.
# -----------------------------------------------------------------
os.environ.setdefault("HUGGINGFACE_HUB_HTTP_TIMEOUT", "30")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")


@dataclass(frozen=True)
class ScorerOutput:
    """Structured assessment from the mini analyzer model."""
    answer_quality: str
    current_topic: str
    candidate_engagement: str
    knowledge_level: str
    suggested_action: str
    question_focus: str
    knowledge_gaps: Tuple[str, ...]
    acknowledgment: str
    raw_reasoning: str
    latency_ms: float

    def to_instruction(self) -> str:
        """Convert assessment to an instruction string for the recruiter model."""
        parts = []
        if self.suggested_action == "follow_up_same_topic":
            parts.append("Ask a FOLLOW-UP on the SAME topic.")
        elif self.suggested_action == "move_to_new_topic":
            parts.append("Move to a NEW topic from the CV.")
        elif self.suggested_action == "ask_simpler":
            parts.append("Ask a SIMPLER, more specific question on the same topic.")
        elif self.suggested_action == "ask_for_example":
            parts.append("Ask for a CONCRETE EXAMPLE.")
        elif self.suggested_action == "rephrase":
            parts.append("REPHRASE your previous question more clearly.")

        if self.question_focus:
            parts.append(f"Focus on: {self.question_focus}")
        if self.knowledge_gaps:
            gaps = ", ".join(self.knowledge_gaps[:3])
            parts.append(f"Gaps to probe: {gaps}")
        if self.acknowledgment:
            parts.append(f"Start with this acknowledgment: \"{self.acknowledgment}\"")

        return " | ".join(parts) if parts else ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer_quality": self.answer_quality,
            "current_topic": self.current_topic,
            "candidate_engagement": self.candidate_engagement,
            "knowledge_level": self.knowledge_level,
            "suggested_action": self.suggested_action,
            "question_focus": self.question_focus,
            "knowledge_gaps": list(self.knowledge_gaps),
            "acknowledgment": self.acknowledgment,
            "latency_ms": self.latency_ms,
        }


# Default fallback when analyzer is unavailable or fails
_DEFAULT_OUTPUT = ScorerOutput(
    answer_quality="unknown",
    current_topic="unknown",
    candidate_engagement="neutral",
    knowledge_level="uncertain",
    suggested_action="follow_up_same_topic",
    question_focus="",
    knowledge_gaps=(),
    acknowledgment="",
    raw_reasoning="analyzer_unavailable",
    latency_ms=0.0,
)


def _run_with_timeout(func, timeout_sec: int, error_msg: str = "Operation timed out"):
    """
    Run *func* in a daemon thread; raise TimeoutError if it doesn't finish
    within *timeout_sec* seconds.

    This is the same pattern used in vllm_engine.py — a thread-based timeout
    that works correctly even when called from a non-main thread (unlike
    signal.alarm which is main-thread only).
    """
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


class ResponseScorer:
    """
    Small LLM that analyzes candidate responses on CPU.

    Produces a structured JSON assessment including:
    - Answer quality (vague/partial/detailed/off_topic/nonsense)
    - Candidate engagement level
    - Knowledge assessment
    - Suggested next action
    - A natural acknowledgment sentence
    - Knowledge gaps to probe

    Performance:
    - Qwen2.5-0.5B: ~200-400ms on CPU
    - Qwen2.5-1.5B: ~400-800ms on CPU
    - Runs in asyncio thread pool, doesn't block the event loop
    """

    SCORER_PROMPT = """Analyze this interview exchange and respond with ONLY valid JSON.

## Candidate CV (summary):
{cv_summary}

## Recruiter's last question:
{last_question}

## Candidate's answer:
{answer}

## Conversation context:
{context}

Produce this exact JSON (pick ONE value for each field, do NOT combine with |):
{{
    "answer_quality": "<pick ONE: vague OR partial OR detailed OR off_topic OR nonsense OR skip_request OR clarification_request>",
    "current_topic": "what specific CV item is being discussed",
    "candidate_engagement": "<pick ONE: engaged OR confused OR evasive OR enthusiastic OR frustrated OR bored>",
    "knowledge_level": "<pick ONE: demonstrated OR surface_level OR uncertain OR no_evidence OR strong>",
    "suggested_action": "<pick ONE: follow_up_same_topic OR move_to_new_topic OR ask_simpler OR ask_for_example OR rephrase>",
    "question_focus": "specific technical aspect to ask about next",
    "knowledge_gaps": ["gap1", "gap2"],
    "acknowledgment": "brief natural 5-10 word acknowledgment of what they said",
    "reasoning": "brief explanation"
}}

Rules for acknowledgment:
- Reference something SPECIFIC they said
- Do NOT use "great", "interesting", "thanks"
- Examples: "So you used Neo4j for the graph layer.", "The SAM integration handled boundary detection."
- If answer was vague or evasive, use empty string ""

Rules for suggested_action:
- "vague" or "partial" → "follow_up_same_topic" or "ask_for_example"
- "detailed" or "strong" → "move_to_new_topic"
- "confused" or "clarification_request" → "ask_simpler" or "rephrase"
- "skip_request" or "evasive" → "move_to_new_topic"
- "nonsense" → "ask_simpler"

Respond with ONLY the JSON. Pick exactly ONE value per field."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "cpu",
        max_cv_chars: int = 600,
        max_context_chars: int = 800,
        load_timeout_sec: int = 300,   # 5 min: enough for first-time download (~1 GB)
    ):
        self.device = device
        self._max_cv_chars = max_cv_chars
        self._max_context_chars = max_context_chars
        self._ready = False

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM

            print(f"[scorer] Loading {model_name} on {device} "
                  f"(timeout={load_timeout_sec}s)...")
            t0 = time.time()

            # ---------------------------------------------------------
            # Wrap the download/load in a thread with a hard deadline.
            # Without this, a slow Hub connection or a corrupt cache
            # causes the process to hang indefinitely with no feedback.
            # ---------------------------------------------------------
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
                        f"Scorer model load timed out after {load_timeout_sec}s. "
                        "Check your internet connection or use a cached model."
                    ),
                )
            except TimeoutError as te:
                print(f"[scorer] WARNING: {te}")
                print("[scorer] Falling back to rule-based analysis")
                self._ready = False
                return

            param_count = sum(p.numel() for p in self._model.parameters()) / 1e6
            load_time = time.time() - t0
            print(f"[scorer] Ready: {param_count:.0f}M params, loaded in {load_time:.1f}s")
            self._ready = True

        except Exception as e:
            print(f"[scorer] WARNING: Failed to load analyzer model: {e}")
            print("[scorer] Falling back to rule-based analysis")
            self._ready = False

    @property
    def is_ready(self) -> bool:
        return self._ready

    def _format_context(
        self,
        history: List[Tuple[str, str]],
    ) -> str:
        if not history:
            return "(Opening turn — no prior conversation)"
        parts = []
        for i, (cand, rec) in enumerate(history[-3:], 1):
            parts.append(f"Recruiter: {rec}")
            parts.append(f"Candidate: {cand}")
        return "\n".join(parts)

    def _get_last_question(self, history: List[Tuple[str, str]]) -> str:
        if history:
            return history[-1][1]
        return "(Opening — no previous question)"

    async def score(
        self,
        candidate_cv: str,
        conversation_history: List[Tuple[str, str]],
        latest_answer: str,
    ) -> ScorerOutput:
        """
        Analyze the candidate's latest answer asynchronously.

        Runs CPU inference in a thread pool to avoid blocking the event loop.
        Called in PARALLEL with other async work (see agent.py).

        The timeout here is intentionally generous (3 s) because:
        - Qwen2.5-1.5B on a slow CPU can take ~800 ms
        - We want to maximise the chance the scorer finishes before the GPU
          has even started loading the prompt, so the instruction is useful.
        - In agent.py we wrap this call in asyncio.wait_for(timeout=0.9s)
          to cap the *wait* time, so a slow scorer never blocks generation.
        """
        if not self._ready:
            return _DEFAULT_OUTPUT

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return self._score_sync(candidate_cv, conversation_history, latest_answer)

        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    self._score_sync,
                    candidate_cv,
                    conversation_history,
                    latest_answer,
                ),
                timeout=5.0,
            )
            return result
        except asyncio.TimeoutError:
            print("[scorer] Timed out (>5s), using default")
            return _DEFAULT_OUTPUT
        except Exception as e:
            print(f"[scorer] Error: {e}")
            return _DEFAULT_OUTPUT

    def _score_sync(
        self,
        candidate_cv: str,
        conversation_history: List[Tuple[str, str]],
        latest_answer: str,
    ) -> ScorerOutput:
        import torch

        t0 = time.perf_counter()

        prompt_text = self.SCORER_PROMPT.format(
            cv_summary=candidate_cv[:self._max_cv_chars],
            last_question=self._get_last_question(conversation_history),
            answer=latest_answer[:500],
            context=self._format_context(conversation_history)[:self._max_context_chars],
        )

        messages = [
            {"role": "system", "content": "You produce only valid JSON. No other text."},
            {"role": "user", "content": prompt_text},
        ]

        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=250,
                temperature=0.1,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        generated = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        latency_ms = (time.perf_counter() - t0) * 1000
        return self._parse_output(generated, latency_ms)

    @staticmethod
    def _repair_json(raw: str) -> str:
        """
        Best-effort repair of JSON emitted by small LLMs.
        Handles the most common failure modes:
          - Trailing commas before } or ]
          - Single-quoted strings  → double-quoted
          - Python-style True/False/None → JSON booleans/null
          - Unquoted string values (limited heuristic)
        """
        import re
        s = raw
        # Remove trailing commas before closing brace/bracket
        s = re.sub(r",\s*([\]}])", r"\1", s)
        # Single quotes → double quotes (only around simple string values/keys)
        s = re.sub(r"'([^'\n]*)'", r'"\1"', s)
        # Python literals
        s = re.sub(r"\bTrue\b", "true", s)
        s = re.sub(r"\bFalse\b", "false", s)
        s = re.sub(r"\bNone\b", "null", s)
        return s

    _VALID_QUALITY = {"vague", "partial", "detailed", "off_topic", "nonsense", "skip_request", "clarification_request"}
    _VALID_ENGAGEMENT = {"engaged", "confused", "evasive", "enthusiastic", "frustrated", "bored"}
    _VALID_KNOWLEDGE = {"demonstrated", "surface_level", "uncertain", "no_evidence", "strong"}
    _VALID_ACTION = {"follow_up_same_topic", "move_to_new_topic", "ask_simpler", "ask_for_example", "rephrase"}

    def _parse_output(self, text: str, latency_ms: float) -> ScorerOutput:
        try:
            json_start = text.find("{")
            json_end = text.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                raw_json = text[json_start:json_end]
                try:
                    data = json.loads(raw_json)
                except json.JSONDecodeError:
                    data = json.loads(self._repair_json(raw_json))

                # Sanitize: if model output pipe-separated enum, take first valid value
                def _pick_one(val: str, valid_set: set, default: str) -> str:
                    if not val or not isinstance(val, str):
                        return default
                    if "|" in val:
                        for part in val.split("|"):
                            part = part.strip()
                            if part in valid_set:
                                return part
                        return default
                    return val if val in valid_set else default

                return ScorerOutput(
                    answer_quality=_pick_one(data.get("answer_quality", ""), self._VALID_QUALITY, "vague"),
                    current_topic=data.get("current_topic", "unknown"),
                    candidate_engagement=_pick_one(data.get("candidate_engagement", ""), self._VALID_ENGAGEMENT, "neutral"),
                    knowledge_level=_pick_one(data.get("knowledge_level", ""), self._VALID_KNOWLEDGE, "uncertain"),
                    suggested_action=_pick_one(data.get("suggested_action", ""), self._VALID_ACTION, "follow_up_same_topic"),
                    question_focus=data.get("question_focus", ""),
                    knowledge_gaps=tuple(data.get("knowledge_gaps", [])),
                    acknowledgment=data.get("acknowledgment", ""),
                    raw_reasoning=data.get("reasoning", ""),
                    latency_ms=latency_ms,
                )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"[scorer] JSON parse failed: {e}")

        return ScorerOutput(
            answer_quality="unknown",
            current_topic="unknown",
            candidate_engagement="neutral",
            knowledge_level="uncertain",
            suggested_action="follow_up_same_topic",
            question_focus="",
            knowledge_gaps=(),
            acknowledgment="",
            raw_reasoning=f"parse_failed: {text[:200]}",
            latency_ms=latency_ms,
        )