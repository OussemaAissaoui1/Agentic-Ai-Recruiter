"""
Response Scorer : 
    
Response Scorer — Mini Model for Interview Answer Assessment

Replaces the rule-based keyword analysis with a small LLM (0.5B-1.5B)
running on CPU. Produces structured JSON assessment of candidate answers.

Runs in PARALLEL with prompt building on the GPU, so latency impact
is minimal (~300-500ms on CPU, hidden behind GPU work).

Architecture:
    CPU Thread Pool → Small LLM → JSON Assessment → Injected into recruiter prompt
"""

from __future__ import annotations

import json
import asyncio
import time
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, field


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

Produce this exact JSON:
{{
    "answer_quality": "vague|partial|detailed|off_topic|nonsense|skip_request|clarification_request",
    "current_topic": "what specific CV item is being discussed",
    "candidate_engagement": "engaged|confused|evasive|enthusiastic|frustrated|bored",
    "knowledge_level": "demonstrated|surface_level|uncertain|no_evidence|strong",
    "suggested_action": "follow_up_same_topic|move_to_new_topic|ask_simpler|ask_for_example|rephrase",
    "question_focus": "specific technical aspect to ask about next",
    "knowledge_gaps": ["gap1", "gap2"],
    "acknowledgment": "brief natural 5-10 word acknowledgment of what they said",
    "reasoning": "brief explanation"
}}

Rules for acknowledgment:
- Reference something SPECIFIC they said
- Do NOT use "great", "interesting", "thanks"
- Examples: "So you used Neo4j for the graph layer.", "The SAM integration handled boundary detection."
- If answer was vague, skip acknowledgment (empty string)

Rules for suggested_action:
- "vague" or "partial" → "follow_up_same_topic" or "ask_for_example"
- "detailed" or "strong" → "move_to_new_topic"
- "confused" or "clarification_request" → "ask_simpler" or "rephrase"
- "skip_request" → "move_to_new_topic"
- "nonsense" → "ask_simpler"

Respond with ONLY the JSON."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "cpu",
        max_cv_chars: int = 600,
        max_context_chars: int = 800,
    ):
        self.device = device
        self._max_cv_chars = max_cv_chars
        self._max_context_chars = max_context_chars
        self._ready = False

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM

            print(f"[scorer] Loading {model_name} on {device}...")
            t0 = time.time()

            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map=device,
                trust_remote_code=True,
            )
            self._model.eval()

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
        Called in PARALLEL with other async work.
        """
        if not self._ready:
            return _DEFAULT_OUTPUT

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop, use sync version directly
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
                timeout=3.0,  # Increased from 2s for larger model
            )
            return result
        except asyncio.TimeoutError:
            print("[scorer] Timed out (>3s), using default")
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
                return ScorerOutput(
                    answer_quality=data.get("answer_quality", "unknown"),
                    current_topic=data.get("current_topic", "unknown"),
                    candidate_engagement=data.get("candidate_engagement", "neutral"),
                    knowledge_level=data.get("knowledge_level", "uncertain"),
                    suggested_action=data.get("suggested_action", "follow_up_same_topic"),
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