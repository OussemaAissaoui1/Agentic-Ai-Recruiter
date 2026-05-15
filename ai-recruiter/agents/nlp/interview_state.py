"""
Interview State Tracker — v6

Now accepts ScorerOutput from the mini analyzer model instead of
doing keyword-based analysis. Falls back to basic heuristics
when the analyzer is unavailable.
"""

from __future__ import annotations

import random
from typing import Optional, Tuple, Set, List
from dataclasses import dataclass

try:
    from response_scorer import ScorerOutput
except ImportError:
    ScorerOutput = None


@dataclass(frozen=True)
class AnswerAnalysis:
    """Analysis results — either from the analyzer model or fallback."""
    word_count: int
    is_detailed: bool
    wants_to_skip: bool
    is_nonsense: bool
    is_asking_for_clarity: bool
    quality: str
    engagement: str
    knowledge_level: str
    suggested_action: str
    question_focus: str
    knowledge_gaps: Tuple[str, ...]
    acknowledgment: str


class InterviewStateTracker:
    """
    Stateful interview tracker — v6.

    Accepts analyzer model output (ScorerOutput) for rich decisions.
    Falls back to basic word-count heuristics when analyzer is unavailable.
    """

    MAX_FOLLOWUPS_PER_TOPIC: int = 3
    CONSECUTIVE_VAGUE_LIMIT: int = 3

    SKIP_KEYWORDS: Tuple[str, ...] = (
        "can we pass", "let's skip", "move on", "next question",
        "skip this", "pass on this", "different topic",
        "ask about something else",
    )

    CLARITY_KEYWORDS: Tuple[str, ...] = (
        "can you be more specific", "what do you mean",
        "which project", "what project", "can you clarify",
        "i don't understand", "be more specific", "what exactly",
        "which one", "mentioned in my cv",
    )

    def __init__(self) -> None:
        self.current_topic: Optional[str] = None
        self.topic_depth: int = 0
        self.topics_covered: List[str] = []
        self.turn_count: int = 0
        self.questions_asked: Set[str] = set()
        self.question_word_sets: List[Tuple[str, frozenset]] = []
        self.consecutive_vague: int = 0
        self._topic_queue: List[str] = []
        self._topic_queue_index: int = 0

    def reset(self) -> None:
        self.current_topic = None
        self.topic_depth = 0
        self.topics_covered.clear()
        self.turn_count = 0
        self.questions_asked.clear()
        self.question_word_sets.clear()
        self.consecutive_vague = 0
        self._topic_queue_index = 0

    def set_topic_queue(self, topics: List[str]) -> None:
        self._topic_queue = list(topics)
        self._topic_queue_index = 0

    def _get_next_topic(self) -> Optional[str]:
        while self._topic_queue_index < len(self._topic_queue):
            topic = self._topic_queue[self._topic_queue_index]
            self._topic_queue_index += 1
            if topic not in self.topics_covered:
                return topic
        return None

    # ------------------------------------------------------------------
    # Build analysis from scorer output or fallback
    # ------------------------------------------------------------------

    def build_analysis(
        self,
        answer: str,
        scorer_output: Optional[object] = None,
    ) -> AnswerAnalysis:
        """
        Build AnswerAnalysis from scorer model output or fallback heuristics.
        """
        word_count = len(answer.split())
        answer_lower = answer.lower()

        if scorer_output is not None and hasattr(scorer_output, "answer_quality"):
            quality = scorer_output.answer_quality
            return AnswerAnalysis(
                word_count=word_count,
                is_detailed=quality in ("detailed", "strong"),
                wants_to_skip=quality == "skip_request",
                is_nonsense=quality == "nonsense",
                is_asking_for_clarity=quality == "clarification_request",
                quality=quality,
                engagement=scorer_output.candidate_engagement,
                knowledge_level=scorer_output.knowledge_level,
                suggested_action=scorer_output.suggested_action,
                question_focus=scorer_output.question_focus,
                knowledge_gaps=scorer_output.knowledge_gaps if hasattr(scorer_output, 'knowledge_gaps') else (),
                acknowledgment=scorer_output.acknowledgment if hasattr(scorer_output, 'acknowledgment') else "",
            )

        # Fallback: basic heuristics
        wants_to_skip = any(kw in answer_lower for kw in self.SKIP_KEYWORDS)
        is_clarity = any(kw in answer_lower for kw in self.CLARITY_KEYWORDS)
        is_nonsense = answer_lower.strip() in ("haha", "lol", "idk", "whatever", "joking")

        is_detailed = word_count >= 30

        if wants_to_skip:
            quality = "skip_request"
        elif is_clarity:
            quality = "clarification_request"
        elif is_nonsense:
            quality = "nonsense"
        elif word_count < 10:
            quality = "vague"
        elif word_count < 25:
            quality = "partial"
        else:
            quality = "detailed"

        return AnswerAnalysis(
            word_count=word_count,
            is_detailed=is_detailed,
            wants_to_skip=wants_to_skip,
            is_nonsense=is_nonsense,
            is_asking_for_clarity=is_clarity,
            quality=quality,
            engagement="neutral",
            knowledge_level="uncertain",
            suggested_action="follow_up_same_topic" if word_count < 25 else "move_to_new_topic",
            question_focus="",
            knowledge_gaps=(),
            acknowledgment="",
        )

    # ------------------------------------------------------------------
    # Stateful decision
    # ------------------------------------------------------------------

    def should_follow_up(
        self,
        analysis: AnswerAnalysis,
    ) -> Tuple[bool, str]:
        """Decide follow-up vs move-on based on analysis."""

        if self.turn_count == 0:
            self.turn_count += 1
            return False, "opening_turn"

        self.turn_count += 1

        # Clarification request
        if analysis.is_asking_for_clarity:
            self._transition_topic()
            return False, "candidate_asked_for_clarity"

        # Skip request
        if analysis.wants_to_skip:
            self._transition_topic()
            return False, "candidate_requested_skip"

        # Evasive candidate — move on faster (don't insist)
        if analysis.engagement in ("evasive", "frustrated") and self.topic_depth >= 1:
            self._transition_topic()
            return False, f"evasive_moving_on (engagement={analysis.engagement})"

        # Nonsense
        if analysis.is_nonsense:
            self.consecutive_vague += 1
            if self.consecutive_vague >= self.CONSECUTIVE_VAGUE_LIMIT:
                self._transition_topic()
                return False, "too_many_vague_moving_on"
            self.topic_depth += 1
            return True, f"nonsense (consecutive={self.consecutive_vague})"

        # Max follow-ups
        if self.topic_depth >= self.MAX_FOLLOWUPS_PER_TOPIC:
            self._transition_topic()
            return False, f"max_followups_reached ({self.MAX_FOLLOWUPS_PER_TOPIC})"

        # Use analyzer's suggested action if available
        if analysis.suggested_action == "move_to_new_topic" and analysis.is_detailed:
            self._transition_topic()
            return False, f"detailed ({analysis.word_count}w, quality={analysis.quality})"

        # Vague or partial answers
        if analysis.quality in ("vague", "partial") or analysis.word_count < 15:
            self.topic_depth += 1
            self.consecutive_vague += 1
            return True, f"{analysis.quality} ({analysis.word_count}w)"

        # Default: trust the analyzer's action
        if analysis.suggested_action in ("follow_up_same_topic", "ask_for_example", "ask_simpler", "rephrase"):
            if self.topic_depth < 2:
                self.topic_depth += 1
                return True, f"analyzer_says_{analysis.suggested_action}"

        # Detailed enough
        self._transition_topic()
        return False, f"detailed ({analysis.word_count}w, quality={analysis.quality})"

    # ------------------------------------------------------------------
    # Planner-driven decision (LLM agent owns the choice; we apply state)
    # ------------------------------------------------------------------

    def apply_planner_decision(
        self,
        action: str,
        target_topic: Optional[str],
    ) -> Tuple[bool, str]:
        """Apply the InterviewPlanner's chosen action to mutable state.

        The planner LLM has already reasoned about turn count, topic
        depth, scorer signals, etc. — this method just performs the
        bookkeeping (topic_depth++, _transition_topic, queue advance) so
        the rest of the pipeline (`get_instruction`) keeps working.

        Returns `(should_follow_up, reason)` so callers can drop this in
        place of `should_follow_up()` without further changes.
        """
        # Opening turn is always handled deterministically — no analysis
        # exists yet. Planner won't have been called in that case.
        if self.turn_count == 0:
            self.turn_count += 1
            return False, "opening_turn"

        self.turn_count += 1

        if action == "clarify_previous":
            return True, "planner:clarify_previous"

        if action == "acknowledge_skip":
            self._transition_topic()
            self._advance_queue_to(target_topic)
            return False, "planner:acknowledge_skip"

        if action in ("move_to_new_topic", "wrap_topic_then_move"):
            self._transition_topic()
            self._advance_queue_to(target_topic)
            return False, f"planner:{action}"

        if action in ("follow_up_same_topic", "ask_for_example", "ask_simpler"):
            self.topic_depth += 1
            return True, f"planner:{action}"

        # Unknown action → safe fallback: treat as follow-up.
        self.topic_depth += 1
        return True, f"planner:unknown_action:{action}"

    def _advance_queue_to(self, target_topic: Optional[str]) -> None:
        """Pull the queue forward so the next `_get_next_topic()` call
        returns `target_topic`. If the planner picked a real topic we
        skip past anything ahead of it; otherwise we leave the queue
        alone and let `_get_next_topic()` pick the next one in order."""
        if not target_topic:
            return
        for i in range(self._topic_queue_index, len(self._topic_queue)):
            if self._topic_queue[i] == target_topic:
                self._topic_queue_index = i
                return

    def remaining_topics(self) -> List[str]:
        """Topics still in the queue and not already covered."""
        return [
            t for t in self._topic_queue[self._topic_queue_index:]
            if t not in self.topics_covered
        ]

    # ------------------------------------------------------------------
    # Instruction generation
    # ------------------------------------------------------------------

    def get_instruction(
        self,
        should_follow: bool,
        reason: str,
        analysis: AnswerAnalysis,
    ) -> Optional[str]:

        # Opening turn: handle explicitly so the model never gets None.
        # An empty instruction + empty history + a strongly-primed system
        # prompt drives small LLMs into transcript-header mode (e.g.
        # "**Date:** ... **Interviewer:** Alex"), which the leak detector
        # silences but the cleaner cannot recover from.
        if reason == "opening_turn":
            next_topic = self._get_next_topic()
            anchor = next_topic or "one of their CV projects"
            self.set_current_topic(next_topic) if next_topic else None
            return (
                "[OPENING TURN. Open with ONE short friendly sentence "
                "(e.g. \"Hi, great to meet you — thanks for joining.\") "
                f"then ask ONE specific, focused question about: {anchor}. "
                "Plain conversational prose only. NO headers, NO bold "
                "labels like **Date:**/**Location:**/**Interviewer:**, "
                "NO speaker tags, NO transcript framing. Max ~30 words.]"
            )

        parts = []

        # Add acknowledgment from analyzer, or provide a default one
        if analysis.acknowledgment:
            parts.append(f"React naturally to their answer, e.g.: \"{analysis.acknowledgment}\"")
        elif reason != "opening_turn":
            # Provide varied, natural acknowledgments based on answer quality
            if analysis.quality == "detailed":
                ack_options = [
                    "React to something specific they mentioned.",
                    "Briefly comment on their approach before asking.",
                    "Show you listened — reference a detail from their answer.",
                ]
            elif analysis.quality in ("vague", "partial"):
                ack_options = [
                    "Briefly acknowledge, then dig deeper on the same topic.",
                    "Show curiosity about what they said — ask them to elaborate.",
                ]
            elif analysis.wants_to_skip:
                ack_options = [
                    "No worries, smoothly transition to the next topic.",
                    "That's fine — move on naturally.",
                ]
            else:
                ack_options = [
                    "React naturally to their answer.",
                    "Briefly acknowledge what they said.",
                ]
            import random
            parts.append(random.choice(ack_options))

        # Main instruction
        if analysis.is_asking_for_clarity:
            next_topic = self._get_next_topic()
            if next_topic:
                parts.append(f"Ask a SPECIFIC question about: {next_topic}")
            else:
                parts.append("Ask about a specific project from the CV")

        elif analysis.wants_to_skip:
            next_topic = self._get_next_topic()
            if next_topic:
                parts.append(f"Ask about: {next_topic}")
            else:
                parts.append("Ask about a different CV project")

        elif reason == "too_many_vague_moving_on":
            next_topic = self._get_next_topic()
            if next_topic:
                parts.append(f"Move to new topic: {next_topic}")
            else:
                parts.append("Move to a different project from the CV")

        elif should_follow:
            if analysis.question_focus:
                parts.append(f"Follow up on SAME topic. Focus: {analysis.question_focus}")
            else:
                hint = self._FOLLOW_UP_HINTS[self.topic_depth % len(self._FOLLOW_UP_HINTS)]
                parts.append(f"Follow up on SAME topic. {hint}")

            if analysis.knowledge_gaps:
                gaps = ", ".join(analysis.knowledge_gaps[:2])
                parts.append(f"Probe: {gaps}")

        elif reason != "opening_turn":
            next_topic = self._get_next_topic()
            if next_topic:
                parts.append(f"Move to: {next_topic}")
            else:
                parts.append("Ask about a DIFFERENT project from the CV")

        if not parts:
            return None
        return "[" + " | ".join(parts) + "]"

    _FOLLOW_UP_HINTS: Tuple[str, ...] = (
        "Ask what THEY personally built or designed.",
        "Ask about a specific challenge or tricky part.",
        "Ask about concrete results or impact.",
        "Ask them to walk through their approach step by step.",
        "Ask why they chose that specific technology or method.",
        "Ask how they tested or validated their solution.",
    )

    # ------------------------------------------------------------------
    # Topic management
    # ------------------------------------------------------------------

    def _transition_topic(self) -> None:
        if self.current_topic and self.current_topic not in self.topics_covered:
            self.topics_covered.append(self.current_topic)
        self.current_topic = None
        self.topic_depth = 0
        self.consecutive_vague = 0

    def set_current_topic(self, topic: str) -> None:
        self.current_topic = topic

    # ------------------------------------------------------------------
    # Duplicate detection
    # ------------------------------------------------------------------

    def record_question(self, question: str) -> None:
        normalized = question.lower().strip().rstrip("?")
        self.questions_asked.add(normalized)
        self.question_word_sets.append(
            (normalized, frozenset(normalized.split()))
        )

    def is_duplicate_question(self, question: str) -> bool:
        normalized = question.lower().strip().rstrip("?")
        if normalized in self.questions_asked:
            return True
        q_words = frozenset(normalized.split())
        if not q_words:
            return False
        for _, prev_words in self.question_word_sets:
            if not prev_words:
                continue
            overlap = len(q_words & prev_words) / max(len(q_words), len(prev_words))
            if overlap > 0.75:
                return True
        return False

    # ------------------------------------------------------------------
    # Recovery
    # ------------------------------------------------------------------

    def get_recovery_question(self) -> str:
        next_topic = self._get_next_topic()
        if next_topic:
            templates = [
                f"Tell me about your work on {next_topic} — what was your specific contribution?",
                f"I see {next_topic} on your CV — can you walk me through your approach?",
                f"Regarding {next_topic} — what technical challenges did you face?",
            ]
            for q in templates:
                if not self.is_duplicate_question(q):
                    return q

        fallback = [
            "What's the most complex technical problem you've solved recently?",
            "How do you approach debugging when a model underperforms?",
            "What technical decision are you most proud of?",
        ]
        available = [q for q in fallback if not self.is_duplicate_question(q)]
        if available:
            return random.choice(available)
        return "What other technical experience would you like to discuss?"