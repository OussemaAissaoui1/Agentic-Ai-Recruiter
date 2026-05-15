"""
NLP Agent — v8  (streaming TTS pipeline)

Coordinates: ResponseScorer (CPU) + VLLMRecruiterEngine (GPU) + TTSEngine (CPU)

v8 changes vs v7:
  - generate_question() uses TTSEngine.synthesize_segmented() instead of
    synthesize() — splits completed text into sentences and synthesizes
    concurrently (faster for multi-sentence responses, same for single).

  - generate_question_streaming() now pipelines generation tokens directly
    into TTSEngine.synthesize_streaming().  As soon as the first sentence
    boundary is detected in the token stream, TTS fires on that chunk
    immediately — without waiting for the rest of generation to finish.
    Yields (text_chunk, TTSResult | None) pairs so the caller can display
    text AND play audio incrementally.

    Perceived latency improvement:
        Before (v7): generation_time + full_tts_time  (~0.75s)
        After  (v8): generation_time_to_first_sentence (~0.2s)
"""

from __future__ import annotations

import asyncio
import random
import time
import threading
from typing import List, Tuple, Optional, Dict, Any, AsyncGenerator
from dataclasses import dataclass

# Use package-relative imports when imported as agents.nlp.agent (unified app),
# bare-name imports when run as a script from the demo directory. We branch on
# __package__ instead of try/except so downstream ImportErrors (e.g. numpy ABI
# crashes in transformers) bubble up with their real traceback rather than
# being misreported as "module not found".
if __package__:
    from .vllm_engine import (
        VLLMRecruiterEngine, GenerationMetrics, check_gpu_status, cleanup_gpu_processes,
    )
    from .interview_state import InterviewStateTracker, AnswerAnalysis
    from .response_scorer import ResponseScorer, ScorerOutput
    from .question_refiner import QuestionRefiner, RefinerOutput
    from .interview_planner import InterviewPlanner, PlannerDecision, StateSnapshot
    from .tts_engine import TTSEngine, TTSResult, StreamingTTSResult
else:
    from vllm_engine import (  # type: ignore
        VLLMRecruiterEngine, GenerationMetrics, check_gpu_status, cleanup_gpu_processes,
    )
    from interview_state import InterviewStateTracker, AnswerAnalysis  # type: ignore
    from response_scorer import ResponseScorer, ScorerOutput  # type: ignore
    from question_refiner import QuestionRefiner, RefinerOutput  # type: ignore
    from interview_planner import InterviewPlanner, PlannerDecision, StateSnapshot  # type: ignore
    from tts_engine import TTSEngine, TTSResult, StreamingTTSResult  # type: ignore

# Module-level lock for agent engine initialization
_AGENT_ENGINE_LOCK = threading.Lock()


@dataclass(frozen=True)
class QuestionGenerationRequest:
    candidate_cv: str
    job_role: str
    conversation_history: List[Tuple[str, str]]
    candidate_latest_answer: str
    session_id: str


@dataclass
class QuestionGenerationResponse:
    question: str
    reasoning: str
    topic_depth: int
    turn_count: int
    metrics: GenerationMetrics
    is_follow_up: bool
    scorer_output: Optional[Dict[str, Any]] = None
    refiner_output: Optional[Dict[str, Any]] = None  # NEW: refinement info
    planner_output: Optional[Dict[str, Any]] = None  # NEW: agent reasoning for this turn
    # v8: audio is now a StreamingTTSResult (has .to_wav_bytes() and .chunks)
    # Backwards-compatible: single-sentence responses have exactly one chunk
    # and StreamingTTSResult.to_wav_bytes() works the same as TTSResult.to_wav_bytes()
    audio: Optional[StreamingTTSResult] = None


@dataclass
class StreamingQuestionChunk:
    """
    One yielded item from generate_question_streaming_with_audio().

    text_delta: new text tokens (append to display buffer)
    tts_result: synthesized audio for a completed sentence, or None if
                this chunk is a mid-sentence token update
    is_final:   True on the last chunk — signals end of stream
    """
    text_delta: str
    tts_result: Optional[TTSResult] = None
    is_final: bool = False


class SystemPromptBuilder:

    @staticmethod
    def extract_topics_from_cv(cv_text: str) -> List[str]:
        topics: List[str] = []
        current_section: Optional[str] = None
        for line in cv_text.strip().split("\n"):
            ls = line.strip()
            ll = ls.lower()
            if any(k in ll for k in ("experience", "projects", "work", "internship")) and ":" in ls:
                current_section = ll
                continue
            if current_section and ls.startswith("-"):
                entry = ls.lstrip("-").strip()
                if len(entry.split()) < 4:
                    continue
                ci = entry.find(":")
                if 0 < ci < 80:
                    entry = entry[:ci].strip()
                if entry and entry not in topics:
                    topics.append(entry)
        if not topics:
            topics = ["Their most impactful project", "A technical challenge they overcame"]
        return topics[:8]

    @staticmethod
    def build_prompt(candidate_cv: str, job_role: str) -> str:
        topics = SystemPromptBuilder.extract_topics_from_cv(candidate_cv)
        random.shuffle(topics)
        tl = "\n".join(f"- {t}" for t in topics)
        return f"""You are Alex, a warm and experienced senior technical recruiter conducting a live interview for the {job_role} position. You genuinely enjoy getting to know candidates and have a knack for making people feel comfortable while still asking sharp, insightful questions.

## IDENTITY (NEVER CHANGE):
- Your name is ALWAYS Alex. You introduced yourself as Alex. Never use any other name.
- If the candidate asks who you are, say: "I'm Alex, your interviewer."
- Never say "I'm Dr. Khan", "I'm Dr.", or any other name.

## Candidate CV:
{candidate_cv}

## Your Personality:
- Warm, curious, and encouraging — like a senior colleague having a coffee chat
- You listen carefully and react naturally to what the candidate says
- You use casual but professional language ("That's really cool", "Oh nice", "Ah, that makes sense")
- You occasionally share brief relatable observations ("I've seen teams struggle with that")
- You never sound like you're reading from a script

## Conversation Flow:
1. LISTEN to what the candidate just said — show you actually heard them
2. React naturally with a brief, genuine acknowledgment that references their specific answer
3. Then transition smoothly into your next question about a CV topic

## Handling Difficult Candidates:
- If the candidate is evasive, rude, or gives one-word answers: stay calm and professional
- Gently redirect to a specific, easy question about something concrete in their CV
- NEVER lecture them about communication skills or professionalism
- NEVER express frustration, disappointment, or concern
- NEVER end the interview early or say "unfortunately"
- Just keep asking good questions — let their answers speak for themselves
- If they say "I don't know", move to a simpler or different topic naturally

## Acknowledgment Style (IMPORTANT — vary these, never repeat the same one):
Instead of robotic "Got it." or "I see.", use natural reactions like:
- "Oh, that's a smart approach — using confidence heuristics as a safety net."
- "SAM for boundary detection, nice. That's not trivial to get right."
- "Okay so you went with a graph-based approach — makes sense for that kind of data."
- "Huh, 30+ screens is a lot to automate. That sounds intense."
- Reference something SPECIFIC from their answer, not generic filler.

## ABSOLUTE RULES — NEVER BREAK THESE:
1. Output ONLY the next thing Alex says out loud, in first person. One short turn. Nothing else.
2. NEVER simulate a transcript. NEVER roleplay both sides. NEVER write the candidate's reply. NEVER write lines like "Candidate:", "Chedly:", "Alex:", "Interviewer:", "Me:", or any speaker label.
3. NEVER write a narration, summary, framing or preamble like "This transcript simulates...", "The following is an interview between...", "In this conversation...", "Below is...".
4. NEVER use markdown headers, bold titles, or section dividers (no **Interview Transcript**, no ##, no --- lines, no bullet lists, no numbered lists). Output is plain conversational prose only.
5. NEVER write meta-commentary like "Here's how I'd respond:", "Here's my response:", "I would say:".
6. NEVER write stage directions like [Looks concerned], [Pauses], [Smiles], (warmly), *nods*.
7. NEVER write internal notes like "Post-interview notes", "Assessment:", "Evaluation:", "Score:".
8. NEVER write "The candidate...", "Based on this conversation...", or any third-person reference to the candidate. Always speak directly to them in second person ("you", "your").
9. NEVER write placeholder brackets like [specific language], [topic], [project name].
10. NEVER end the interview or say goodbye. NEVER say "thanks for coming in", "final question", "to wrap up", "that's all". Keep asking questions.
11. NEVER ask about topics NOT on the CV. Only ask about what they actually worked on.
12. ONE question only per turn. Not two, not three. One.
13. Max ~30 words total. Aim for 1–2 short conversational sentences. Be tight.

## OUTPUT SHAPE — exactly one of these forms, no decoration:
✓ "Oh nice, vector search at that scale is tricky. How did you handle the latency budget on the retrieval side?"
✓ "Interesting choice using SAM there. What made you go with that over a more classical segmentation pipeline?"
✗ "**Interview Transcript**\n\nAlex: Hi Chedly, thanks for coming in..."   ← forbidden, you are simulating the whole interview
✗ "This transcript simulates an interview..."                                ← forbidden, narration
✗ "Alex: That's cool. How did you..."                                        ← forbidden, speaker label
✗ "Here's my response: That's a great answer..."                             ← forbidden, meta

## What to AVOID:
- Topics NOT mentioned in the CV — stick to what they actually worked on
- Evaluating or grading ("great job", "impressive work", "excellent")
- Ending the interview ("thank you for your time", "final question")
- Generic HR questions ("describe a time when", "what's your biggest weakness")
- Asking them to write code
- Repeating a question you already asked
- Ignoring what they just said and jumping to an unrelated topic
- Long multi-part questions — keep it to ONE clear question

## Format:
- 1-2 sentences: natural acknowledgment + smooth transition
- Then ONE focused question ending with "?"
- Max ~30 words total. Be tight and conversational.

## Topics from their CV to explore:
{tl}
"""

    @staticmethod
    def build_compact_prompt(job_role: str) -> str:
        return " "


class NLPAgent:

    SESSION_TTL_SEC: int = 7200

    def __init__(
        self,
        model_path: str = "/teamspace/studios/this_studio/Agentic-Ai-Recruiter/model_cache",
        dtype: str = "bfloat16",
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.60,
        tensor_parallel_size: int = 0,
        use_compact_prompt: bool = False,
        enable_scorer: bool = False,
        scorer_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        enable_refiner: bool = True,  # NEW: enable question refinement
        refiner_model: str = "Qwen/Qwen2.5-1.5B-Instruct",  # NEW: refiner model
        enable_planner: bool = True,  # NEW: LLM-driven planner agent owns flow control
        planner_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        planner_share_scorer_model: bool = True,  # reuse scorer weights to save VRAM
        enable_tts: bool = True,
        tts_voice: str = "af_heart",
        auto_cleanup_gpu: bool = True,
        load_timeout_sec: int = 180,
    ) -> None:

        self._model_path = model_path
        self._dtype = dtype
        self._max_model_len = max_model_len
        self._gpu_mem_util = gpu_memory_utilization
        self._tp_size = tensor_parallel_size
        self._auto_cleanup = auto_cleanup_gpu
        self._load_timeout = load_timeout_sec

        self._engine: Optional[VLLMRecruiterEngine] = None
        self._engine_init_error: Optional[str] = None

        self._scorer: Optional[ResponseScorer] = None
        self._refiner: Optional[QuestionRefiner] = None  # NEW
        self._planner: Optional[InterviewPlanner] = None  # NEW: agentic flow control
        self._tts: Optional[TTSEngine] = None
        self._enable_scorer = enable_scorer
        self._scorer_model = scorer_model
        self._enable_refiner = enable_refiner  # NEW
        self._refiner_model = refiner_model  # NEW
        self._enable_planner = enable_planner  # NEW
        self._planner_model = planner_model  # NEW
        self._planner_share_scorer = planner_share_scorer_model  # NEW
        self._enable_tts = enable_tts
        self._tts_voice = tts_voice
        self._scorer_loaded = False
        self._refiner_loaded = False  # NEW
        self._planner_loaded = False  # NEW
        self._tts_loaded = False

        self._sessions: Dict[str, InterviewStateTracker] = {}
        self._session_prompts: Dict[str, str] = {}
        self._session_locks: Dict[str, asyncio.Lock] = {}
        self._session_timestamps: Dict[str, float] = {}
        self._use_compact = use_compact_prompt
        self._warmup_done = False

    def _ensure_models_loaded(self) -> None:
        if not self._scorer_loaded and self._enable_scorer:
            try:
                print("[agent] Loading scorer model on GPU...")
                self._scorer = ResponseScorer(model_name=self._scorer_model, device="cuda")
            except Exception as e:
                print(f"[agent] Scorer GPU failed, falling back to CPU: {e}")
                try:
                    self._scorer = ResponseScorer(model_name=self._scorer_model, device="cpu")
                except Exception as e2:
                    print(f"[agent] Scorer init warning: {e2}")
            self._scorer_loaded = True

        if not self._refiner_loaded and self._enable_refiner:
            try:
                print("[agent] Loading refiner model on GPU...")
                self._refiner = QuestionRefiner(model_name=self._refiner_model, device="cuda")
                self._refiner._lazy_load()  # Eagerly load so is_ready=True
            except Exception as e:
                print(f"[agent] Refiner GPU failed, falling back to CPU: {e}")
                try:
                    self._refiner = QuestionRefiner(model_name=self._refiner_model, device="cpu")
                    self._refiner._lazy_load()
                except Exception as e2:
                    print(f"[agent] Refiner init warning: {e2}")
            self._refiner_loaded = True

        if not self._planner_loaded and self._enable_planner:
            try:
                # If we already loaded the scorer model and they're the
                # same architecture, share the weights to avoid eating
                # another ~3 GB of CPU/VRAM. The planner's own LLM call
                # is independent (different prompt, different generation
                # params) — only the underlying tokenizer + weights are
                # reused.
                shared = (
                    self._planner_share_scorer
                    and self._scorer is not None
                    and self._scorer.is_ready
                    and self._planner_model == self._scorer_model
                )
                if shared:
                    print("[agent] Sharing scorer weights with planner...")
                    self._planner = InterviewPlanner(autoload=False)
                    self._planner.attach_shared_model(
                        tokenizer=self._scorer._tokenizer,
                        model=self._scorer._model,
                        device=self._scorer.device,
                    )
                else:
                    print("[agent] Loading planner model on GPU...")
                    self._planner = InterviewPlanner(
                        model_name=self._planner_model, device="cuda",
                    )
                    if not self._planner.is_ready:
                        print("[agent] Planner GPU failed, falling back to CPU")
                        self._planner = InterviewPlanner(
                            model_name=self._planner_model, device="cpu",
                        )
            except Exception as e:
                print(f"[agent] Planner init warning: {e}")
                self._planner = None
            self._planner_loaded = True

        if not self._tts_loaded and self._enable_tts:
            try:
                print("[agent] Loading TTS model on CPU...")
                self._tts = TTSEngine(voice=self._tts_voice)
            except Exception as e:
                print(f"[agent] TTS init warning: {e}")
            self._tts_loaded = True

    @property
    def engine(self) -> VLLMRecruiterEngine:
        if self._engine is None:
            with _AGENT_ENGINE_LOCK:
                if self._engine is None:
                    if self._engine_init_error:
                        raise RuntimeError(self._engine_init_error)
                    try:
                        print("[agent] Initializing vLLM engine (lazy load)...")
                        self._engine = VLLMRecruiterEngine(
                            model_path=self._model_path,
                            dtype=self._dtype,
                            max_model_len=self._max_model_len,
                            gpu_memory_utilization=self._gpu_mem_util,
                            tensor_parallel_size=self._tp_size,
                            auto_cleanup=self._auto_cleanup,
                            load_timeout_sec=self._load_timeout,
                        )
                    except Exception as e:
                        self._engine_init_error = str(e)
                        raise
        return self._engine

    @property
    def is_ready(self) -> bool:
        try:
            return self._engine is not None and self._engine.is_ready
        except Exception:
            return False

    def get_status(self) -> Dict[str, Any]:
        gpu_status = check_gpu_status()
        return {
            "engine_ready": self.is_ready,
            "engine_error": self._engine_init_error,
            "scorer_ready": self._scorer.is_ready if self._scorer else False,
            "scorer_loaded": self._scorer_loaded,
            "refiner_ready": self._refiner.is_ready if self._refiner else False,
            "refiner_loaded": self._refiner_loaded,
            "planner_ready": self._planner.is_ready if self._planner else False,
            "planner_loaded": self._planner_loaded,
            "tts_ready": self._tts.is_ready if self._tts else False,
            "tts_loaded": self._tts_loaded,
            "gpu_available": gpu_status.available,
            "gpu_free_gb": gpu_status.free_memory_gb,
            "gpu_total_gb": gpu_status.total_memory_gb,
            "gpu_blocking_pids": gpu_status.blocking_pids,
            "active_sessions": len(self._sessions),
        }

    async def warmup(self, job_role: str = "AI Engineering Intern") -> None:
        if self._warmup_done:
            return
        sp = SystemPromptBuilder.build_compact_prompt(job_role)
        await self.engine.warmup(sp)
        self._ensure_models_loaded()
        self._warmup_done = True

    def _get_lock(self, sid: str) -> asyncio.Lock:
        if sid not in self._session_locks:
            self._session_locks[sid] = asyncio.Lock()
        return self._session_locks[sid]

    def _get_or_create_session(self, sid: str, cv: str = "") -> InterviewStateTracker:
        self._session_timestamps[sid] = time.monotonic()
        if sid not in self._sessions:
            t = InterviewStateTracker()
            if cv:
                topics = SystemPromptBuilder.extract_topics_from_cv(cv)
                random.shuffle(topics)
                t.set_topic_queue(topics)
            self._sessions[sid] = t
        return self._sessions[sid]

    def _get_or_create_prompt(self, sid: str, cv: str, role: str) -> str:
        if sid not in self._session_prompts:
            if self._use_compact:
                p = SystemPromptBuilder.build_compact_prompt(role)
            else:
                p = SystemPromptBuilder.build_prompt(cv, role)
            self._session_prompts[sid] = p
        return self._session_prompts[sid]

    def reset_session(self, sid: str) -> None:
        if sid in self._sessions:
            self._sessions[sid].reset()

    def end_session(self, sid: str) -> None:
        self._sessions.pop(sid, None)
        self._session_prompts.pop(sid, None)
        self._session_locks.pop(sid, None)
        self._session_timestamps.pop(sid, None)

    def cleanup_stale_sessions(self) -> int:
        now = time.monotonic()
        stale = [s for s, t in self._session_timestamps.items() if now - t > self.SESSION_TTL_SEC]
        for s in stale:
            self.end_session(s)
        return len(stale)

    # ------------------------------------------------------------------
    # Agentic flow control: planner LLM owns the decision
    # ------------------------------------------------------------------

    async def _decide_next_action(
        self,
        state: InterviewStateTracker,
        analysis: AnswerAnalysis,
        scorer_output: Optional[ScorerOutput],
        conversation_history: List[Tuple[str, str]],
        latest_answer: str,
    ) -> Tuple[bool, str, Optional["PlannerDecision"]]:
        """Decide follow-up vs new-topic for this turn.

        Strategy:
          1. Skip the planner on the opening turn — there is nothing to
             reason over yet.
          2. Otherwise call the planner agent. The planner is the
             primary decision-maker; the rule engine is the fallback.
          3. If the planner is unavailable, errors out, or its output
             can't be used, fall back to `state.should_follow_up()` so
             the interview never wedges.

        Returns `(should_follow_up, reason, planner_decision_or_None)`.
        The decision is propagated up so it can be logged and shipped
        to the frontend for transparency.
        """
        if state.turn_count == 0 or not (self._enable_planner and self._planner and self._planner.is_ready):
            should_follow, reason = state.should_follow_up(analysis)
            return should_follow, reason, None

        scorer_dict: Dict[str, Any] = (
            scorer_output.to_dict() if scorer_output and hasattr(scorer_output, "to_dict") else {}
        )
        # to_dict() doesn't include knowledge_gaps if missing — fold in
        # the structured fields the planner expects.
        scorer_dict.setdefault("quality", analysis.quality)
        scorer_dict.setdefault("engagement", analysis.engagement)
        scorer_dict.setdefault("knowledge_level", analysis.knowledge_level)
        scorer_dict.setdefault("question_focus", analysis.question_focus)
        scorer_dict.setdefault("knowledge_gaps", analysis.knowledge_gaps)
        scorer_dict.setdefault("suggested_action", analysis.suggested_action)

        snapshot = StateSnapshot(
            turn_count=state.turn_count,
            current_topic=state.current_topic,
            topic_depth=state.topic_depth,
            topics_covered=list(state.topics_covered),
            topics_remaining=state.remaining_topics(),
            consecutive_vague=state.consecutive_vague,
            max_followups_per_topic=state.MAX_FOLLOWUPS_PER_TOPIC,
        )

        try:
            decision = await self._planner.plan(
                scorer_analysis=scorer_dict,
                state=snapshot,
                latest_answer=latest_answer,
                conversation_history=conversation_history,
            )
        except Exception as e:
            print(f"   ✗ Planner error: {e} — falling back to rule engine")
            should_follow, reason = state.should_follow_up(analysis)
            return should_follow, reason, None

        # If the planner returned the default ("planner_unavailable_default")
        # it means it couldn't reason — defer to the rule engine.
        if decision.reasoning == "planner_unavailable_default":
            should_follow, reason = state.should_follow_up(analysis)
            return should_follow, reason, None

        should_follow, reason = state.apply_planner_decision(
            action=decision.action,
            target_topic=decision.target_topic,
        )
        return should_follow, reason, decision

    # ------------------------------------------------------------------
    # generate_question — non-streaming path (uses segmented TTS)
    # ------------------------------------------------------------------

    async def generate_question(
        self,
        request: QuestionGenerationRequest,
        max_tokens: int = 48,
        temperature: float = 0.2,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
        history_window: int = 20,
        max_retries: int = 3,
        synthesize_audio: bool = True,
    ) -> QuestionGenerationResponse:

        self._ensure_models_loaded()

        lock = self._get_lock(request.session_id)
        async with lock:

            state = self._get_or_create_session(request.session_id, request.candidate_cv)
            sys_prompt = self._get_or_create_prompt(
                request.session_id, request.candidate_cv, request.job_role
            )

            # Run scorer and wait for it (fast on CPU, ~300-500ms)
            print("\n" + "="*70)
            print("📊 PIPELINE LOG - Session:", request.session_id[:8])
            print("="*70)
            
            scorer_output: Optional[ScorerOutput] = None
            if self._scorer and self._scorer.is_ready:
                try:
                    print("\n🧠 SCORER - Analyzing candidate answer...")
                    scorer_output = await self._scorer.score(
                        request.candidate_cv,
                        request.conversation_history,
                        request.candidate_latest_answer,
                    )
                    
                    # Log scorer analysis
                    if scorer_output:
                        print(f"   ✓ Scorer completed in {scorer_output.latency_ms:.0f}ms")
                        print(f"   • Answer Quality: {scorer_output.answer_quality}")
                        print(f"   • Engagement: {scorer_output.candidate_engagement}")
                        print(f"   • Knowledge Level: {scorer_output.knowledge_level}")
                        print(f"   • Suggested Action: {scorer_output.suggested_action}")
                        if scorer_output.question_focus:
                            print(f"   • Focus Area: {scorer_output.question_focus}")
                        if scorer_output.knowledge_gaps:
                            print(f"   • Knowledge Gaps: {', '.join(scorer_output.knowledge_gaps[:2])}")
                        if scorer_output.acknowledgment:
                            print(f"   • Acknowledgment: \"{scorer_output.acknowledgment}\"")
                except Exception as e:
                    print(f"   ✗ Scorer error: {e}")
                    scorer_output = None
            else:
                print("\n🧠 SCORER - Disabled or not ready, using fallback heuristics")

            # Build analysis using scorer output
            analysis = state.build_analysis(request.candidate_latest_answer, scorer_output)
            should_follow, reason, planner_decision = await self._decide_next_action(
                state=state,
                analysis=analysis,
                scorer_output=scorer_output,
                conversation_history=request.conversation_history,
                latest_answer=request.candidate_latest_answer,
            )
            instruction = state.get_instruction(should_follow, reason, analysis)

            if planner_decision is not None:
                print(f"\n🧭 PLANNER (agent) - chose action in {planner_decision.latency_ms:.0f}ms")
                print(f"   • Action: {planner_decision.action} (confidence={planner_decision.confidence:.2f})")
                if planner_decision.target_topic:
                    print(f"   • Target topic: {planner_decision.target_topic}")
                if planner_decision.reasoning:
                    snippet = planner_decision.reasoning[:160]
                    print(f"   • Reasoning: {snippet}{'...' if len(planner_decision.reasoning) > 160 else ''}")
            print(f"\n📋 DECISION - {reason}")
            print(f"   • Follow-up: {'Yes' if should_follow else 'No (move to new topic)'}")
            if instruction:
                print(f"   • Instruction: {instruction[:100]}{'...' if len(instruction) > 100 else ''}")

            # Generate with retries
            print("\n🤖 MAIN LLM - Generating question...")
            question = ""
            metrics = None
            for attempt in range(max_retries + 1):
                question, metrics = await self.engine.generate(
                    system_prompt=sys_prompt,
                    history=request.conversation_history,
                    candidate_input=request.candidate_latest_answer,
                    instruction=instruction,
                    max_tokens=max_tokens,
                    temperature=temperature + (attempt * 0.1),
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    history_window=history_window,
                )
                is_bad = (
                    not question or len(question.strip()) < 10
                    or self.engine.is_termination_attempt(question)
                    or state.is_duplicate_question(question)
                    or self.engine.is_generic_question(question)
                )
                if not is_bad:
                    break
                if attempt < max_retries:
                    print(f"   ⚠ Retry {attempt + 1}/{max_retries}: Bad output, regenerating...")

            if (not question or len(question.strip()) < 10
                    or self.engine.is_termination_attempt(question)
                    or state.is_duplicate_question(question)
                    or self.engine.is_generic_question(question)):
                print("   ⚠ All attempts failed, using recovery question")
                question = state.get_recovery_question()
                reason = "recovery"
            
            # Log original question
            if metrics:
                print(f"   ✓ Generated in {metrics.latency_sec*1000:.0f}ms ({metrics.tokens_per_sec:.1f} tok/s)")
            print(f"\n   📝 Original Question:")
            print(f"   \"{question}\"")

            state.record_question(question)

            # Refine the question using the small LLM + scorer analysis
            refiner_output: Optional[RefinerOutput] = None
            original_question = question
            if self._refiner and analysis:
                try:
                    print("\n✨ REFINER - Enhancing question depth & specificity...")
                    refiner_output = await self._refiner.refine(
                        original_question=question,
                        cv_topic=analysis.question_focus or state.current_topic or "CV items",
                        knowledge_level=analysis.knowledge_level,
                        response_quality=analysis.quality,
                        question_focus=analysis.question_focus,
                        acknowledgment=analysis.acknowledgment,
                        timeout_sec=2.5,  # Fast refinement
                    )
                    
                    # Log refinement results
                    if refiner_output:
                        print(f"   ✓ Refinement completed in {refiner_output.latency_ms:.0f}ms")
                        print(f"   • Type: {refiner_output.refinement_type}")
                        print(f"   • Improvements: {refiner_output.improvements}")
                        
                        # Use refined question if refinement succeeded
                        if refiner_output.refinement_type != "maintained":
                            question = refiner_output.refined_question
                            state.record_question(question)  # Update recorded question
                            print(f"\n   ✏️  Refined Question:")
                            print(f"   \"{question}\"")
                        else:
                            print(f"   ℹ️  Original question maintained (no changes)")
                except Exception as e:
                    print(f"   ✗ Refiner error: {e}")
                    refiner_output = None
            else:
                print("\n✨ REFINER - Disabled or not ready, using original question")

            # v8: use synthesize_segmented instead of synthesize
            # For single-sentence questions (the common case) this is
            # identical in latency. For multi-sentence it runs concurrently.
            audio_result: Optional[StreamingTTSResult] = None
            if synthesize_audio and self._tts and self._tts.is_ready:
                print("\n🔊 TTS - Synthesizing audio...")
                audio_result = await self._tts.synthesize_segmented(question)
                if audio_result:
                    print(f"   ✓ Audio synthesized ({len(audio_result.chunks)} segment(s))")
            
            print("\n" + "="*70)
            print(f"✅ FINAL QUESTION: \"{question}\"")
            print("="*70 + "\n")

            return QuestionGenerationResponse(
                question=question,
                reasoning=reason,
                topic_depth=state.topic_depth,
                turn_count=state.turn_count,
                metrics=metrics,
                is_follow_up=should_follow,
                scorer_output=scorer_output.to_dict() if scorer_output and hasattr(scorer_output, "to_dict") else None,
                refiner_output=refiner_output.to_dict() if refiner_output else None,
                planner_output=planner_decision.to_dict() if planner_decision else None,
                audio=audio_result,
            )

    # ------------------------------------------------------------------
    # generate_question_streaming — text only (backwards-compatible)
    # ------------------------------------------------------------------

    async def generate_question_streaming(
        self,
        request: QuestionGenerationRequest,
        max_tokens: int = 48,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
        history_window: int = 6,
    ) -> AsyncGenerator[str, None]:
        """
        Stream question tokens as they're generated.
        Yields plain text chunks. Backwards-compatible with v7.
        For the full streaming + audio pipeline use
        generate_question_streaming_with_audio().
        """
        self._ensure_models_loaded()

        state = self._get_or_create_session(request.session_id, request.candidate_cv)
        sys_prompt = self._get_or_create_prompt(
            request.session_id, request.candidate_cv, request.job_role
        )

        # Run scorer if available so the planner has signal to reason over.
        # The text-only path historically skipped the scorer to save latency;
        # we re-enable it when the scorer is loaded so the planner is useful.
        scorer_output: Optional[ScorerOutput] = None
        if (
            self._enable_scorer and self._scorer and self._scorer.is_ready
            and request.candidate_latest_answer and request.candidate_latest_answer.strip()
        ):
            try:
                scorer_output = await self._scorer.score(
                    request.candidate_cv,
                    request.conversation_history,
                    request.candidate_latest_answer,
                )
            except Exception as e:
                print(f"[agent] scorer error in text-only path: {e}")
                scorer_output = None

        analysis = state.build_analysis(request.candidate_latest_answer, scorer_output)
        should_follow, reason, _planner_decision = await self._decide_next_action(
            state=state,
            analysis=analysis,
            scorer_output=scorer_output,
            conversation_history=request.conversation_history,
            latest_answer=request.candidate_latest_answer,
        )
        instruction = state.get_instruction(should_follow, reason, analysis)

        # Same head-window gate as generate_question_streaming_with_audio:
        # buffer until classification, then either flush head + passthrough or
        # hold silently and emit the cleaned single chunk at end.
        streamed_chunks: List[str] = []
        head_buf: List[str] = []
        classified = False
        is_leak = False
        HEAD_DECIDE_CHARS = 50

        async for token in self.engine.generate_streaming(
            system_prompt=sys_prompt,
            history=request.conversation_history,
            candidate_input=request.candidate_latest_answer,
            instruction=instruction,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            history_window=history_window,
        ):
            streamed_chunks.append(token)
            if not classified:
                head_buf.append(token)
                joined = "".join(head_buf)
                if len(joined.strip()) >= HEAD_DECIDE_CHARS or self.engine.looks_like_leak_head(joined):
                    is_leak = self.engine.looks_like_leak_head(joined)
                    classified = True
                    if not is_leak:
                        yield joined
                continue
            if is_leak:
                continue
            yield token

        final_question = self.engine._clean_response("".join(streamed_chunks))
        # If the stream was held silently (leak mode) or ended pre-classification,
        # emit the cleaned text now as a single delta so the caller sees something.
        if not classified or is_leak:
            if final_question:
                yield final_question
        if (
            not final_question or len(final_question.strip()) < 10
            or self.engine.is_termination_attempt(final_question)
            or state.is_duplicate_question(final_question)
            or self.engine.is_generic_question(final_question)
        ):
            final_question = state.get_recovery_question()

        state.record_question(final_question)

    # ------------------------------------------------------------------
    # NEW: generate_question_streaming_with_audio
    # Pipelines generation tokens directly into sentence-boundary TTS.
    # ------------------------------------------------------------------

    async def generate_question_streaming_with_audio(
        self,
        request: QuestionGenerationRequest,
        max_tokens: int = 48,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
        history_window: int = 6,
    ) -> AsyncGenerator[StreamingQuestionChunk, None]:
        """
        Full streaming pipeline: generation tokens → sentence-boundary TTS.

        Yields StreamingQuestionChunk objects:
          - text_delta:  new text to append to the display buffer
          - tts_result:  TTSResult when a sentence is ready to play, else None
          - is_final:    True on the last chunk

        How it works:
            1. vLLM streams tokens via generate_streaming()
            2. Tokens are teed: one copy goes to the display (yielded as
               text_delta), the other feeds TTSEngine.synthesize_streaming()
            3. synthesize_streaming() buffers tokens until a sentence boundary
               (.?!) then fires Kokoro synthesis on that chunk immediately,
               without waiting for the rest of generation
            4. When a TTSResult is ready it's yielded with tts_result set,
               so the caller can play audio while text keeps streaming in

        Caller in app.py should:
            - Append text_delta to a display buffer and call placeholder.markdown()
            - When tts_result is not None, queue the WAV bytes for playback
        """
        if not self._tts or not self._tts.is_ready:
            # TTS not available — fall back to text-only streaming
            async for token in self.generate_question_streaming(
                request, max_tokens, temperature, top_p, repetition_penalty, history_window
            ):
                yield StreamingQuestionChunk(text_delta=token)
            yield StreamingQuestionChunk(text_delta="", is_final=True)
            return

        self._ensure_models_loaded()

        state = self._get_or_create_session(request.session_id, request.candidate_cv)
        sys_prompt = self._get_or_create_prompt(
            request.session_id, request.candidate_cv, request.job_role
        )

        # Log pipeline start
        print("\n" + "="*70)
        print(f"📊 STREAMING PIPELINE - Session: {request.session_id[:12]}")
        print("="*70)

        # Run scorer if enabled and we have an answer to score
        scorer_output = None
        if self._enable_scorer and self._scorer and self._scorer.is_ready and request.candidate_latest_answer and request.candidate_latest_answer.strip():
            print("\n🧠 SCORER - Analyzing candidate answer...")
            try:
                scorer_start = time.time()
                scorer_output = await self._scorer.score(
                    request.candidate_cv,
                    request.conversation_history,
                    request.candidate_latest_answer,
                )
                scorer_latency_ms = (time.time() - scorer_start) * 1000
                
                print(f"   ✓ Scorer completed in {scorer_latency_ms:.0f}ms")
                print(f"   • Answer Quality: {scorer_output.answer_quality}")
                print(f"   • Engagement: {scorer_output.candidate_engagement}")
                print(f"   • Knowledge Level: {scorer_output.knowledge_level}")
                print(f"   • Suggested Action: {scorer_output.suggested_action}")
                if scorer_output.acknowledgment:
                    print(f"   • Acknowledgment: \"{scorer_output.acknowledgment}\"")
            except Exception as e:
                print(f"   ✗ Scorer error: {e}")
                scorer_output = None
        else:
            print("\n🧠 SCORER - Skipped (no answer or disabled)")

        # Build analysis with scorer output
        analysis = state.build_analysis(request.candidate_latest_answer, scorer_output)
        should_follow, reason, planner_decision = await self._decide_next_action(
            state=state,
            analysis=analysis,
            scorer_output=scorer_output,
            conversation_history=request.conversation_history,
            latest_answer=request.candidate_latest_answer,
        )

        if planner_decision is not None:
            print(f"\n🧭 PLANNER (agent) - chose action in {planner_decision.latency_ms:.0f}ms")
            print(f"   • Action: {planner_decision.action} (confidence={planner_decision.confidence:.2f})")
            if planner_decision.target_topic:
                print(f"   • Target topic: {planner_decision.target_topic}")
            if planner_decision.reasoning:
                snippet = planner_decision.reasoning[:160]
                print(f"   • Reasoning: {snippet}{'...' if len(planner_decision.reasoning) > 160 else ''}")

        # Log decision
        print(f"\n📋 DECISION - {'Follow up' if should_follow else 'Move to new topic'}")
        print(f"   • Reason: {reason}")

        instruction = state.get_instruction(should_follow, reason, analysis)
        
        print(f"\n🤖 MAIN LLM - Generating question (streaming)...")
        if instruction:
            print(f"   • Instruction: {instruction[:100]}{'...' if len(instruction) > 100 else ''}")
        else:
            print(f"   • Instruction: (none — opening turn)")

        # We tee the model output to two sinks: the SSE display (`all_tokens`)
        # and the TTS pipeline (`token_queue`). Both consume the SAME stream so
        # we sanitize once at the source.
        #
        # The producer runs a small head-window classifier: it buffers tokens
        # silently until it has enough text to decide whether the model is
        # behaving (clean conversational reply) or has slipped into transcript
        # roleplay / narration mode. On clean, the head buffer is flushed and
        # subsequent tokens passthrough live. On leak, no tokens are forwarded
        # at all — the producer collects the full output, runs `_clean_response`
        # once generation finishes, and emits the recovered first-utterance as
        # a single chunk to both sinks.
        #
        # Net effect: the user never sees or hears `**Interview Transcript**`,
        # `Alex:`, narration paragraphs, or candidate-side simulated lines.

        token_queue: asyncio.Queue = asyncio.Queue()
        all_tokens: List[str] = []
        raw_tokens_for_state: List[str] = []  # untouched buffer for record_question

        _END = object()
        HEAD_DECIDE_CHARS = 50  # buffer this much before classifying

        async def _token_producer() -> None:
            head_buf: List[str] = []
            classified = False
            is_leak = False

            def _emit_clean(text: str) -> None:
                if not text:
                    return
                all_tokens.append(text)
                # token_queue feeds TTS; can't await here in a sync helper,
                # caller must do it.

            try:
                async for token in self.engine.generate_streaming(
                    system_prompt=sys_prompt,
                    history=request.conversation_history,
                    candidate_input=request.candidate_latest_answer,
                    instruction=instruction,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    history_window=history_window,
                ):
                    raw_tokens_for_state.append(token)

                    if not classified:
                        head_buf.append(token)
                        joined = "".join(head_buf)
                        # Decide as soon as we have enough text or hit a clear
                        # leak marker mid-buffer.
                        ready = (
                            len(joined.strip()) >= HEAD_DECIDE_CHARS
                            or self.engine.looks_like_leak_head(joined)
                        )
                        if not ready:
                            continue
                        is_leak = self.engine.looks_like_leak_head(joined)
                        classified = True
                        if not is_leak:
                            # Flush head verbatim; subsequent tokens passthrough.
                            all_tokens.append(joined)
                            await token_queue.put(joined)
                        # If leak: stay silent, keep collecting raw_tokens_for_state.
                        continue

                    if is_leak:
                        # Hold silently; final flush happens below.
                        continue

                    # Clean passthrough.
                    all_tokens.append(token)
                    await token_queue.put(token)
            finally:
                # Final flush: handle the cases where the stream ended before
                # we reached the decide threshold, or we were in silent leak mode.
                if not classified:
                    # Stream ended before HEAD_DECIDE_CHARS — sanitize whatever
                    # we have and emit it as a single chunk.
                    full = "".join(raw_tokens_for_state)
                    cleaned = self.engine._clean_response(full) or ""
                    if cleaned:
                        all_tokens.append(cleaned)
                        await token_queue.put(cleaned)
                elif is_leak:
                    full = "".join(raw_tokens_for_state)
                    cleaned = self.engine._clean_response(full) or ""
                    if cleaned:
                        all_tokens.append(cleaned)
                        await token_queue.put(cleaned)
                await token_queue.put(_END)

        async def _token_consumer() -> AsyncGenerator[str, None]:
            """Async generator that drains the queue for TTS consumption."""
            while True:
                item = await token_queue.get()
                if item is _END:
                    return
                yield item

        # Start producer as a background task
        producer_task = asyncio.create_task(_token_producer())

        # Run TTS streaming consumer — yields TTSResult per sentence
        tts_gen = self._tts.synthesize_streaming(_token_consumer())

        # We need to interleave text tokens (for display) with TTS results
        # (for audio). The trick: drain the token queue for display while
        # also awaiting the TTS generator.
        #
        # Strategy: run both concurrently using asyncio.create_task for TTS
        # and yield text tokens as they arrive in all_tokens.

        tts_results: asyncio.Queue[Optional[TTSResult]] = asyncio.Queue()
        tts_done = asyncio.Event()

        async def _tts_collector() -> None:
            try:
                async for tts_result in tts_gen:
                    await tts_results.put(tts_result)
            finally:
                await tts_results.put(None)  # sentinel
                tts_done.set()

        tts_task = asyncio.create_task(_tts_collector())

        # Yield text tokens as they arrive, interspersed with TTS results
        emitted = 0
        while True:
            # Drain any new text tokens first
            while emitted < len(all_tokens):
                yield StreamingQuestionChunk(text_delta=all_tokens[emitted])
                emitted += 1

            # Check for a ready TTS chunk (non-blocking)
            try:
                tts_chunk = tts_results.get_nowait()
                if tts_chunk is None:
                    # TTS done sentinel
                    break
                yield StreamingQuestionChunk(text_delta="", tts_result=tts_chunk)
                continue
            except asyncio.QueueEmpty:
                pass

            # If both producer and TTS are done, flush and exit
            if producer_task.done() and tts_done.is_set():
                # Final drain
                while emitted < len(all_tokens):
                    yield StreamingQuestionChunk(text_delta=all_tokens[emitted])
                    emitted += 1
                # Drain remaining TTS results
                while not tts_results.empty():
                    tts_chunk = await tts_results.get()
                    if tts_chunk is not None:
                        yield StreamingQuestionChunk(text_delta="", tts_result=tts_chunk)
                break

            # Wait a tiny bit before next poll to avoid busy-loop
            await asyncio.sleep(0.005)

        # Ensure tasks complete cleanly
        await producer_task
        await tts_task

        # State bookkeeping — clean the RAW buffer (all_tokens may already
        # contain a producer-cleaned single chunk in leak mode).
        original_question = self.engine._clean_response("".join(raw_tokens_for_state))
        
        print(f"\n   ✓ Generation completed")
        print(f"   📝 Original Question: \"{original_question}\"")
        
        if (
            not original_question or len(original_question.strip()) < 10
            or self.engine.is_termination_attempt(original_question)
            or state.is_duplicate_question(original_question)
            or self.engine.is_generic_question(original_question)
        ):
            original_question = state.get_recovery_question()
            print(f"   ⚠️  Used recovery question: \"{original_question}\"")

        # Apply refiner if enabled
        final_question = original_question
        refiner_output = None
        if self._enable_refiner and self._refiner and request.candidate_latest_answer:
            print("\n✨ REFINER - Enhancing question...")
            try:
                refiner_start = time.time()
                refiner_output = await self._refiner.refine(
                    original_question=original_question,
                    cv_topic=analysis.question_focus or state.current_topic or "CV items",
                    knowledge_level=analysis.knowledge_level,
                    response_quality=analysis.quality,
                    question_focus=analysis.question_focus,
                    acknowledgment=analysis.acknowledgment,
                    timeout_sec=2.5,
                )
                refiner_latency_ms = (time.time() - refiner_start) * 1000
                
                print(f"   ✓ Refinement completed in {refiner_latency_ms:.0f}ms")
                print(f"   • Type: {refiner_output.refinement_type}")
                print(f"   • Improvements: {refiner_output.improvements}")
                
                if refiner_output.refinement_type != "maintained":
                    final_question = refiner_output.refined_question
                    print(f"\n   ✏️  Refined Question:")
                    print(f"   \"{final_question}\"")
                else:
                    print(f"   ℹ️  Original question maintained (no changes)")
            except Exception as e:
                print(f"   ✗ Refiner error: {e}")
                refiner_output = None
        else:
            print("\n✨ REFINER - Skipped (disabled or no answer)")

        state.record_question(final_question)

        print("\n" + "="*70)
        print(f"✅ FINAL QUESTION: \"{final_question}\"")
        print("="*70 + "\n")

        # If the producer's leak gate suppressed the entire stream (cleaner
        # returned empty), nothing has reached the SSE display yet. Emit the
        # recovery question now so the frontend has something to render.
        emitted_text = "".join(all_tokens).strip()
        if not emitted_text and final_question:
            yield StreamingQuestionChunk(text_delta=final_question)
            # And feed it to TTS too so the user actually hears the question.
            if self._tts is not None:
                try:
                    tts_result = await self._tts.synthesize(final_question)
                    if tts_result is not None:
                        yield StreamingQuestionChunk(text_delta="", tts_result=tts_result)
                except Exception as e:
                    print(f"   ✗ Recovery TTS error: {e}")

        yield StreamingQuestionChunk(text_delta="", is_final=True)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def analyze_response(self, sid: str, answer: str) -> Dict[str, Any]:
        state = self._get_or_create_session(sid)
        analysis = state.build_analysis(answer)
        return {
            "word_count": analysis.word_count,
            "quality": analysis.quality,
            "engagement": analysis.engagement,
            "is_detailed": analysis.is_detailed,
            "wants_to_skip": analysis.wants_to_skip,
        }

    def get_session_stats(self, sid: str) -> Dict[str, Any]:
        state = self._get_or_create_session(sid)
        return {
            "turn_count": state.turn_count,
            "topic_depth": state.topic_depth,
            "consecutive_vague": state.consecutive_vague,
            "questions_asked": len(state.questions_asked),
            "topics_covered": list(state.topics_covered),
        }

    @property
    def active_session_count(self) -> int:
        return len(self._sessions)