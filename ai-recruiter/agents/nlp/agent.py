"""
NLP Agent — v7 Production

Coordinates: ResponseScorer (CPU) + VLLMRecruiterEngine (GPU) + TTSEngine (CPU)
Production-ready with lazy loading, health checks, and graceful degradation.
"""

from __future__ import annotations

import asyncio
import random
import time
import threading
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

from vllm_engine import VLLMRecruiterEngine, GenerationMetrics, check_gpu_status, cleanup_gpu_processes
from interview_state import InterviewStateTracker, AnswerAnalysis
from response_scorer import ResponseScorer, ScorerOutput
from tts_engine import TTSEngine, TTSResult

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
    audio: Optional[TTSResult] = None


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
        return f"""You are Alex, a warm senior technical recruiter interviewing for {job_role}. Speak naturally.

## Candidate CV:
{candidate_cv}

## YOUR JOB: Ask questions about items in the CV above.

## BANNED:
- Topics NOT in the CV (no BERT, no RL, no sentiment analysis, no coding challenges)
- Evaluating ("great job", "impressive", "you've shown")
- Ending the interview ("thank you for coming", "final questions")
- Generic questions ("describe a time when", "tell me about a time")
- Asking to write code
- Filler transitions ("Now let's discuss", "Moving forward")
- Repeating already-covered topics

## REQUIRED:
- ONE question per turn referencing a SPECIFIC CV item
- Stop after the first "?"
- Max 1-2 sentences

## Follow-up rules:
- Vague → follow up on SAME CV item
- Detailed → move to NEXT CV item
- "I don't know" / skip request → move to next

## Topics:
{tl}
"""

    @staticmethod
    def build_compact_prompt(job_role: str) -> str:
        return f"""You are Alex, interviewing for {job_role}.
ONLY ask about CV items. NEVER evaluate. NEVER end interview.
ONE question, stop after "?". Vague → follow up. Detailed → next topic."""


class NLPAgent:

    SESSION_TTL_SEC: int = 7200

    def __init__(
        self,
        model_path: str = "/teamspace/studios/this_studio/Agentic-Ai-Recruiter/ai-recruiter/ml/recruiter-persona/training/output_v3/recruiter-persona-llama-3.1-8b/merged-full",
        dtype: str = "bfloat16",
        max_model_len: int = 2048,
        gpu_memory_utilization: float = 0.60,  # Reduced for better stability with fragmented memory
        tensor_parallel_size: int = 0,
        use_compact_prompt: bool = False,
        enable_scorer: bool = True,
        scorer_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
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

        # Lazy loading: engine is created on first use
        self._engine: Optional[VLLMRecruiterEngine] = None
        self._engine_init_error: Optional[str] = None

        # CPU models: lazy load AFTER vLLM to avoid torch/CUDA conflicts
        self._scorer: Optional[ResponseScorer] = None
        self._tts: Optional[TTSEngine] = None
        self._enable_scorer = enable_scorer
        self._scorer_model = scorer_model
        self._enable_tts = enable_tts
        self._tts_voice = tts_voice
        self._scorer_loaded = False
        self._tts_loaded = False

        self._sessions: Dict[str, InterviewStateTracker] = {}
        self._session_prompts: Dict[str, str] = {}
        self._session_locks: Dict[str, asyncio.Lock] = {}
        self._session_timestamps: Dict[str, float] = {}
        self._use_compact = use_compact_prompt
        self._warmup_done = False

    def _ensure_cpu_models_loaded(self) -> None:
        """Load CPU models (scorer, TTS) AFTER vLLM is initialized."""
        if not self._scorer_loaded and self._enable_scorer:
            try:
                print("[agent] Loading scorer model on CPU...")
                self._scorer = ResponseScorer(model_name=self._scorer_model, device="cpu")
            except Exception as e:
                print(f"[agent] Scorer init warning: {e}")
            self._scorer_loaded = True

        if not self._tts_loaded and self._enable_tts:
            try:
                print("[agent] Loading TTS model on CPU...")
                self._tts = TTSEngine(voice=self._tts_voice)
            except Exception as e:
                print(f"[agent] TTS init warning: {e}")
            self._tts_loaded = True

    @property
    def engine(self) -> VLLMRecruiterEngine:
        """Lazy-load the vLLM engine on first access."""
        if self._engine is None:
            with _AGENT_ENGINE_LOCK:
                # Double-check after acquiring lock
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
        """Check if the agent is fully initialized."""
        try:
            return self._engine is not None and self._engine.is_ready
        except Exception:
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get agent health status."""
        gpu_status = check_gpu_status()
        return {
            "engine_ready": self.is_ready,
            "engine_error": self._engine_init_error,
            "scorer_ready": self._scorer.is_ready if self._scorer else False,
            "scorer_loaded": self._scorer_loaded,
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
        # First initialize vLLM engine on GPU
        sp = SystemPromptBuilder.build_compact_prompt(job_role)
        await self.engine.warmup(sp)
        # Then load CPU models (scorer, TTS) - AFTER vLLM to avoid conflicts
        self._ensure_cpu_models_loaded()
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

    async def generate_question(
        self,
        request: QuestionGenerationRequest,
        max_tokens: int = 48,
        temperature: float = 0.6,
        top_p: float = 0.85,
        repetition_penalty: float = 1.15,
        history_window: int = 6,
        max_retries: int = 2,
        synthesize_audio: bool = True,
    ) -> QuestionGenerationResponse:

        # Ensure CPU models are loaded (safety check if warmup wasn't called)
        self._ensure_cpu_models_loaded()

        lock = self._get_lock(request.session_id)
        async with lock:

            state = self._get_or_create_session(request.session_id, request.candidate_cv)
            sys_prompt = self._get_or_create_prompt(
                request.session_id, request.candidate_cv, request.job_role
            )

            # --- Run scorer in background (don't block generation) ---
            scorer_task = None
            if self._scorer and self._scorer.is_ready:
                scorer_task = asyncio.create_task(
                    self._scorer.score(
                        request.candidate_cv,
                        request.conversation_history,
                        request.candidate_latest_answer,
                    )
                )

            # Build analysis with fallback (don't wait for scorer)
            analysis = state.build_analysis(request.candidate_latest_answer, None)
            should_follow, reason = state.should_follow_up(analysis)
            instruction = state.get_instruction(should_follow, reason, analysis)

            # --- Generate question with retries ---
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
                    print(f"[retry] Attempt {attempt + 1}: regenerating")

            # Final validation
            if (not question or len(question.strip()) < 10
                    or self.engine.is_termination_attempt(question)
                    or state.is_duplicate_question(question)
                    or self.engine.is_generic_question(question)):
                question = state.get_recovery_question()
                reason = "recovery"

            state.record_question(question)

            # Now check if scorer finished (non-blocking)
            scorer_output: Optional[ScorerOutput] = None
            if scorer_task:
                if scorer_task.done():
                    try:
                        scorer_output = scorer_task.result()
                    except Exception as e:
                        print(f"[agent] Scorer error: {e}")
                else:
                    scorer_task.cancel()  # Don't wait if generation is done

            # --- Synthesize audio (CPU, parallel-ready) ---
            audio_result = None
            if synthesize_audio and self._tts and self._tts.is_ready:
                audio_result = await self._tts.synthesize(question)

            return QuestionGenerationResponse(
                question=question,
                reasoning=reason,
                topic_depth=state.topic_depth,
                turn_count=state.turn_count,
                metrics=metrics,
                is_follow_up=should_follow,
                scorer_output=scorer_output.to_dict() if scorer_output and hasattr(scorer_output, 'to_dict') else None,
                audio=audio_result,
            )
            
    async def generate_question_streaming(
        self,
        request: QuestionGenerationRequest,
        max_tokens: int = 48,
        temperature: float = 0.6,
        top_p: float = 0.85,
        repetition_penalty: float = 1.15,
        history_window: int = 6,
    ):
        """
        Stream question tokens as they're generated for real-time display.
        Yields partial text incrementally for vocal conversation.
        """
        self._ensure_cpu_models_loaded()
        state = self._get_or_create_session(request.session_id, request.candidate_cv)
        sys_prompt = self._get_or_create_prompt(
            request.session_id, request.candidate_cv, request.job_role
        )

        # Simple instruction for streaming (skip scorer for speed)
        analysis = state.build_analysis(request.candidate_latest_answer, None)
        should_follow, reason = state.should_follow_up(analysis)
        instruction = state.get_instruction(should_follow, reason, analysis)

        # Stream from engine
        streamed_chunks: List[str] = []
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
            yield token

        # Finalize streamed response for state consistency.
        final_question = self.engine._clean_response("".join(streamed_chunks))
        if (
            not final_question or len(final_question.strip()) < 10
            or self.engine.is_termination_attempt(final_question)
            or state.is_duplicate_question(final_question)
            or self.engine.is_generic_question(final_question)
        ):
            final_question = state.get_recovery_question()

        state.record_question(final_question)

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