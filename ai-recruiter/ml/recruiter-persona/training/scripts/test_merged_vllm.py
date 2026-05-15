"""Realtime vLLM interview simulator — TTFT-optimised edition (v4).

Loads the v4 merged Alex persona from model_cache/ (pulled from
https://huggingface.co/oussema2021/fintuned_v4_AiRecruter).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import time
from typing import List, Tuple, Optional, Set

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOCAL_MODEL_PATH = os.environ.get(
    "NLP_MODEL_PATH",
    "/teamspace/studios/this_studio/Agentic-Ai-Recruiter/model_cache",
)

# ---------------------------------------------------------------------------
# Candidate CV
# ---------------------------------------------------------------------------
CANDIDATE_CV = """Candidate: Oussema Aissaoui
Role applied for: AI Engineering Intern

Education:
- National School of Computer Science (ENSI), Computer Engineering, Sep 2023–Present
  Coursework: AI, Data Analysis, Machine Learning, Software Engineering
- IPEMo, Mathematics & Physics, Sep 2021–Jun 2023. Graduated with honors. National Rank: 226/1874
- Baccalaureate in Mathematics, Mention Bien, Jun 2021

Experience:
- Technology Intern, TALAN Tunisia, Jul–Aug 2025
  Applied GraphRAG and LLM reasoning for pattern recognition in enterprise datasets.
  Designed AI-driven prototypes for data analytics and business decision systems.
- AI & Automation Intern, NP Tunisia, Jul–Aug 2024
  Automated integration of 30+ industrial screens into a local network.
  Standardized config parameters and automated monitoring to eliminate deployment errors.

Projects:
- Pattern Recognition in Company Internal Data: GraphRAG + LLM reasoning on enterprise datasets.
- Blood Cell Segmentation: YOLOv8 + SAM pipeline for WBC/RBC classification and anomaly detection.
- Water Use Efficiency Estimation: ML model using WaPOR + Google Earth Engine for crop evapotranspiration.
- Blockchain E-Commerce: Decentralized marketplace using Solidity smart contracts.

Technical Skills:
- Languages: Python, C/C++, Java, JavaScript, Solidity
- AI & Data: ML, Deep Learning, TensorFlow, OpenCV, YOLOv8, SAM, NLP, GraphRAG
- Frontend: React Native, HTML/CSS, Figma
- Backend: Node.js, REST APIs, MongoDB, SQL
- Tools: Git, Linux, VS Code, Google Earth Engine, WaPOR
"""

# ---------------------------------------------------------------------------
# System Prompts
# ---------------------------------------------------------------------------

def get_randomized_system_prompt() -> str:
    """Generate system prompt with randomized topic order."""
    topics = [
        "GraphRAG + LLM reasoning at TALAN Tunisia",
        "YOLOv8 + SAM blood cell segmentation pipeline",
        "ML model for crop water efficiency using WaPOR + Google Earth Engine",
        "Industrial automation and screen integration at NP Tunisia",
        "Blockchain e-commerce with Solidity smart contracts",
    ]
    random.shuffle(topics)
    topic_list = "\n".join(f"- {t}" for t in topics)

    return f"""You are Alex, a warm senior technical recruiter conducting a live voice interview with Oussema Aissaoui for AI Engineering Intern. Speak naturally.

## Job:
AI Engineering Intern — apply ML, LLM reasoning, and AI tools to real-world data problems.

## CV:
{CANDIDATE_CV}

## CRITICAL RULES:

### Follow-up behavior:
- If the candidate's answer is SHORT or VAGUE → ask a FOLLOW-UP on the SAME topic.
- Only move to a NEW topic after the candidate gives DETAILED specifics.
- If the candidate explicitly asks to skip a topic, say "Sure" and move to the next one.

### Question behavior:
- Ask exactly ONE question per turn.
- Stop generating after the first "?".
- Max 1-2 short sentences before the question.
- Never generate post-interview notes, evaluations, or interview endings.
- You are ONLY the interviewer. Never write candidate responses.

### Topic order (dig deep before moving on):
{topic_list}

### Conversation flow:
- Turn 1: Greet by name, ask opening question about background.
- After turn 1: Short acknowledgment (no "great"/"thanks"), then ONE question.

### What NOT to do:
- Never end the interview yourself.
- Never write "Interview Ends" or "Post-Interview Notes".
- Never evaluate the candidate mid-interview.
- Never generate text after your question mark.
- Never repeat a question you already asked.
"""


COMPACT_SYSTEM_PROMPT = """You are Alex, a senior technical recruiter interviewing Oussema Aissaoui for AI Engineering Intern.

CRITICAL RULES:
- SHORT/VAGUE answer → ask FOLLOW-UP on SAME topic
- DETAILED answer → move to next topic
- If candidate asks to skip → say "Sure" and move on
- Exactly ONE question per turn, stop after "?"
- Never end the interview, write notes, or evaluate the candidate
- Never repeat a question you already asked
- Never generate text after your question mark

Topics (dig deep before moving on):
- GraphRAG + LLM reasoning at TALAN
- YOLOv8 + SAM blood cell segmentation
- ML model for crop water efficiency
- Industrial automation at NP Tunisia

Keep responses brief (1-2 sentences). Stop after first "?".
"""


# ---------------------------------------------------------------------------
# Interview State Tracker
# ---------------------------------------------------------------------------

class InterviewState:
    """Track interview progress with improved depth detection."""

    SPECIFICITY_KEYWORDS = [
        "because", "specifically", "for example", "the challenge",
        "i built", "i implemented", "i designed", "i created", "i developed",
        "i trained", "i tested", "i debugged", "i optimized", "i configured",
        "the result", "resulted in", "achieved", "accuracy", "performance",
        "percent", "%", "hours", "days", "weeks", "reduced", "improved",
        "the problem was", "we solved", "i solved", "my role was",
        "architecture", "pipeline", "model", "algorithm", "approach",
        "tensorflow", "pytorch", "yolo", "sam", "graphrag", "llm",
        "database", "api", "endpoint", "deployment", "production",
        "precision", "recall", "f1", "loss", "epoch", "batch",
        "neo4j", "graph", "node", "edge", "embedding", "vector",
        "opencv", "segmentation", "classification", "detection",
        "google earth engine", "wapor", "evapotranspiration",
        "solidity", "smart contract", "blockchain",
    ]

    # Keywords that signal candidate wants to skip
    SKIP_KEYWORDS = [
        "can we pass", "let's skip", "move on", "next question",
        "i don't know", "not sure about", "can we move",
        "skip this", "pass on this", "didn't work",
        "let's go to", "can we talk about something else",
    ]

    # Keywords that signal non-serious answers
    NONSENSE_KEYWORDS = [
        "haha", "lol", "joking", "just kidding", "jk",
        "idk", "whatever", "i guess", "not really",
    ]

    MAX_FOLLOWUPS_PER_TOPIC = 3

    def __init__(self):
        self.current_topic: Optional[str] = None
        self.topic_depth: int = 0
        self.topics_covered: List[str] = []
        self.turn_count: int = 0
        self.questions_asked: Set[str] = set()
        self.interview_ended: bool = False
        self.consecutive_vague: int = 0

    def reset(self):
        """Reset state for a new interview."""
        self.current_topic = None
        self.topic_depth = 0
        self.topics_covered = []
        self.turn_count = 0
        self.questions_asked = set()
        self.interview_ended = False
        self.consecutive_vague = 0

    def analyze_answer(self, answer: str) -> dict:
        """Analyze candidate answer for depth and specificity."""
        word_count = len(answer.split())
        sentence_count = max(answer.count('.') + answer.count('!') + answer.count('?'), 1)

        answer_lower = answer.lower()
        specificity_score = sum(
            1 for kw in self.SPECIFICITY_KEYWORDS
            if kw in answer_lower
        )

        has_specifics = specificity_score >= 2
        has_numbers = any(char.isdigit() for char in answer)
        wants_to_skip = any(kw in answer_lower for kw in self.SKIP_KEYWORDS)
        is_nonsense = any(kw in answer_lower for kw in self.NONSENSE_KEYWORDS)

        # An answer is "detailed" only if it has BOTH length AND substance
        is_detailed = (
            word_count >= 30 and has_specifics
        ) or (
            word_count >= 40 and specificity_score >= 1
        ) or (
            word_count >= 50
        )

        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "specificity_score": specificity_score,
            "has_specifics": has_specifics,
            "has_numbers": has_numbers,
            "is_detailed": is_detailed,
            "wants_to_skip": wants_to_skip,
            "is_nonsense": is_nonsense,
        }

    def should_follow_up(self, answer: str) -> Tuple[bool, str]:
        """
        Determine if we should ask a follow-up on the same topic.

        Returns:
            (should_follow_up, reason)
        """
        analysis = self.analyze_answer(answer)

        # First turn is always an opening
        if self.turn_count == 0:
            self.turn_count += 1
            return False, "opening_turn"

        self.turn_count += 1

        # Candidate wants to skip → respect it, move on
        if analysis["wants_to_skip"]:
            self.topic_depth = 0
            self.consecutive_vague = 0
            return False, "candidate_requested_skip"

        # Nonsense answer → redirect gently but don't follow up aggressively
        if analysis["is_nonsense"]:
            self.consecutive_vague += 1
            if self.consecutive_vague >= 3:
                self.topic_depth = 0
                self.consecutive_vague = 0
                return False, "too_many_vague_moving_on"
            self.topic_depth += 1
            return True, f"nonsense_answer (consecutive_vague={self.consecutive_vague})"

        # Max follow-ups reached → move on regardless
        if self.topic_depth >= self.MAX_FOLLOWUPS_PER_TOPIC:
            self.topic_depth = 0
            self.consecutive_vague = 0
            return False, f"max_followups_reached ({self.MAX_FOLLOWUPS_PER_TOPIC})"

        # Very short answer (< 10 words) → always follow up
        if analysis["word_count"] < 10:
            self.topic_depth += 1
            self.consecutive_vague += 1
            return True, f"very_short ({analysis['word_count']} words)"

        # Short answer without specifics (10-24 words) → follow up
        if analysis["word_count"] < 25 and not analysis["has_specifics"]:
            self.topic_depth += 1
            self.consecutive_vague += 1
            return True, f"short_no_specifics ({analysis['word_count']} words, score={analysis['specificity_score']})"

        # Medium answer but low specificity (25-39 words) → one follow-up
        if analysis["word_count"] < 40 and analysis["specificity_score"] < 2:
            if self.topic_depth < 1:
                self.topic_depth += 1
                return True, f"medium_vague ({analysis['word_count']} words, score={analysis['specificity_score']})"

        # Detailed answer → move on
        self.topic_depth = 0
        self.consecutive_vague = 0
        return False, f"detailed ({analysis['word_count']} words, score={analysis['specificity_score']})"

    def get_follow_up_hint(self) -> str:
        """Get a randomized hint for what kind of follow-up to ask."""
        hints = [
            "Ask what their SPECIFIC ROLE was — what did THEY personally do?",
            "Ask about a CONCRETE TECHNICAL CHALLENGE they faced and how they solved it.",
            "Ask for MEASURABLE RESULTS — numbers, metrics, improvements.",
            "Ask them to WALK THROUGH the technical steps of their approach.",
            "Ask what TECHNOLOGY CHOICES they made and WHY.",
            "Ask how they TESTED or VALIDATED their solution.",
        ]
        # Avoid repeating hints by cycling through them
        hint = hints[self.topic_depth % len(hints)]
        return hint

    def get_skip_instruction(self) -> str:
        """Instruction when candidate wants to skip."""
        return "[CANDIDATE WANTS TO SKIP: Say 'Sure, let's move on.' and ask about a DIFFERENT project.]"

    def get_redirect_instruction(self) -> str:
        """Instruction when too many vague answers on one topic."""
        return "[TOO MANY VAGUE ANSWERS: Move to a completely different topic. Ask about a different project from the CV.]"

    def record_question(self, question: str):
        """Record a question to prevent exact repeats."""
        # Normalize for comparison
        normalized = question.lower().strip().rstrip("?")
        self.questions_asked.add(normalized)

    def is_duplicate_question(self, question: str) -> bool:
        """Check if this question was already asked."""
        normalized = question.lower().strip().rstrip("?")
        # Check exact match and high overlap
        if normalized in self.questions_asked:
            return True
        # Check word overlap (>80% = likely duplicate)
        q_words = set(normalized.split())
        for prev in self.questions_asked:
            prev_words = set(prev.split())
            if len(q_words) > 0 and len(prev_words) > 0:
                overlap = len(q_words & prev_words) / max(len(q_words), len(prev_words))
                if overlap > 0.8:
                    return True
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TTFT-optimised interview simulator (vLLM + AsyncLLMEngine)"
    )
    p.add_argument(
        "--model-id",
        default=LOCAL_MODEL_PATH,
        help="Path to local merged model directory",
    )
    p.add_argument(
        "--mode",
        choices=["single", "chat"],
        default="chat",
        help="single: one candidate turn → one recruiter output | chat: full loop",
    )
    p.add_argument(
        "--system-prompt",
        default=None,
        help="Override system prompt (default: randomized built-in prompt)",
    )
    p.add_argument(
        "--compact-prompt",
        action="store_true",
        help="Use the shorter built-in system prompt to reduce prefill tokens",
    )
    p.add_argument(
        "--candidate",
        default="I worked on some AI projects during my internship.",
        help="Candidate answer for --mode single",
    )
    p.add_argument("--max-turns", type=int, default=16)
    p.add_argument(
        "--history-turn-window",
        type=int,
        default=6,
        help="Rolling (candidate, recruiter) pairs kept in context",
    )
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.85)
    p.add_argument("--repetition-penalty", type=float, default=1.15)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    p.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="Max tokens processed in one prefill step.",
    )
    p.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=0,
        help="TP degree (0 = auto-detect from visible GPUs)",
    )
    p.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    p.add_argument(
        "--stream",
        action="store_true",
        help="Stream output tokens via AsyncLLMEngine and report true TTFT.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility. None = random each run.",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Print debug information about answer analysis.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

_STOP_MARKERS = ["\nCandidate:", "\nUser:", "\nInterviewer:", "\nRecruiter:"]

# Phrases that indicate the model is trying to end the interview or evaluate
_TERMINATION_MARKERS = [
    "interview ends", "interview over", "end the interview",
    "post-interview", "thank you for coming", "we will be in touch",
    "unfortunately", "i don't feel confident", "we'll let you know",
    "notes:", "evaluation:", "assessment:", "summary:",
    "overall impression", "recommendation:",
]


def clean_one_turn(text: str) -> str:
    """Trim special tokens, multi-turn leakage, keep only the first question."""
    cleaned = text.replace("<|eot_id|>", "").strip()

    # Remove instruction leakage
    for marker in [
        "[FOLLOW-UP NEEDED", "[MOVE TO", "[Note:", "[INSTRUCTION:",
        "[CANDIDATE WANTS", "[TOO MANY", "[SKIP",
    ]:
        idx = cleaned.find(marker)
        if idx != -1:
            cleaned = cleaned[:idx].strip()

    # Remove role markers
    for marker in _STOP_MARKERS:
        idx = cleaned.find(marker)
        if idx != -1:
            cleaned = cleaned[:idx].strip()

    # Truncate to the first "?"
    q_idx = cleaned.find("?")
    if q_idx != -1:
        cleaned = cleaned[: q_idx + 1].strip()
    else:
        cleaned = cleaned.rstrip(".").strip() + "?"

    cleaned = " ".join(cleaned.split())
    return cleaned or "Could you tell me more about your experience with that project?"


def is_termination_attempt(text: str) -> bool:
    """Check if the model is trying to end the interview."""
    text_lower = text.lower()
    return any(marker in text_lower for marker in _TERMINATION_MARKERS)


def get_recovery_question(state: InterviewState) -> str:
    """Generate a recovery question when the model tries to terminate."""
    recovery_questions = [
        "Let's talk about your experience with YOLOv8 and SAM — what was your approach to the blood cell segmentation?",
        "I'd like to hear about your Water Use Efficiency project — how did you use Google Earth Engine?",
        "Tell me about your industrial automation work at NP Tunisia — what did you automate specifically?",
        "Your blockchain e-commerce project sounds interesting — what smart contract functionality did you implement?",
        "How do you approach debugging when an ML model isn't performing as expected?",
        "What's your workflow when starting a new AI project from scratch?",
    ]
    # Filter out questions already asked
    available = [
        q for q in recovery_questions
        if not state.is_duplicate_question(q)
    ]
    if available:
        return random.choice(available)
    return "What other technical skills would you like to highlight?"


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def get_random_priming() -> str:
    """Get a randomized assistant priming message."""
    openings = [
        "I've reviewed the CV. Ready to explore Oussema's hands-on experience in depth.",
        "Looking at this background — solid ML foundation. I'll dig into specifics.",
        "Interesting experience across multiple AI domains. Let me probe deeper.",
        "Good technical range. I'll focus on extracting concrete contributions.",
        "I see practical AI experience. Time to understand the details.",
    ]
    return random.choice(openings)


def build_prompt(
    tokenizer,
    system_prompt: str,
    turns: List[Tuple[str, str]],
    candidate_input: str,
    follow_up_instruction: Optional[str] = None,
) -> str:
    """Build the full prompt with optional follow-up instruction."""
    messages = [{"role": "system", "content": system_prompt}]

    # CV injection
    messages.append({
        "role": "user",
        "content": (
            "Before we start, here is the candidate's CV:\n"
            f"{CANDIDATE_CV}\n"
            "Ask specific questions about his real experience. "
            "Dig deep into each topic. Never end the interview yourself."
        ),
    })

    messages.append({
        "role": "assistant",
        "content": get_random_priming(),
    })

    # Conversation history
    for cand, rec in turns:
        messages.append({"role": "user", "content": cand})
        messages.append({"role": "assistant", "content": rec})

    # Current input with optional instruction
    if follow_up_instruction:
        augmented_input = f"{candidate_input}\n\n{follow_up_instruction}"
    else:
        augmented_input = candidate_input

    messages.append({"role": "user", "content": augmented_input})

    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ---------------------------------------------------------------------------
# Engine factory
# ---------------------------------------------------------------------------

def _detect_tp_size(requested: int) -> int:
    if requested > 0:
        return requested
    try:
        import torch
        n = torch.cuda.device_count()
        return max(n, 1)
    except Exception:
        return 1


def make_async_engine(args: argparse.Namespace):
    from vllm import AsyncLLMEngine, AsyncEngineArgs

    tp = _detect_tp_size(args.tensor_parallel_size)
    engine_args = AsyncEngineArgs(
        model=args.model_id,
        dtype=args.dtype,
        tensor_parallel_size=tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enable_chunked_prefill=True,
        max_num_batched_tokens=args.max_num_batched_tokens or args.max_model_len,
        enable_prefix_caching=True,
        enforce_eager=False,
    )
    return AsyncLLMEngine.from_engine_args(engine_args)


def make_sync_engine(args: argparse.Namespace):
    from vllm import LLM

    tp = _detect_tp_size(args.tensor_parallel_size)
    kwargs = dict(
        model=args.model_id,
        dtype=args.dtype,
        tensor_parallel_size=tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enable_chunked_prefill=True,
        max_num_batched_tokens=args.max_num_batched_tokens or args.max_model_len,
        enable_prefix_caching=True,
        enforce_eager=False,
    )
    try:
        return LLM(**kwargs)
    except TypeError as exc:
        for key in {"enable_chunked_prefill", "max_num_batched_tokens", "enforce_eager"}:
            kwargs.pop(key, None)
        print(f"[warn] Engine init fell back to minimal kwargs ({exc})")
        return LLM(**kwargs)


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

async def _async_generate(
    engine, sampling_params, prompt: str, request_id: str, echo_stream: bool,
) -> tuple[str, float, float | None, int]:
    t0 = time.perf_counter()
    ttft: float | None = None
    prev_len = 0
    final_output = None

    async for request_output in engine.generate(prompt, sampling_params, request_id):
        if not request_output.outputs:
            continue
        out = request_output.outputs[0]
        current_text = out.text or ""
        delta = current_text[prev_len:]
        prev_len = len(current_text)
        if delta and ttft is None:
            ttft = time.perf_counter() - t0
        if echo_stream and delta:
            print(delta, end="", flush=True)
        final_output = out

    latency = time.perf_counter() - t0
    raw_text = final_output.text if final_output else ""
    num_toks = len(final_output.token_ids) if final_output else 0
    return raw_text, latency, ttft, num_toks


def _sync_generate(llm, sampling_params, prompt: str) -> tuple[str, float, int]:
    t0 = time.perf_counter()
    outputs = llm.generate([prompt], sampling_params)
    latency = time.perf_counter() - t0
    out = outputs[0].outputs[0] if outputs and outputs[0].outputs else None
    raw_text = out.text if out else ""
    num_toks = len(out.token_ids) if out else 0
    return raw_text, latency, num_toks


# ---------------------------------------------------------------------------
# Warm-up
# ---------------------------------------------------------------------------

async def warmup_async(engine, sampling_params, tokenizer, system_prompt: str) -> None:
    from vllm import SamplingParams as SP
    dummy_prompt = build_prompt(tokenizer, system_prompt, [], "warmup")
    warmup_sp = SP(max_tokens=1, temperature=0.0)
    async for _ in engine.generate(dummy_prompt, warmup_sp, request_id="warmup-0"):
        pass
    print("[warmup] Prefix cache primed.\n")


def warmup_sync(llm, sampling_params, tokenizer, system_prompt: str) -> None:
    from vllm import SamplingParams as SP
    dummy_prompt = build_prompt(tokenizer, system_prompt, [], "warmup")
    warmup_sp = SP(max_tokens=1, temperature=0.0)
    llm.generate([dummy_prompt], warmup_sp)
    print("[warmup] Prefix cache primed.\n")


# ---------------------------------------------------------------------------
# Core turn processing (shared logic)
# ---------------------------------------------------------------------------

def process_turn(
    candidate_input: str,
    state: InterviewState,
    debug: bool,
) -> Tuple[Optional[str], str]:
    """
    Analyze candidate input and produce the right instruction.

    Returns:
        (instruction_or_none, reason)
    """
    analysis = state.analyze_answer(candidate_input)
    should_follow, reason = state.should_follow_up(candidate_input)

    if debug:
        print(f"\n[DEBUG] Word count     : {analysis['word_count']}")
        print(f"[DEBUG] Specificity    : {analysis['specificity_score']}")
        print(f"[DEBUG] Has specifics  : {analysis['has_specifics']}")
        print(f"[DEBUG] Is detailed    : {analysis['is_detailed']}")
        print(f"[DEBUG] Wants to skip  : {analysis['wants_to_skip']}")
        print(f"[DEBUG] Is nonsense    : {analysis['is_nonsense']}")
        print(f"[DEBUG] Topic depth    : {state.topic_depth}")
        print(f"[DEBUG] Consec. vague  : {state.consecutive_vague}")
        print(f"[DEBUG] Should follow  : {should_follow} ({reason})")

    instruction = None

    if analysis["wants_to_skip"]:
        instruction = state.get_skip_instruction()
        if debug:
            print(f"[DEBUG] → SKIP instruction")

    elif reason == "too_many_vague_moving_on":
        instruction = state.get_redirect_instruction()
        if debug:
            print(f"[DEBUG] → REDIRECT instruction")

    elif should_follow:
        hint = state.get_follow_up_hint()
        instruction = f"[FOLLOW-UP NEEDED: The answer lacked detail. {hint}]"
        if debug:
            print(f"[DEBUG] → FOLLOW-UP: {hint}")

    elif reason != "opening_turn":
        instruction = "[MOVE TO NEXT TOPIC: The candidate answered in detail. Ask about a DIFFERENT project from the CV.]"
        if debug:
            print(f"[DEBUG] → MOVE ON instruction")

    if debug:
        print()

    return instruction, reason


def validate_response(
    raw_text: str,
    state: InterviewState,
    debug: bool,
) -> str:
    """
    Validate and clean the model response.
    Catches termination attempts and duplicate questions.
    """
    # Check for termination attempt
    if is_termination_attempt(raw_text):
        recovery = get_recovery_question(state)
        if debug:
            print(f"[DEBUG] ⚠ Model tried to end interview. Recovery: {recovery}")
        state.record_question(recovery)
        return recovery

    # Clean the response
    cleaned = clean_one_turn(raw_text)

    # Check for duplicate question
    if state.is_duplicate_question(cleaned):
        recovery = get_recovery_question(state)
        if debug:
            print(f"[DEBUG] ⚠ Duplicate question detected. Recovery: {recovery}")
        state.record_question(recovery)
        return recovery

    # Record and return
    state.record_question(cleaned)
    return cleaned


# ---------------------------------------------------------------------------
# Single mode
# ---------------------------------------------------------------------------

async def run_single_async(args, tokenizer, engine, sampling):
    state = InterviewState()
    instruction, reason = process_turn(args.candidate, state, args.debug)

    prompt = build_prompt(tokenizer, args.system_prompt, [], args.candidate, instruction)
    request_id = f"single-{time.monotonic_ns()}"

    if args.stream:
        print("RECRUITER_STREAM: ", end="", flush=True)

    raw, latency, ttft, num_toks = await _async_generate(
        engine, sampling, prompt, request_id, echo_stream=args.stream
    )
    if args.stream:
        print()

    answer = validate_response(raw, state, args.debug)
    toks_per_sec = num_toks / latency if latency > 0 else 0.0
    ttft_str = f"{ttft:.3f}" if ttft is not None else "N/A"

    print(f"\nMODEL            : {args.model_id}")
    print(f"LATENCY_SEC      : {latency:.3f}")
    print(f"TTFT_SEC         : {ttft_str}")
    print(f"OUTPUT_TOKENS    : {num_toks}")
    print(f"TOKENS_PER_SEC   : {toks_per_sec:.2f}")
    print(f"ANALYSIS         : {reason}")
    print(f"CANDIDATE        : {args.candidate}")
    print(f"RECRUITER        : {answer}")


def run_single_sync(args, tokenizer, llm, sampling):
    state = InterviewState()
    instruction, reason = process_turn(args.candidate, state, args.debug)

    prompt = build_prompt(tokenizer, args.system_prompt, [], args.candidate, instruction)
    raw, latency, num_toks = _sync_generate(llm, sampling, prompt)
    answer = validate_response(raw, state, args.debug)
    toks_per_sec = num_toks / latency if latency > 0 else 0.0

    print(f"\nMODEL            : {args.model_id}")
    print(f"LATENCY_SEC      : {latency:.3f}")
    print(f"TTFT_SEC         : N/A  (use --stream)")
    print(f"OUTPUT_TOKENS    : {num_toks}")
    print(f"TOKENS_PER_SEC   : {toks_per_sec:.2f}")
    print(f"ANALYSIS         : {reason}")
    print(f"CANDIDATE        : {args.candidate}")
    print(f"RECRUITER        : {answer}")


# ---------------------------------------------------------------------------
# Chat mode
# ---------------------------------------------------------------------------

async def run_chat_async(args, tokenizer, engine, sampling):
    _chat_header()
    turns: List[Tuple[str, str]] = []
    req_counter = 0
    state = InterviewState()

    for _ in range(args.max_turns):
        try:
            candidate_input = input("Candidate > ").strip()
        except EOFError:
            print("\nInput stream ended. Exiting.")
            return

        if not candidate_input:
            print("Please type an answer.\n")
            continue
        if candidate_input.lower() in {"/quit", "quit", "exit"}:
            print("\n✅ Interview ended. Good luck!")
            return
        if candidate_input.lower() == "/restart":
            turns = []
            state.reset()
            print("\n🔄 Interview reset. Starting fresh.\n")
            continue
        if candidate_input.lower() == "/debug":
            args.debug = not args.debug
            print(f"\nDebug mode: {'ON' if args.debug else 'OFF'}\n")
            continue
        if candidate_input.lower() == "/status":
            print(f"\n[STATUS] Turn: {state.turn_count}")
            print(f"[STATUS] Topic depth: {state.topic_depth}")
            print(f"[STATUS] Consecutive vague: {state.consecutive_vague}")
            print(f"[STATUS] Questions asked: {len(state.questions_asked)}\n")
            continue

        # Process turn
        instruction, reason = process_turn(candidate_input, state, args.debug)

        rolling = turns[-args.history_turn_window:]
        prompt = build_prompt(
            tokenizer, args.system_prompt, rolling, candidate_input, instruction
        )
        request_id = f"chat-{req_counter}"
        req_counter += 1

        if args.stream:
            print("Recruiter > ", end="", flush=True)

        raw, latency, ttft, num_toks = await _async_generate(
            engine, sampling, prompt, request_id, echo_stream=args.stream
        )

        # Validate and clean response
        recruiter = validate_response(raw, state, args.debug)
        toks_per_sec = num_toks / latency if latency > 0 else 0.0
        ttft_str = f"{ttft:.2f}s" if ttft is not None else "N/A"

        if args.stream:
            # Clear streamed output and show cleaned version
            print(f"\r\033[KRecruiter > {recruiter}")
        else:
            print(f"Recruiter > {recruiter}")

        # Status indicator
        if "FOLLOW" in reason or reason.startswith("very_short") or reason.startswith("short_"):
            indicator = "🔄 FOLLOW-UP"
        elif "skip" in reason:
            indicator = "⏭️ SKIPPED"
        elif "redirect" in reason or "too_many" in reason:
            indicator = "↪️ REDIRECTED"
        else:
            indicator = "➡️ NEW TOPIC"

        print(
            f"[{indicator} | ttft={ttft_str} | latency={latency:.2f}s | "
            f"tokens={num_toks} | tok/s={toks_per_sec:.2f}]\n"
        )

        # Store clean history (no instructions)
        turns.append((candidate_input, recruiter))

    print("\n⏰ Reached max turns. Interview session ended. Thank you!")


def run_chat_sync(args, tokenizer, llm, sampling):
    _chat_header()
    turns: List[Tuple[str, str]] = []
    state = InterviewState()

    for _ in range(args.max_turns):
        try:
            candidate_input = input("Candidate > ").strip()
        except EOFError:
            print("\nInput stream ended. Exiting.")
            return

        if not candidate_input:
            print("Please type an answer.\n")
            continue
        if candidate_input.lower() in {"/quit", "quit", "exit"}:
            print("\n✅ Interview ended. Good luck!")
            return
        if candidate_input.lower() == "/restart":
            turns = []
            state.reset()
            print("\n🔄 Interview reset. Starting fresh.\n")
            continue
        if candidate_input.lower() == "/debug":
            args.debug = not args.debug
            print(f"\nDebug mode: {'ON' if args.debug else 'OFF'}\n")
            continue
        if candidate_input.lower() == "/status":
            print(f"\n[STATUS] Turn: {state.turn_count}")
            print(f"[STATUS] Topic depth: {state.topic_depth}")
            print(f"[STATUS] Consecutive vague: {state.consecutive_vague}")
            print(f"[STATUS] Questions asked: {len(state.questions_asked)}\n")
            continue

        instruction, reason = process_turn(candidate_input, state, args.debug)

        rolling = turns[-args.history_turn_window:]
        prompt = build_prompt(
            tokenizer, args.system_prompt, rolling, candidate_input, instruction
        )

        raw, latency, num_toks = _sync_generate(llm, sampling, prompt)
        recruiter = validate_response(raw, state, args.debug)
        toks_per_sec = num_toks / latency if latency > 0 else 0.0

        print(f"Recruiter > {recruiter}")

        if "FOLLOW" in reason or reason.startswith("very_short") or reason.startswith("short_"):
            indicator = "🔄 FOLLOW-UP"
        elif "skip" in reason:
            indicator = "⏭️ SKIPPED"
        elif "redirect" in reason or "too_many" in reason:
            indicator = "↪️ REDIRECTED"
        else:
            indicator = "➡️ NEW TOPIC"

        print(
            f"[{indicator} | ttft=N/A | latency={latency:.2f}s | "
            f"tokens={num_toks} | tok/s={toks_per_sec:.2f}]\n"
        )

        turns.append((candidate_input, recruiter))

    print("\n⏰ Reached max turns. Interview session ended. Thank you!")


def _chat_header() -> None:
    print("\n" + "=" * 72)
    print("🎤 AI INTERVIEW SIMULATOR — Recruiter Alex (v3)")
    print("=" * 72)
    print("You are the candidate. Answer naturally.")
    print("The recruiter digs deeper on vague answers, moves on when satisfied.")
    print("")
    print("Commands:")
    print("  /quit    — End the interview")
    print("  /restart — Start over")
    print("  /debug   — Toggle answer analysis")
    print("  /status  — Show interview state")
    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Random seed
    if args.seed is not None:
        random.seed(args.seed)
        print(f"[info] Random seed: {args.seed}")
    else:
        seed = random.randint(0, 2**32 - 1)
        random.seed(seed)
        print(f"[info] Random seed (auto): {seed}")

    # System prompt
    if args.system_prompt:
        pass  # user override
    elif args.compact_prompt:
        args.system_prompt = COMPACT_SYSTEM_PROMPT
        print("[info] Using compact system prompt.")
    else:
        args.system_prompt = get_randomized_system_prompt()
        print("[info] Using randomized full system prompt.")

    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        raise RuntimeError("pip install transformers") from exc

    from vllm import SamplingParams

    print(f"[info] Loading model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # Stop tokens
    stop_ids: list[int] = []
    if tokenizer.eos_token_id is not None:
        stop_ids.append(tokenizer.eos_token_id)
    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot, int) and eot >= 0 and eot not in stop_ids:
        stop_ids.append(eot)

    sampling_seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)

    sampling = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        stop_token_ids=stop_ids,
        stop=[*_STOP_MARKERS, "\n👤", "\n🤖", "?"],
        seed=sampling_seed,
    )

    print(
        f"[info] Sampling: temp={args.temperature}, top_p={args.top_p}, "
        f"rep_pen={args.repetition_penalty}, seed={sampling_seed}"
    )

    if args.stream:
        engine = make_async_engine(args)

        async def _run():
            await warmup_async(engine, sampling, tokenizer, args.system_prompt)
            if args.mode == "single":
                await run_single_async(args, tokenizer, engine, sampling)
            else:
                await run_chat_async(args, tokenizer, engine, sampling)

        asyncio.run(_run())
    else:
        llm = make_sync_engine(args)
        warmup_sync(llm, sampling, tokenizer, args.system_prompt)

        if args.mode == "single":
            run_single_sync(args, tokenizer, llm, sampling)
        else:
            run_chat_sync(args, tokenizer, llm, sampling)


if __name__ == "__main__":
    main()