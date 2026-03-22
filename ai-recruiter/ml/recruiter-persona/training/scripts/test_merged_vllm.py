"""Realtime vLLM interview simulator — TTFT-optimised edition.

Key improvements over v1
------------------------
Fix #1  Streaming: sync ``vllm.LLM`` silently ignores ``stream=True``; replaced
        with ``AsyncLLMEngine`` so TTFT is a real first-token measurement.
Fix #2  TTFT fallback: non-stream path was reporting full latency as TTFT; now
        exposes a clear N/A when streaming is disabled.
Fix #3  Delta decode: removed fragile ``startswith`` branch; cumulative vLLM
        output means a simple slice always works.
Fix #4  Chunked prefill (``enable_chunked_prefill=True``) added alongside
        prefix caching — splits long system-prompt prefills into smaller
        chunks so the engine reaches the first new token sooner.
Fix #5  Warm-up now primes the KV cache with the *actual system prompt* so
        turn-1 has a cache hit instead of a cold miss.
Fix #6  Token count derived from the vLLM ``CompletionOutput`` directly
        (``len(output.token_ids)``) — no extra tokenizer round-trip.
Fix #7  CUDA-graph path left on (``enforce_eager=False``); added
        ``max_num_batched_tokens`` arg for prefill throughput tuning.
Fix #8  ``tensor_parallel_size`` auto-detected from available GPUs.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Candidate CV — Oussema Aissaoui (hardcoded for testing)
# ---------------------------------------------------------------------------
CANDIDATE_CV = """
Candidate: Oussema Aissaoui
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

DEFAULT_SYSTEM_PROMPT = (
    "You are Alex, a senior technical recruiter interviewing Oussema Aissaoui "
    "for an AI Engineering Internship position. "
    "You have already reviewed his CV. Use it to ask targeted, specific questions. "
    "Ask exactly ONE concise follow-up question per turn based on his latest answer and CV. "
    "Do not greet or use filler words like 'great' or 'thanks'. "
    "Do not write candidate dialogue. Do not write multiple questions. "
    "Stop after the first question mark. "
    f"\n\nCandidate CV:\n{CANDIDATE_CV}"
)

# Leaner variant — same CV context, shorter instruction overhead.
COMPACT_SYSTEM_PROMPT = (
    "You are Alex, a senior recruiter interviewing Oussema Aissaoui for an AI Engineering Internship. "
    "You have read his CV. Ask exactly ONE specific, technical follow-up question per turn. "
    "No greetings. No filler. No multiple questions. Stop after the first '?'. "
    f"\n\nCandidate CV:\n{CANDIDATE_CV}"
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TTFT-optimised interview simulator (vLLM + AsyncLLMEngine)"
    )
    p.add_argument(
        "--model-id",
        default="oussema2021/recruiter-persona-llama-3.1-8b-merged-v2",
        help="Merged model repo on Hugging Face Hub",
    )
    p.add_argument(
        "--mode",
        choices=["single", "chat"],
        default="chat",
        help="single: one candidate turn → one recruiter output | chat: full loop",
    )
    p.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="Override system prompt (use --compact-prompt for a shorter default)",
    )
    p.add_argument(
        "--compact-prompt",
        action="store_true",
        help="Use the shorter built-in system prompt to reduce prefill tokens",
    )
    p.add_argument(
        "--candidate",
        default="I am a backend engineer with 6 years of Python and distributed systems experience.",
        help="Candidate answer for --mode single",
    )
    p.add_argument("--max-turns", type=int, default=8)
    p.add_argument(
        "--history-turn-window",
        type=int,
        default=4,
        help="Rolling (candidate, recruiter) pairs kept in context",
    )
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--temperature", type=float, default=0.15)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--repetition-penalty", type=float, default=1.05)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    p.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help=(
            "Max tokens processed in one prefill step. "
            "Lower values (e.g. 512) reduce TTFT on long prompts via chunked prefill. "
            "Defaults to max-model-len when unset."
        ),
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
        help=(
            "Stream output tokens via AsyncLLMEngine and report true TTFT. "
            "Without this flag TTFT is not measurable and shown as N/A."
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

_STOP_MARKERS = ["\nCandidate:", "\nUser:", "\nInterviewer:", "\nRecruiter:"]


def clean_one_turn(text: str) -> str:
    """Trim special tokens, multi-turn leakage, and keep only the first question.

    Because "?" is used as a vLLM stop string, it is stripped from the raw
    output. We always re-append it so the recruiter's turn reads as a proper
    question.
    """
    cleaned = text.replace("<|eot_id|>", "").strip()

    for marker in _STOP_MARKERS:
        idx = cleaned.find(marker)
        if idx != -1:
            cleaned = cleaned[:idx].strip()

    # Truncate to the first "?" if the model somehow emitted multiple questions
    # (shouldn't happen with stop="?" but acts as a safety net).
    q_idx = cleaned.find("?")
    if q_idx != -1:
        cleaned = cleaned[: q_idx + 1].strip()
    else:
        # vLLM consumed the "?" as a stop string — add it back.
        cleaned = cleaned.rstrip(".").strip() + "?"

    cleaned = " ".join(cleaned.split())
    return cleaned or "Could you tell me more about your experience with AI systems?"


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(
    tokenizer,
    system_prompt: str,
    turns: List[Tuple[str, str]],
    candidate_input: str,
) -> str:
    messages = [{"role": "system", "content": system_prompt}]

    # CV injected as a synthetic prior exchange.
    # Fine-tuned recruiter models ignore system-prompt content and run their
    # trained generic script instead. Framing the CV as an already-established
    # conversation fact forces the model to treat it as ground truth and ask
    # project-specific questions rather than hallucinating a generic interview.
    messages.append({
        "role": "user",
        "content": (
            "Before we start, here is the candidate's CV:\n"
            f"{CANDIDATE_CV}\n"
            "Use it to ask specific, targeted questions only about his real experience."
        ),
    })
    messages.append({
        "role": "assistant",
        "content": (
            "Understood. I've reviewed Oussema's CV. He has hands-on experience with "
            "GraphRAG and LLM reasoning at TALAN Tunisia, a YOLOv8 + SAM blood cell "
            "segmentation pipeline, and an ML model for crop water efficiency using "
            "Google Earth Engine. His core stack is Python, TensorFlow, and deep learning. "
            "I'll ask targeted technical questions based solely on these real projects."
        ),
    })

    for cand, rec in turns:
        messages.append({"role": "user", "content": cand})
        messages.append({"role": "assistant", "content": rec})
    messages.append({"role": "user", "content": candidate_input})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ---------------------------------------------------------------------------
# Engine factory
# ---------------------------------------------------------------------------

def _detect_tp_size(requested: int) -> int:
    """Return TP size: explicit arg > GPU count > 1."""
    if requested > 0:
        return requested
    try:
        import torch
        n = torch.cuda.device_count()
        return max(n, 1)
    except Exception:
        return 1


def make_async_engine(args: argparse.Namespace):
    """
    Build an AsyncLLMEngine — the only vLLM path that supports real streaming
    and therefore real TTFT measurement.
    """
    from vllm import AsyncLLMEngine, AsyncEngineArgs  # type: ignore

    tp = _detect_tp_size(args.tensor_parallel_size)

    engine_args = AsyncEngineArgs(
        model=args.model_id,
        dtype=args.dtype,
        tensor_parallel_size=tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        # Fix #4 — chunked prefill: large prompts are split so the first chunk
        # reaches the decode phase sooner, cutting TTFT on long contexts.
        enable_chunked_prefill=True,
        max_num_batched_tokens=args.max_num_batched_tokens or args.max_model_len,
        # Fix #5 prerequisite — keep prefix caching on so system-prompt KV
        # blocks are reused across turns.
        enable_prefix_caching=True,
        # Keep CUDA graphs enabled (default) for lower per-step kernel overhead.
        enforce_eager=False,
    )
    return AsyncLLMEngine.from_engine_args(engine_args)


def make_sync_engine(args: argparse.Namespace):
    """Fallback sync engine used when --stream is not requested."""
    from vllm import LLM  # type: ignore

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
        # Older vLLM builds may not support all kwargs — strip unknowns gracefully.
        unsupported = {"enable_chunked_prefill", "max_num_batched_tokens", "enforce_eager"}
        for key in unsupported:
            kwargs.pop(key, None)
        print(f"[warn] Engine init fell back to minimal kwargs ({exc})")
        return LLM(**kwargs)


# ---------------------------------------------------------------------------
# Async generation (real streaming TTFT)
# ---------------------------------------------------------------------------

async def _async_generate(
    engine,
    sampling_params,
    prompt: str,
    request_id: str,
    echo_stream: bool,
) -> tuple[str, float, float | None, int]:
    """
    Stream tokens from AsyncLLMEngine.

    Returns
    -------
    raw_text        : cumulative generated text
    latency_sec     : wall time for full generation
    ttft_sec        : time to first token (None if no tokens produced)
    num_output_toks : token count from vLLM internals (Fix #6)
    """
    t0 = time.perf_counter()
    ttft: float | None = None
    prev_len = 0
    final_output = None

    async for request_output in engine.generate(prompt, sampling_params, request_id):
        if not request_output.outputs:
            continue

        out = request_output.outputs[0]
        current_text = out.text or ""

        # Fix #3 — simple slice on cumulative string; no startswith guard needed.
        delta = current_text[prev_len:]
        prev_len = len(current_text)

        if delta and ttft is None:
            ttft = time.perf_counter() - t0

        if echo_stream and delta:
            print(delta, end="", flush=True)

        final_output = out

    latency = time.perf_counter() - t0
    raw_text = final_output.text if final_output else ""
    # Fix #6 — read token count directly from vLLM output, no extra encode().
    num_toks = len(final_output.token_ids) if final_output else 0

    return raw_text, latency, ttft, num_toks


# ---------------------------------------------------------------------------
# Sync generation (no real TTFT available)
# ---------------------------------------------------------------------------

def _sync_generate(
    llm,
    sampling_params,
    prompt: str,
) -> tuple[str, float, int]:
    t0 = time.perf_counter()
    outputs = llm.generate([prompt], sampling_params)
    latency = time.perf_counter() - t0
    out = outputs[0].outputs[0] if outputs and outputs[0].outputs else None
    raw_text = out.text if out else ""
    num_toks = len(out.token_ids) if out else 0  # Fix #6
    return raw_text, latency, num_toks


# ---------------------------------------------------------------------------
# Warm-up: prime the KV prefix cache with the system prompt (Fix #5)
# ---------------------------------------------------------------------------

async def warmup_async(engine, sampling_params, tokenizer, system_prompt: str) -> None:
    """
    Generate a single dummy turn so the system-prompt KV blocks are cached
    before the first real user interaction.
    """
    from vllm import SamplingParams as SP  # type: ignore

    dummy_prompt = build_prompt(tokenizer, system_prompt, [], "warmup")
    warmup_sp = SP(max_tokens=1, temperature=0.0)
    async for _ in engine.generate(dummy_prompt, warmup_sp, request_id="warmup-0"):
        pass
    print("[warmup] Prefix cache primed with system prompt.\n")


def warmup_sync(llm, sampling_params, tokenizer, system_prompt: str) -> None:
    from vllm import SamplingParams as SP  # type: ignore

    dummy_prompt = build_prompt(tokenizer, system_prompt, [], "warmup")
    warmup_sp = SP(max_tokens=1, temperature=0.0)
    llm.generate([dummy_prompt], warmup_sp)
    print("[warmup] Prefix cache primed with system prompt.\n")


# ---------------------------------------------------------------------------
# Single mode
# ---------------------------------------------------------------------------

async def run_single_async(args, tokenizer, engine, sampling):
    from vllm import SamplingParams  # type: ignore  (for stop tokens)

    prompt = build_prompt(tokenizer, args.system_prompt, [], args.candidate)
    request_id = f"single-{time.monotonic_ns()}"

    if args.stream:
        print("RECRUITER_STREAM: ", end="", flush=True)

    raw, latency, ttft, num_toks = await _async_generate(
        engine, sampling, prompt, request_id, echo_stream=args.stream
    )
    if args.stream:
        print()

    answer = clean_one_turn(raw)
    toks_per_sec = num_toks / latency if latency > 0 else 0.0
    ttft_str = f"{ttft:.3f}" if ttft is not None else "N/A"

    print(f"\nMODEL            : {args.model_id}")
    print(f"LATENCY_SEC      : {latency:.3f}")
    print(f"TTFT_SEC         : {ttft_str}")
    print(f"OUTPUT_TOKENS    : {num_toks}")
    print(f"TOKENS_PER_SEC   : {toks_per_sec:.2f}")
    print(f"CANDIDATE        : {args.candidate}")
    print(f"RECRUITER        : {answer}")


def run_single_sync(args, tokenizer, llm, sampling):
    prompt = build_prompt(tokenizer, args.system_prompt, [], args.candidate)
    raw, latency, num_toks = _sync_generate(llm, sampling, prompt)
    answer = clean_one_turn(raw)
    toks_per_sec = num_toks / latency if latency > 0 else 0.0

    print(f"\nMODEL            : {args.model_id}")
    print(f"LATENCY_SEC      : {latency:.3f}")
    print(f"TTFT_SEC         : N/A  (run with --stream for real TTFT)")
    print(f"OUTPUT_TOKENS    : {num_toks}")
    print(f"TOKENS_PER_SEC   : {toks_per_sec:.2f}")
    print(f"CANDIDATE        : {args.candidate}")
    print(f"RECRUITER        : {answer}")


# ---------------------------------------------------------------------------
# Chat mode
# ---------------------------------------------------------------------------

async def run_chat_async(args, tokenizer, engine, sampling):
    _chat_header()
    turns: List[Tuple[str, str]] = []
    req_counter = 0

    for _ in range(args.max_turns):
        candidate_input = input("Candidate > ").strip()
        if not candidate_input:
            print("Please type an answer before continuing.\n")
            continue
        if candidate_input.lower() in {"/quit", "quit", "exit"}:
            print("\nInterview ended.")
            return
        if candidate_input.lower() == "/restart":
            turns = []
            print("\nInterview history reset.\n")
            continue

        rolling = turns[-args.history_turn_window:]
        prompt = build_prompt(tokenizer, args.system_prompt, rolling, candidate_input)
        request_id = f"chat-{req_counter}"
        req_counter += 1

        if args.stream:
            print("Recruiter > ", end="", flush=True)

        raw, latency, ttft, num_toks = await _async_generate(
            engine, sampling, prompt, request_id, echo_stream=args.stream
        )

        recruiter = clean_one_turn(raw)
        toks_per_sec = num_toks / latency if latency > 0 else 0.0
        ttft_str = f"{ttft:.2f}s" if ttft is not None else "N/A"

        if args.stream:
            print()
        else:
            print(f"Recruiter > {recruiter}")

        print(
            f"[ttft={ttft_str} | latency={latency:.2f}s | "
            f"output_tokens={num_toks} | tok/s={toks_per_sec:.2f}]\n"
        )
        turns.append((candidate_input, recruiter))

    print("Reached max turns. Interview session ended.")


def run_chat_sync(args, tokenizer, llm, sampling):
    _chat_header()
    turns: List[Tuple[str, str]] = []

    for _ in range(args.max_turns):
        candidate_input = input("Candidate > ").strip()
        if not candidate_input:
            print("Please type an answer before continuing.\n")
            continue
        if candidate_input.lower() in {"/quit", "quit", "exit"}:
            print("\nInterview ended.")
            return
        if candidate_input.lower() == "/restart":
            turns = []
            print("\nInterview history reset.\n")
            continue

        rolling = turns[-args.history_turn_window:]
        prompt = build_prompt(tokenizer, args.system_prompt, rolling, candidate_input)
        raw, latency, num_toks = _sync_generate(llm, sampling, prompt)
        recruiter = clean_one_turn(raw)
        toks_per_sec = num_toks / latency if latency > 0 else 0.0

        print(f"Recruiter > {recruiter}")
        print(
            f"[ttft=N/A (use --stream) | latency={latency:.2f}s | "
            f"output_tokens={num_toks} | tok/s={toks_per_sec:.2f}]\n"
        )
        turns.append((candidate_input, recruiter))

    print("Reached max turns. Interview session ended.")


def _chat_header() -> None:
    print("\n" + "=" * 72)
    print("SIMULATED INTERVIEW MODE")
    print("=" * 72)
    print("Provide your candidate answers, model responds like recruiter Alex.")
    print("Commands: /quit  /restart")
    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Pick system prompt.
    if args.compact_prompt:
        args.system_prompt = COMPACT_SYSTEM_PROMPT
        print("[info] Using compact system prompt to reduce prefill tokens.")

    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "AutoTokenizer import failed. "
            "Repair your env: pip install transformers."
        ) from exc

    from vllm import SamplingParams  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # Build stop-token list.
    stop_ids: list[int] = []
    if tokenizer.eos_token_id is not None:
        stop_ids.append(tokenizer.eos_token_id)
    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot, int) and eot >= 0 and eot not in stop_ids:
        stop_ids.append(eot)

    sampling = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        stop_token_ids=stop_ids,
        # "?" hard-stops after the first complete question — fixes multi-question drift.
        # clean_one_turn() re-appends it since vLLM excludes the stop string.
        stop=[*_STOP_MARKERS, "\n👤", "\n🤖", "?"],
    )

    if args.stream:
        # ------------------------------------------------------------------ #
        # Async path — AsyncLLMEngine with real token streaming & real TTFT   #
        # ------------------------------------------------------------------ #
        engine = make_async_engine(args)

        async def _run():
            await warmup_async(engine, sampling, tokenizer, args.system_prompt)
            if args.mode == "single":
                await run_single_async(args, tokenizer, engine, sampling)
            else:
                await run_chat_async(args, tokenizer, engine, sampling)

        asyncio.run(_run())

    else:
        # ------------------------------------------------------------------ #
        # Sync path — simpler, no real TTFT, but lower setup overhead         #
        # ------------------------------------------------------------------ #
        llm = make_sync_engine(args)
        warmup_sync(llm, sampling, tokenizer, args.system_prompt)

        if args.mode == "single":
            run_single_sync(args, tokenizer, llm, sampling)
        else:
            run_chat_sync(args, tokenizer, llm, sampling)
if __name__ == "__main__":
    main()