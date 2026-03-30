#!/usr/bin/env python3
"""
Realtime inference tester (v2) for one-turn recruiter adapters.

Design goals:
- vLLM-first path for low-latency generation
- One-turn interaction aligned with one-turn training objective
- Simple benchmark mode with latency + throughput stats
"""

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import List, Optional


SYSTEM_PROMPT = (
    "You are Alex, a senior recruiter conducting a structured and professional "
    "interview. Ask exactly one concise follow-up question based on the "
    "candidate's latest answer. Avoid roleplay continuation, avoid simulating "
    "candidate replies, and avoid discriminatory or illegal questions."
)


def infer_base_model(adapter_path: Path) -> Optional[str]:
    """Infer base model from adapter config when available."""
    cfg_path = adapter_path / "adapter_config.json"
    if not cfg_path.exists():
        return None
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg.get("base_model_name_or_path")
    except Exception:
        return None


def clean_one_turn_response(text: str) -> str:
    """Keep a single recruiter follow-up and cut multi-speaker continuation."""
    cleaned = text.replace("<|eot_id|>", "").strip()
    if not cleaned:
        return "Could you tell me more about your relevant experience for this role?"

    lower = cleaned.lower()
    markers = [
        "candidate:",
        "interviewer:",
        "recruiter:",
        "user:",
        "assistant:",
        "\n👤",
        "\n🤖",
    ]
    cut_idx = -1
    for m in markers:
        idx = lower.find(m)
        if idx != -1 and (cut_idx == -1 or idx < cut_idx):
            cut_idx = idx
    if cut_idx != -1:
        cleaned = cleaned[:cut_idx].strip()

    q_idx = cleaned.find("?")
    if q_idx != -1:
        cleaned = cleaned[: q_idx + 1].strip()

    cleaned = " ".join(cleaned.split())
    if not cleaned:
        return "Could you share more about your experience with this role's core responsibilities?"
    return cleaned


class RealtimeVLLMTester:
    """Low-latency vLLM inference wrapper for one-turn recruiter testing."""

    def __init__(
        self,
        base_model: str,
        adapter_path: str,
        dtype: str = "bfloat16",
        max_model_len: int = 2048,
        gpu_memory_utilization: float = 0.92,
        max_lora_rank: int = 64,
    ):
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_lora_rank = max_lora_rank
        self.llm = None
        self.lora_request = None
        self.tokenizer = None
        self.stop_token_ids: List[int] = []

    def load(self):
        """Initialize tokenizer + vLLM engine."""
        try:
            from transformers import AutoTokenizer
        except Exception as exc:
            raise RuntimeError(
                "Failed to import AutoTokenizer from transformers. "
                "Please fix the Python environment first."
            ) from exc

        try:
            from vllm import LLM
            from vllm.lora.request import LoRARequest
        except ImportError as exc:
            raise RuntimeError(
                "vLLM is not installed. Install with: pip install vllm"
            ) from exc

        # Prefer adapter tokenizer if shipped, then fallback to base model tokenizer.
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.adapter_path)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)

        eos = self.tokenizer.eos_token_id
        if isinstance(eos, int) and eos >= 0:
            self.stop_token_ids.append(eos)
        eot = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if isinstance(eot, int) and eot >= 0 and eot not in self.stop_token_ids:
            self.stop_token_ids.append(eot)

        llm_kwargs = {
            "model": self.base_model,
            "dtype": self.dtype,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "enable_lora": True,
            "max_lora_rank": self.max_lora_rank,
            "trust_remote_code": True,
        }

        # Some vLLM builds support this; fallback gracefully if not.
        try:
            self.llm = LLM(enable_prefix_caching=True, **llm_kwargs)
        except TypeError:
            self.llm = LLM(**llm_kwargs)

        self.lora_request = LoRARequest(
            lora_name="recruiter_persona_adapter_v2",
            lora_int_id=1,
            lora_path=self.adapter_path,
        )

    def _build_prompt(self, candidate_text: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": candidate_text.strip()},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate_one_turn(
        self,
        candidate_text: str,
        max_new_tokens: int = 64,
        temperature: float = 0.2,
        top_p: float = 0.9,
        repetition_penalty: float = 1.05,
    ):
        """Generate one recruiter follow-up and return text + timing stats."""
        from vllm import SamplingParams

        prompt = self._build_prompt(candidate_text)
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop_token_ids=self.stop_token_ids if self.stop_token_ids else None,
            stop=["\nCandidate:", "\nInterviewer:", "\nUser:", "\n👤", "\n🤖"],
        )

        t0 = time.perf_counter()
        outputs = self.llm.generate(
            [prompt],
            sampling_params=sampling_params,
            lora_request=self.lora_request,
        )
        t1 = time.perf_counter()

        raw = ""
        if outputs and outputs[0].outputs:
            raw = outputs[0].outputs[0].text
        text = clean_one_turn_response(raw)

        elapsed = max(t1 - t0, 1e-6)
        token_count = len(self.tokenizer.encode(text, add_special_tokens=False))
        toks_per_sec = token_count / elapsed

        return {
            "response": text,
            "latency_sec": elapsed,
            "tokens": token_count,
            "tok_per_sec": toks_per_sec,
        }


def run_one_turn_chat(tester: RealtimeVLLMTester, args):
    print("\n" + "=" * 72)
    print("Realtime One-Turn Mode (v2)")
    print("=" * 72)
    print("Type a candidate response and get one recruiter follow-up.")
    print("Commands: quit | exit")
    print("=" * 72 + "\n")

    while True:
        candidate = input("Candidate > ").strip()
        if candidate.lower() in {"quit", "exit", "q"}:
            print("\nSession ended.")
            break
        if not candidate:
            print("Please enter a candidate response.\n")
            continue

        out = tester.generate_one_turn(
            candidate_text=candidate,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        print(f"Recruiter > {out['response']}")
        print(
            f"[latency={out['latency_sec']:.2f}s | tokens={out['tokens']} | "
            f"speed={out['tok_per_sec']:.1f} tok/s]\n"
        )


def run_benchmark(tester: RealtimeVLLMTester, args):
    scenarios = [
        "I have 5 years of backend experience with Python and distributed systems.",
        "I built recommendation models in a fintech startup and improved CTR by 15%.",
        "I managed a cross-functional team shipping B2B SaaS products.",
        "I focus on UX research and prototyping in Figma for mobile apps.",
    ]
    latencies = []
    throughputs = []

    print("\nRunning realtime benchmark...")
    print(f"Iterations: {args.bench_iterations}")

    for i in range(args.bench_iterations):
        text = scenarios[i % len(scenarios)]
        out = tester.generate_one_turn(
            candidate_text=text,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        latencies.append(out["latency_sec"])
        throughputs.append(out["tok_per_sec"])

    lat_sorted = sorted(latencies)
    p50 = statistics.median(lat_sorted)
    p95_idx = min(len(lat_sorted) - 1, int(0.95 * len(lat_sorted)) - 1)
    p95 = lat_sorted[max(0, p95_idx)]

    print("\nBenchmark results")
    print("-" * 40)
    print(f"avg latency: {statistics.mean(latencies):.3f}s")
    print(f"p50 latency: {p50:.3f}s")
    print(f"p95 latency: {p95:.3f}s")
    print(f"avg tok/s:   {statistics.mean(throughputs):.2f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Realtime v2 inference tester (vLLM) for one-turn recruiter adapters"
    )
    parser.add_argument(
        "--adapter-path",
        required=True,
        help="Path to LoRA adapter directory (final checkpoint)",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Base model id; inferred from adapter_config.json when omitted",
    )
    parser.add_argument(
        "--mode",
        choices=["chat", "bench", "single"],
        default="chat",
        help="chat: interactive one-turn, bench: latency benchmark, single: one input/one output",
    )
    parser.add_argument("--candidate-text", default="", help="Candidate input for single mode")

    # Realtime defaults
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)

    # vLLM engine knobs
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--max-lora-rank", type=int, default=64)

    # Benchmark
    parser.add_argument("--bench-iterations", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    adapter_path = Path(args.adapter_path)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")

    base_model = args.base_model or infer_base_model(adapter_path)
    if not base_model:
        raise ValueError(
            "Base model could not be inferred. Provide --base-model explicitly."
        )

    tester = RealtimeVLLMTester(
        base_model=base_model,
        adapter_path=str(adapter_path),
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_lora_rank=args.max_lora_rank,
    )

    print("\nInitializing vLLM engine...")
    print(f"base model: {base_model}")
    print(f"adapter:    {adapter_path}")
    tester.load()
    print("vLLM ready.\n")

    if args.mode == "chat":
        run_one_turn_chat(tester, args)
        return

    if args.mode == "bench":
        run_benchmark(tester, args)
        return

    if not args.candidate_text.strip():
        raise ValueError("--candidate-text is required in single mode")

    out = tester.generate_one_turn(
        candidate_text=args.candidate_text,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    print("response:")
    print(out["response"])
    print(
        f"\nlatency={out['latency_sec']:.3f}s, tokens={out['tokens']}, "
        f"speed={out['tok_per_sec']:.2f} tok/s"
    )


if __name__ == "__main__":
    main()
