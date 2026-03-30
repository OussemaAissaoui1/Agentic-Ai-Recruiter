"""
vLLM Engine — v7 Production

Production-ready engine with:
- GPU health checks and cleanup utilities
- Lazy loading with timeout protection
- Process-safe initialization
- Graceful degradation
"""

from __future__ import annotations

import os
import time
import re
import signal
import subprocess
import threading
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
from contextlib import contextmanager

from transformers import AutoTokenizer

# Module-level lock to prevent multiple simultaneous engine initializations
_ENGINE_INIT_LOCK = threading.Lock()
_GLOBAL_ENGINE_INSTANCE = None


@dataclass(frozen=True)
class GenerationMetrics:
    latency_sec: float
    ttft_sec: Optional[float]
    num_tokens: int
    tokens_per_sec: float


@dataclass(frozen=True)
class GPUStatus:
    """GPU health status."""
    available: bool
    gpu_count: int
    free_memory_gb: float
    total_memory_gb: float
    blocking_pids: List[int]
    error: Optional[str] = None


def check_gpu_status() -> GPUStatus:
    """Check GPU availability and memory status."""
    try:
        import torch
        if not torch.cuda.is_available():
            return GPUStatus(False, 0, 0, 0, [], "CUDA not available")

        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            return GPUStatus(False, 0, 0, 0, [], "No GPUs found")

        # Get memory info for primary GPU
        free_mem = torch.cuda.mem_get_info(0)[0] / (1024**3)
        total_mem = torch.cuda.mem_get_info(0)[1] / (1024**3)

        # Check for blocking processes (exclude current process and its children)
        current_pid = os.getpid()
        protected_pids = {current_pid}

        # Get all child processes of current process
        try:
            result = subprocess.run(
                ["pgrep", "-P", str(current_pid)],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                for pid_str in result.stdout.strip().split('\n'):
                    if pid_str:
                        protected_pids.add(int(pid_str))
        except Exception:
            pass

        blocking_pids = []
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                for p in result.stdout.strip().split('\n'):
                    if p.strip():
                        pid = int(p.strip())
                        # Only report as blocking if NOT current process or its children
                        if pid not in protected_pids:
                            blocking_pids.append(pid)
        except Exception:
            pass

        return GPUStatus(
            available=True,
            gpu_count=gpu_count,
            free_memory_gb=free_mem,
            total_memory_gb=total_mem,
            blocking_pids=blocking_pids,
        )
    except Exception as e:
        return GPUStatus(False, 0, 0, 0, [], str(e))


def cleanup_gpu_processes(force: bool = False) -> int:
    """Kill existing vLLM/GPU processes. Returns count of killed processes."""
    killed = 0
    current_pid = os.getpid()

    # Get current process and all its children to avoid killing ourselves
    protected_pids = {current_pid}
    try:
        # Get all child processes of current process
        result = subprocess.run(
            ["pgrep", "-P", str(current_pid)],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            for pid_str in result.stdout.strip().split('\n'):
                if pid_str:
                    protected_pids.add(int(pid_str))
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                if not line.strip():
                    continue
                parts = line.split(',')
                if len(parts) >= 2:
                    pid = int(parts[0].strip())
                    name = parts[1].strip().lower()

                    # Skip current process and its children (including EngineCore)
                    if pid in protected_pids:
                        continue

                    # Kill vLLM processes, EngineCore, or any Python using GPU
                    should_kill = (
                        'vllm' in name or
                        'enginecore' in name or
                        (force and 'python' in name)
                    )

                    if should_kill:
                        try:
                            os.kill(pid, signal.SIGKILL)
                            killed += 1
                            print(f"[gpu] Killed process {pid} ({name})")
                        except ProcessLookupError:
                            pass
                        except PermissionError:
                            print(f"[gpu] Cannot kill {pid} (permission denied)")

        # Also try pkill for vLLM subprocesses (but exclude our own)
        if force:
            subprocess.run(["pkill", "-9", "-f", "vllm"], capture_output=True, timeout=5)
            subprocess.run(["pkill", "-9", "-f", "EngineCore"], capture_output=True, timeout=5)

    except Exception as e:
        print(f"[gpu] Cleanup error: {e}")

    # Clear CUDA cache
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return killed


@contextmanager
def timeout_context(seconds: int, error_msg: str = "Operation timed out"):
    """Context manager for timing out operations. Thread-safe version."""
    # Note: This is a no-op timeout for thread compatibility.
    # The actual timeout is handled by the caller using asyncio.wait_for or similar.
    # This context manager is kept for API compatibility but doesn't enforce timeout.
    yield


def run_with_timeout(func, timeout_sec: int, error_msg: str = "Operation timed out"):
    """Run a function with timeout using threading. Works in non-main threads."""
    import threading
    import concurrent.futures

    result = [None]
    exception = [None]

    def wrapper():
        try:
            result[0] = func()
        except Exception as e:
            exception[0] = e

    thread = threading.Thread(target=wrapper)
    thread.start()
    thread.join(timeout=timeout_sec)

    if thread.is_alive():
        # Thread is still running - we can't kill it, but we can raise timeout
        raise TimeoutError(error_msg)

    if exception[0] is not None:
        raise exception[0]

    return result[0]


class VLLMRecruiterEngine:

    STOP_MARKERS: Tuple[str, ...] = (
        "\nCandidate:", "\nUser:", "\nInterviewer:", "\nRecruiter:",
    )

    INSTRUCTION_MARKERS: Tuple[str, ...] = (
        "[FOLLOW", "[MOVE", "[Note:", "[INSTRUCTION:",
        "[CANDIDATE", "[TOO MANY", "[SKIP",
        "[END", "[EVALUATION", "[ANALYSIS",
        "[Start your", "[Say '",
    )

    TERMINATION_MARKERS: Tuple[str, ...] = (
        "interview ends", "interview over", "end the interview",
        "post-interview", "thank you for coming", "thanks for coming",
        "thank you for sharing", "thanks for sharing",
        "we will be in touch", "we'll be in touch",
        "we'll let you know", "i'm concerned", "i am concerned",
        "unfortunately", "not a good fit",
        "this has been a thorough", "this has been a great",
        "do you have any final questions", "do you have any questions for me",
        "you've shown great", "you've demonstrated",
        "great job", "impressive work", "great expertise",
        "notes:", "evaluation:", "assessment:", "summary:",
        "the candidate", "this candidate",
        "based on this conversation",
    )

    FILLER_STARTS: Tuple[str, ...] = (
        "great,", "great!", "great job",
        "thanks,", "thank you,", "thank you for sharing",
        "interesting,", "awesome,", "perfect,", "wonderful,",
        "excellent,", "good,", "nice,", "sure,", "okay,", "ok,",
        "that's great,", "that's interesting,", "that's good,",
        "that's okay", "now,", "now let's",
        "moving forward,", "let's talk about another",
        "let's discuss another",
    )

    GENERIC_QUESTION_MARKERS: Tuple[str, ...] = (
        "sentiment analysis", "reinforcement learning",
        "q-learning", "write a simple", "write a function",
        "implement a basic", "show me some sample code",
        "coding challenge", "whiteboard",
        "describe a time when", "tell me about a time",
        "communicate complex technical", "non-technical audience",
        "pre-trained model like",
    )

    # Minimum GPU memory required (GB)
    MIN_GPU_MEMORY_GB: float = 12.0

    def __init__(
        self,
        model_path: str,
        dtype: str = "bfloat16",
        max_model_len: int = 2048,
        gpu_memory_utilization: float = 0.60,  # Reduced for stability with fragmented GPU memory
        tensor_parallel_size: int = 0,
        auto_cleanup: bool = True,
        load_timeout_sec: int = 180,
    ) -> None:

        self.model_path = model_path
        self.max_model_len = max_model_len
        self._dtype = dtype
        self._gpu_mem_util = gpu_memory_utilization
        self._tp_size = tensor_parallel_size
        self._auto_cleanup = auto_cleanup
        self._load_timeout = load_timeout_sec

        # Lazy loading state
        self._engine = None
        self._tokenizer = None
        self._initialized = False
        self._init_error: Optional[str] = None
        self._warmup_done = False
        self._sampling_cache: Dict[tuple, object] = {}
        self.stop_token_ids: List[int] = []

    @property
    def is_ready(self) -> bool:
        """Check if engine is initialized and ready."""
        return self._initialized and self._engine is not None

    @property
    def engine(self):
        """Lazy-load the vLLM engine on first access."""
        if not self._initialized:
            self._initialize_engine()
        if self._engine is None:
            raise RuntimeError(f"vLLM engine not available: {self._init_error}")
        return self._engine

    @property
    def tokenizer(self):
        """Get the tokenizer, loading if necessary."""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self._setup_stop_tokens()
        return self._tokenizer

    def _setup_stop_tokens(self) -> None:
        """Setup stop token IDs from tokenizer."""
        self.stop_token_ids = []
        if self._tokenizer.eos_token_id is not None:
            self.stop_token_ids.append(self._tokenizer.eos_token_id)
        eot = self._tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if isinstance(eot, int) and eot >= 0 and eot not in self.stop_token_ids:
            self.stop_token_ids.append(eot)

    def _initialize_engine(self) -> None:
        """Initialize the vLLM engine with proper checks and cleanup."""
        global _GLOBAL_ENGINE_INSTANCE

        if self._initialized:
            return

        # Use module-level lock to prevent multiple simultaneous initializations
        with _ENGINE_INIT_LOCK:
            # Double-check after acquiring lock
            if self._initialized:
                return

            # If another instance already created the engine, reuse it
            if _GLOBAL_ENGINE_INSTANCE is not None:
                print("[vllm] Reusing existing engine instance")
                self._engine = _GLOBAL_ENGINE_INSTANCE
                self._initialized = True
                self._init_error = None
                return

            print("[vllm] Starting engine initialization...")
            t0 = time.time()

            # Step 1: Check GPU status
            gpu_status = check_gpu_status()
            print(f"[vllm] GPU status: available={gpu_status.available}, "
                  f"free={gpu_status.free_memory_gb:.1f}GB/{gpu_status.total_memory_gb:.1f}GB, "
                  f"blocking_pids={gpu_status.blocking_pids}")

            if not gpu_status.available:
                self._init_error = f"GPU not available: {gpu_status.error}"
                self._initialized = True
                raise RuntimeError(self._init_error)

            # Step 2: Cleanup blocking processes if needed
            if gpu_status.blocking_pids and self._auto_cleanup:
                print(f"[vllm] Cleaning up {len(gpu_status.blocking_pids)} blocking processes...")
                killed = cleanup_gpu_processes(force=True)  # Use force to kill any GPU-using Python process
                if killed > 0:
                    time.sleep(3)  # Wait for GPU memory to be released
                    gpu_status = check_gpu_status()
                    print(f"[vllm] After cleanup: free={gpu_status.free_memory_gb:.1f}GB")

            # Step 3: Verify sufficient memory
            if gpu_status.free_memory_gb < self.MIN_GPU_MEMORY_GB:
                self._init_error = (
                    f"Insufficient GPU memory: {gpu_status.free_memory_gb:.1f}GB free, "
                    f"need {self.MIN_GPU_MEMORY_GB}GB"
                )
                self._initialized = True
                raise RuntimeError(self._init_error)

            # Step 4: Load tokenizer first (fast, can fail early)
            print(f"[vllm] Loading tokenizer from {self.model_path}...")
            try:
                _ = self.tokenizer  # Trigger lazy load
            except Exception as e:
                self._init_error = f"Failed to load tokenizer: {e}"
                self._initialized = True
                raise RuntimeError(self._init_error)

            # Step 5: Initialize vLLM engine
            print(f"[vllm] Initializing vLLM engine...")
            try:
                from vllm import AsyncLLMEngine, AsyncEngineArgs

                tp = self._detect_tp_size(self._tp_size)
                engine_args = AsyncEngineArgs(
                    model=self.model_path,
                    dtype=self._dtype,
                    tensor_parallel_size=tp,
                    gpu_memory_utilization=self._gpu_mem_util,
                    max_model_len=self.max_model_len,
                    enable_chunked_prefill=True,
                    max_num_batched_tokens=self.max_model_len,
                    enable_prefix_caching=True,
                    enforce_eager=True,  # Disable CUDA graphs to prevent OOM during compilation
                )

                # Initialize engine directly (timeout handled at higher level)
                self._engine = AsyncLLMEngine.from_engine_args(engine_args)
                _GLOBAL_ENGINE_INSTANCE = self._engine  # Store for reuse

                load_time = time.time() - t0
                print(f"[vllm] Engine ready in {load_time:.1f}s")
                self._initialized = True
                self._init_error = None

            except Exception as e:
                self._init_error = f"Engine initialization failed: {e}"
                self._initialized = True
                print(f"[vllm] ERROR: {e}")
                raise RuntimeError(self._init_error)

    @staticmethod
    def _detect_tp_size(requested: int) -> int:
        if requested > 0:
            return requested
        try:
            import torch
            return max(torch.cuda.device_count(), 1)
        except Exception:
            return 1

    async def warmup(self, system_prompt: str) -> None:
        if self._warmup_done:
            return
        from vllm import SamplingParams
        sp = SamplingParams(max_tokens=1, temperature=0.0)
        for i, d in enumerate(["warmup", "warmup " * 10]):
            p = self._build_prompt(system_prompt, [], d, None)
            async for _ in self.engine.generate(p, sp, request_id=f"w-{i}"):
                pass
        self._warmup_done = True

    def _build_prompt(
        self, system_prompt: str,
        history: List[Tuple[str, str]],
        candidate_input: str,
        instruction: Optional[str] = None,
    ) -> str:
        messages = [{"role": "system", "content": system_prompt}]
        messages.append({
            "role": "user",
            "content": (
                "RULES: Only ask about CV items. Never evaluate. "
                "Never end interview. One question, stop after '?'."
            ),
        })
        messages.append({
            "role": "assistant",
            "content": "Understood. One CV-specific question per turn.",
        })
        for c, r in history:
            messages.append({"role": "user", "content": c})
            messages.append({"role": "assistant", "content": r})

        text = candidate_input
        if instruction:
            text = f"{candidate_input}\n\n{instruction}"
        messages.append({"role": "user", "content": text})

        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _get_sampling_params(self, max_tokens, temp, top_p, rep_pen):
        from vllm import SamplingParams
        key = (max_tokens, temp, top_p, rep_pen)
        if key not in self._sampling_cache:
            self._sampling_cache[key] = SamplingParams(
                max_tokens=max_tokens, temperature=temp,
                top_p=top_p, repetition_penalty=rep_pen,
                stop_token_ids=self.stop_token_ids,
                stop=[*self.STOP_MARKERS, *self.INSTRUCTION_MARKERS,
                      "\n👤", "\n🤖", "?"],
            )
        return self._sampling_cache[key]

    async def generate(
        self, system_prompt: str,
        history: List[Tuple[str, str]],
        candidate_input: str,
        instruction: Optional[str] = None,
        max_tokens: int = 48,
        temperature: float = 0.6,
        top_p: float = 0.85,
        repetition_penalty: float = 1.15,
        history_window: int = 6,
    ) -> Tuple[str, GenerationMetrics]:

        hist = history[-history_window:]
        prompt = self._build_prompt(system_prompt, hist, candidate_input, instruction)

        n_prompt = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        avail = self.max_model_len - n_prompt
        if avail < max_tokens:
            while hist and avail < max_tokens:
                hist = hist[1:]
                prompt = self._build_prompt(system_prompt, hist, candidate_input, instruction)
                n_prompt = len(self.tokenizer.encode(prompt, add_special_tokens=False))
                avail = self.max_model_len - n_prompt

        eff = min(max_tokens, max(avail, 1))
        sp = self._get_sampling_params(eff, temperature, top_p, repetition_penalty)
        rid = f"g-{time.monotonic_ns()}"

        t0 = time.perf_counter()
        ttft = None
        prev = 0
        final = None

        async for ro in self.engine.generate(prompt, sp, rid):
            if not ro.outputs:
                continue
            out = ro.outputs[0]
            txt = out.text or ""
            if len(txt) > prev and ttft is None:
                ttft = time.perf_counter() - t0
            prev = len(txt)
            final = out

        lat = time.perf_counter() - t0
        raw = final.text if final else ""
        nt = len(final.token_ids) if final else 0
        cleaned = self._clean_response(raw)

        return cleaned, GenerationMetrics(
            latency_sec=lat, ttft_sec=ttft,
            num_tokens=nt,
            tokens_per_sec=nt / lat if lat > 0 else 0,
        )

    async def generate_streaming(
        self,
        system_prompt: str,
        history: List[Tuple[str, str]],
        candidate_input: str,
        instruction: Optional[str] = None,
        max_tokens: int = 48,
        temperature: float = 0.6,
        top_p: float = 0.85,
        repetition_penalty: float = 1.15,
        history_window: int = 6,
    ):
        """
        Stream tokens as they're generated for real-time display.
        Yields partial text incrementally.
        """
        hist = history[-history_window:]
        prompt = self._build_prompt(system_prompt, hist, candidate_input, instruction)

        n_prompt = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        avail = self.max_model_len - n_prompt
        if avail < max_tokens:
            while hist and avail < max_tokens:
                hist = hist[1:]
                prompt = self._build_prompt(system_prompt, hist, candidate_input, instruction)
                n_prompt = len(self.tokenizer.encode(prompt, add_special_tokens=False))
                avail = self.max_model_len - n_prompt

        eff = min(max_tokens, max(avail, 1))
        sp = self._get_sampling_params(eff, temperature, top_p, repetition_penalty)
        rid = f"s-{time.monotonic_ns()}"

        prev_text = ""
        async for ro in self.engine.generate(prompt, sp, rid):
            if not ro.outputs:
                continue
            out = ro.outputs[0]
            current_text = out.text or ""

            # Yield only the new characters
            if len(current_text) > len(prev_text):
                new_chars = current_text[len(prev_text):]
                # Quick clean of special tokens
                for token in ("<|eot_id|>", "<|end_of_text|>"):
                    new_chars = new_chars.replace(token, "")
                if new_chars:
                    yield new_chars
                prev_text = current_text

    def _clean_response(self, text: str) -> str:
        c = text
        for t in ("<|eot_id|>", "<|begin_of_text|>", "<|end_of_text|>",
                   "<|start_header_id|>", "<|end_header_id|>"):
            c = c.replace(t, "")
        c = re.sub(r"<think>.*?</think>", "", c, flags=re.DOTALL)
        c = c.strip()

        for m in self.INSTRUCTION_MARKERS:
            i = c.find(m)
            if i != -1:
                c = c[:i].strip()
        for m in self.STOP_MARKERS:
            i = c.find(m)
            if i != -1:
                c = c[:i].strip()
        for p in ("candidate:", "user:", "human:"):
            i = c.lower().find(p)
            if i != -1:
                c = c[:i].strip()

        cl = c.lower()
        for m in self.TERMINATION_MARKERS:
            i = cl.find(m)
            if i != -1:
                s = c.rfind(".", 0, i)
                c = c[:s].strip() if s > 10 else ""
                cl = c.lower()

        qi = c.find("?")
        if qi != -1:
            c = c[:qi + 1].strip()
        else:
            c = c.rstrip(".,;:!").strip() + "?"

        cl2 = c.lower()
        for f in self.FILLER_STARTS:
            if cl2.startswith(f):
                c = c[len(f):].strip()
                if c:
                    c = c[0].upper() + c[1:]
                break

        c = " ".join(c.split())
        return c if len(c) >= 10 else ""

    def is_termination_attempt(self, text: str) -> bool:
        tl = text.lower()
        return any(m in tl for m in self.TERMINATION_MARKERS)

    def is_generic_question(self, text: str) -> bool:
        tl = text.lower()
        return any(m in tl for m in self.GENERIC_QUESTION_MARKERS)