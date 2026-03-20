#!/usr/bin/env python3
"""
Inference tester for the fine-tuned recruiter persona model.

Supports:
- Interactive interview chat
- Batch scenario testing with JSON export
- Quality spot-check conversation
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class RecruiterInferenceTester:
    """Inference tester for a LoRA/QLoRA fine-tuned recruiter model."""

    def __init__(
        self,
        base_model_path: Optional[str],
        lora_weights_path: str,
        backend: str = "transformers",
        load_in_4bit: bool = True,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.92,
        max_model_len: int = 4096,
        max_input_tokens: int = 3072,
    ):
        """
        Initialize the tester.

        Args:
            base_model_path: Base model HuggingFace ID. If None, inferred from adapter config.
            lora_weights_path: Path to LoRA adapter weights
            backend: Inference backend: "transformers" or "vllm"
            load_in_4bit: Whether to load base model in 4-bit quantized mode
            dtype: Compute dtype: "bfloat16" or "float16"
            gpu_memory_utilization: vLLM GPU utilization target
            max_model_len: Maximum model context length (mainly for vLLM)
            max_input_tokens: Input truncation budget for lower latency
        """
        self.base_model_path = base_model_path
        self.lora_weights_path = lora_weights_path
        self.backend = backend
        self.load_in_4bit = load_in_4bit
        self.dtype = dtype
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.max_input_tokens = max_input_tokens
        self.model = None
        self.tokenizer = None
        self.eos_token_ids: List[int] = []
        self.vllm_llm = None
        self.vllm_lora_request = None

    def _infer_base_model_from_adapter(self) -> Optional[str]:
        """Infer base model from adapter_config.json if available."""
        adapter_config_path = Path(self.lora_weights_path) / "adapter_config.json"
        if not adapter_config_path.exists():
            return None

        try:
            with open(adapter_config_path, "r", encoding="utf-8") as f:
                adapter_cfg = json.load(f)
            return adapter_cfg.get("base_model_name_or_path")
        except Exception:
            return None

    def load_model(self):
        """Load model and tokenizer."""
        print("\n" + "=" * 70)
        print("Loading model...")
        print("=" * 70)

        if not self.base_model_path:
            self.base_model_path = self._infer_base_model_from_adapter()

        if not self.base_model_path:
            raise ValueError(
                "Base model could not be inferred from adapter_config.json. "
                "Pass --base-model explicitly."
            )

        print(f"Base model: {self.base_model_path}")
        print(f"LoRA weights: {self.lora_weights_path}")
        print(f"Backend: {self.backend}")
        print(f"Load in 4-bit: {self.load_in_4bit}")
        print(f"Dtype: {self.dtype}")

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Load tokenizer
        print("\n1/3 Loading tokenizer...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.lora_weights_path)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"

        eos_ids = []
        if self.tokenizer.eos_token_id is not None:
            eos_ids.append(self.tokenizer.eos_token_id)
        eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if isinstance(eot_id, int) and eot_id >= 0 and eot_id not in eos_ids:
            eos_ids.append(eot_id)
        self.eos_token_ids = eos_ids

        if self.backend == "vllm":
            print("2/3 Initializing vLLM engine...")
            self._load_vllm_model()
            print("3/3 Attaching LoRA adapter (vLLM)...")
        else:
            print("2/3 Loading base model...")
            self._load_transformers_model()
            print("3/3 Loading LoRA adapter...")
            self.model = PeftModel.from_pretrained(self.model, self.lora_weights_path)
            self.model.eval()

        print("\n✅ Model loaded successfully!")
        print("=" * 70 + "\n")

    def _load_transformers_model(self):
        """Load model with Transformers + PEFT path."""
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }
        torch_dtype = dtype_map.get(self.dtype, torch.bfloat16)

        model_kwargs: Dict[str, Any] = {
            "device_map": "auto",
            "dtype": torch_dtype,
        }

        if self.load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            **model_kwargs,
        )

    def _load_vllm_model(self):
        """Load model with vLLM backend and LoRA enabled."""
        try:
            from vllm import LLM
            from vllm.lora.request import LoRARequest
        except ImportError as exc:
            raise RuntimeError(
                "vLLM is not installed. Install it with: pip install vllm"
            ) from exc

        self.vllm_llm = LLM(
            model=self.base_model_path,
            dtype=self.dtype,
            tensor_parallel_size=1,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            enable_lora=True,
            max_lora_rank=64,
        )
        self.vllm_lora_request = LoRARequest(
            lora_name="recruiter_persona_adapter",
            lora_int_id=1,
            lora_path=self.lora_weights_path,
        )

    def _clean_response(self, raw_text: str) -> str:
        """Trim generated content to one clean assistant turn."""
        text = raw_text.strip()
        if not text:
            return "Could you tell me more about your relevant experience for this role?"

        # Stop at next header/speaker marker if model keeps generating dialogue.
        cut_markers = [
            "<|eot_id|>",
            "<|start_header_id|>",
            "\nCandidate:",
            "\nUser:",
            "\nRecruiter:",
            "\n👤",
            "\n🤖",
        ]
        for marker in cut_markers:
            idx = text.find(marker)
            if idx != -1:
                text = text[:idx].strip()

        # Keep only one recruiter question to prevent long multi-turn continuations.
        q_idx = text.find("?")
        if q_idx != -1:
            text = text[: q_idx + 1].strip()

        if not text:
            return "Could you share an example of a recent project you are most proud of?"
        return text

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate response from model.

        Args:
            messages: List of message dictionaries
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Generated response string
        """
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        truncated_prompt = self.tokenizer.decode(
            encoded["input_ids"][0],
            skip_special_tokens=False,
        )

        if self.backend == "vllm":
            raw_response = self._generate_with_vllm(
                prompt=truncated_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            return self._clean_response(raw_response)

        # Tokenize
        inputs = {k: v.to(self.model.device) for k, v in encoded.items()}
        prompt_tokens = inputs["input_ids"].shape[-1]

        # Generate
        with torch.inference_mode():
            do_sample = temperature > 0.0
            eos_for_generate: Any = self.eos_token_ids[0] if len(self.eos_token_ids) == 1 else self.eos_token_ids
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                repetition_penalty=1.15,
                no_repeat_ngram_size=4,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=eos_for_generate,
            )

        # Decode only generated completion (avoid brittle string slicing)
        new_tokens = outputs[0][prompt_tokens:]
        raw_response = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
        response = self._clean_response(raw_response)
        return response

    def _generate_with_vllm(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        """Generate text with vLLM backend."""
        from vllm import SamplingParams

        if self.vllm_llm is None:
            raise RuntimeError("vLLM backend is not initialized.")

        do_sample = temperature > 0.0
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature if do_sample else 0.0,
            top_p=top_p if do_sample else 1.0,
            repetition_penalty=1.15,
            stop_token_ids=self.eos_token_ids if self.eos_token_ids else None,
        )

        outputs = self.vllm_llm.generate(
            [prompt],
            sampling_params=sampling_params,
            lora_request=self.vllm_lora_request,
        )

        if not outputs or not outputs[0].outputs:
            return ""
        return outputs[0].outputs[0].text

    def interactive_mode(
        self,
        max_new_tokens: int = 128,
        temperature: float = 0.4,
        top_p: float = 0.9,
    ):
        """Run interactive interview simulation."""
        print("\n" + "=" * 70)
        print("🎤 INTERACTIVE INTERVIEW MODE")
        print("=" * 70)
        print("\nYou are now in an interview with Alex, the recruiter.")
        print("Type your responses, and Alex will ask follow-up questions.")
        print("\nCommands:")
        print("  - Type 'quit' or 'exit' to end the interview")
        print("  - Type 'restart' to start a new interview")
        print("=" * 70 + "\n")

        self._run_interview(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    def _run_interview(
        self,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ):
        """Run a single interview session."""
        # Initialize conversation
        messages = [
            {
                "role": "system",
                "content": "You are Alex, a senior recruiter conducting a structured and professional interview. Ask one question at a time, adapt follow-up questions to the candidate's last answer, and keep the tone concise, respectful, and role-relevant. Avoid discriminatory or illegal questions."
            }
        ]

        # Generate first question
        print("🤖 Alex (Recruiter):")
        first_question = self.generate_response(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        print(f"   {first_question}\n")
        messages.append({"role": "assistant", "content": first_question})

        # Conversation loop
        turn = 1
        while True:
            # Get candidate response
            print(f"👤 You (Candidate):")
            try:
                candidate_response = input("   > ").strip()
            except EOFError:
                break

            # Handle commands
            if candidate_response.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Interview ended. Thank you!")
                break
            elif candidate_response.lower() == 'restart':
                print("\n🔄 Starting new interview...\n")
                self._run_interview(
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                return

            if not candidate_response:
                print("   (Please provide a response)\n")
                continue

            # Add to conversation
            messages.append({"role": "user", "content": candidate_response})

            # Generate recruiter response
            print(f"\n🤖 Alex (Recruiter):")
            recruiter_response = self.generate_response(
                messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            print(f"   {recruiter_response}\n")

            # Add to conversation
            messages.append({"role": "assistant", "content": recruiter_response})

            turn += 1
            print("-" * 70 + "\n")

    def batch_test_mode(
        self,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> List[Dict[str, Any]]:
        """Run batch testing with predefined scenarios and return collected results."""
        print("\n" + "=" * 70)
        print("🧪 BATCH TEST MODE")
        print("=" * 70 + "\n")

        test_cases = [
            {
                "role": "Software Engineer",
                "background": "I have 5 years of experience in full-stack development, primarily working with React, Node.js, and PostgreSQL. I'm interested in this role because I want to work on more challenging problems and scale my impact.",
            },
            {
                "role": "Data Scientist",
                "background": "I've been working in data science for 3 years, focusing on machine learning and predictive analytics. Most recently, I built recommendation systems at a fintech startup. I'm excited about this opportunity because of your company's work in AI and the chance to work with larger datasets.",
            },
            {
                "role": "Product Manager",
                "background": "I have 7 years of experience leading product teams in SaaS companies. I've successfully launched several B2B products and led cross-functional teams of 15+ people. I'm interested in this role because I want to work on products that have real impact at scale.",
            },
            {
                "role": "UX Designer",
                "background": "I'm a UX designer with 4 years of experience in both agency and in-house roles. I specialize in user research and interaction design. I've led design projects for mobile apps and web applications. I'm interested in this role because I admire your company's design philosophy.",
            },
        ]

        results: List[Dict[str, Any]] = []

        for i, test in enumerate(test_cases, 1):
            print(f"Test Case {i}/{len(test_cases)}: {test['role']}")
            print("-" * 70)

            messages = [
                {
                    "role": "system",
                    "content": "You are Alex, a senior recruiter conducting a structured and professional interview. Ask one question at a time, adapt follow-up questions to the candidate's last answer, and keep the tone concise, respectful, and role-relevant. Avoid discriminatory or illegal questions."
                },
                {
                    "role": "assistant",
                    "content": f"Good morning! Thank you for coming in today. Can you start by telling me about your background and why you're interested in this {test['role']} position?"
                },
                {
                    "role": "user",
                    "content": test['background']
                }
            ]

            print(f"\n👤 Candidate Response:")
            print(f"   {test['background']}\n")

            print("🤖 Alex (Follow-up):")
            response = self.generate_response(
                messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            print(f"   {response}\n")

            results.append(
                {
                    "case_id": i,
                    "role": test["role"],
                    "candidate_background": test["background"],
                    "recruiter_follow_up": response,
                }
            )

            print("=" * 70 + "\n")

        return results

    def quality_check(
        self,
        max_new_tokens: int = 150,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Dict[str, Any]:
        """Run quality checks on model outputs and return generated conversation."""
        print("\n" + "=" * 70)
        print("✅ QUALITY CHECK MODE")
        print("=" * 70 + "\n")

        criteria = [
            "Asks one question at a time",
            "Follows up based on candidate's answer",
            "Maintains professional tone",
            "Avoids discriminatory questions",
            "Stays in recruiter persona",
        ]

        print("Quality Criteria:")
        for i, criterion in enumerate(criteria, 1):
            print(f"  {i}. {criterion}")

        print("\n" + "-" * 70)
        print("Sample Interview for Manual Review:")
        print("-" * 70 + "\n")

        # Run sample interview
        messages = [
            {
                "role": "system",
                "content": "You are Alex, a senior recruiter conducting a structured and professional interview. Ask one question at a time, adapt follow-up questions to the candidate's last answer, and keep the tone concise, respectful, and role-relevant. Avoid discriminatory or illegal questions."
            }
        ]

        # First exchange
        print("🤖 Alex (Opening):")
        q1 = self.generate_response(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        print(f"   {q1}\n")
        messages.append({"role": "assistant", "content": q1})

        print("👤 Candidate:")
        r1 = "I have 6 years of experience in machine learning and AI. I've worked at both startups and large tech companies."
        print(f"   {r1}\n")
        messages.append({"role": "user", "content": r1})

        # Second exchange
        print("🤖 Alex (Follow-up #1):")
        q2 = self.generate_response(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        print(f"   {q2}\n")
        messages.append({"role": "assistant", "content": q2})

        print("👤 Candidate:")
        r2 = "At my last company, I built a recommendation system that increased engagement by 30%. I used collaborative filtering and deep learning."
        print(f"   {r2}\n")
        messages.append({"role": "user", "content": r2})

        # Third exchange
        print("🤖 Alex (Follow-up #2):")
        q3 = self.generate_response(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        print(f"   {q3}\n")

        print("\n" + "=" * 70)
        print("Manual Quality Assessment:")
        print("Review the above conversation and check:")
        for i, criterion in enumerate(criteria, 1):
            print(f"  [ ] {criterion}")
        print("=" * 70 + "\n")

        return {
            "criteria": criteria,
            "conversation": [
                {"speaker": "assistant", "text": q1},
                {"speaker": "user", "text": r1},
                {"speaker": "assistant", "text": q2},
                {"speaker": "user", "text": r2},
                {"speaker": "assistant", "text": q3},
            ],
        }


def save_json(path: str, payload: Any):
    """Save JSON payload to disk."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved results to: {target}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test fine-tuned recruiter persona model")
    parser.add_argument(
        "--backend",
        choices=["transformers", "vllm"],
        default="transformers",
        help="Inference backend to use",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Base model HuggingFace ID (optional; inferred from adapter config when omitted)",
    )
    parser.add_argument(
        "--lora-weights",
        default="/teamspace/studios/this_studio/PFE/ai-recruiter/ml/recruiter-persona/training/output/recruiter-persona-llama-3.1-8b/final",
        help="Path to LoRA adapter weights",
    )
    parser.add_argument(
        "--mode",
        choices=["interactive", "batch", "quality"],
        help="Testing mode (if not specified, will prompt)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate per response",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.4,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling top-p",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        default=True,
        help="Load base model in 4-bit mode (recommended for single-GPU inference)",
    )
    parser.add_argument(
        "--no-load-in-4bit",
        dest="load_in_4bit",
        action="store_false",
        help="Disable 4-bit loading",
    )
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16"],
        default="bfloat16",
        help="Compute dtype",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=3072,
        help="Input token truncation budget (lower = faster turns)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum model context length (used by vLLM)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.92,
        help="vLLM GPU memory utilization target",
    )
    parser.add_argument(
        "--optimize-latency",
        action="store_true",
        help="Apply low-latency defaults for interactive inference",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Optional output JSON path for batch/quality mode",
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("🎯 Recruiter Persona Model - Inference Testing")
    print("=" * 70)

    # Check if LoRA weights exist
    lora_path = Path(args.lora_weights)
    if not lora_path.exists():
        print(f"\n❌ Error: LoRA weights not found at: {args.lora_weights}")
        print(f"\nPlease train the model first or update the path.")
        sys.exit(1)

    if args.optimize_latency:
        args.max_new_tokens = min(args.max_new_tokens, 96)
        args.temperature = min(args.temperature, 0.3)
        args.max_input_tokens = min(args.max_input_tokens, 2048)
        print("\n⚡ Latency optimization profile enabled:")
        print(f"   max_new_tokens={args.max_new_tokens}")
        print(f"   temperature={args.temperature}")
        print(f"   max_input_tokens={args.max_input_tokens}")
        if args.backend == "transformers" and args.load_in_4bit:
            print("   note: 4-bit is memory-efficient but can be slower; use --no-load-in-4bit for max speed if VRAM allows")

    # Initialize tester
    tester = RecruiterInferenceTester(
        base_model_path=args.base_model,
        lora_weights_path=args.lora_weights,
        backend=args.backend,
        load_in_4bit=args.load_in_4bit,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_input_tokens=args.max_input_tokens,
    )
    tester.load_model()

    # Select mode
    mode = args.mode
    if not mode:
        print("\n📋 Select Testing Mode:")
        print("  1. Interactive - Chat with the model")
        print("  2. Batch Test - Run predefined test cases")
        print("  3. Quality Check - Manual quality assessment")
        print("  4. Exit")

        choice = input("\nSelect option (1-4): ").strip()
        mode_map = {"1": "interactive", "2": "batch", "3": "quality", "4": "exit"}
        mode = mode_map.get(choice, "exit")

    # Run selected mode
    if mode == "interactive":
        tester.interactive_mode(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    elif mode == "batch":
        batch_results = tester.batch_test_mode(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        if args.output_file:
            save_json(args.output_file, batch_results)
    elif mode == "quality":
        quality_results = tester.quality_check(
            max_new_tokens=min(args.max_new_tokens, 200),
            temperature=args.temperature,
            top_p=args.top_p,
        )
        if args.output_file:
            save_json(args.output_file, quality_results)
    else:
        print("\nGoodbye!")
        sys.exit(0)
if __name__ == "__main__":
    main()
