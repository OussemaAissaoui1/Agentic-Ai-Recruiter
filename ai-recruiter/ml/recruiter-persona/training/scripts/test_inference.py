#!/usr/bin/env python3
"""
Test inference script for fine-tuned recruiter persona model
Interactive and batch testing modes
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sys
from pathlib import Path


class RecruiterInferenceTester:
    """Test fine-tuned recruiter persona model"""

    def __init__(self, base_model_path: str, lora_weights_path: str):
        """
        Initialize the tester.

        Args:
            base_model_path: Base model HuggingFace ID
            lora_weights_path: Path to LoRA adapter weights
        """
        self.base_model_path = base_model_path
        self.lora_weights_path = lora_weights_path
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load model and tokenizer."""
        print("\n" + "=" * 70)
        print("Loading model...")
        print("=" * 70)

        print(f"Base model: {self.base_model_path}")
        print(f"LoRA weights: {self.lora_weights_path}")

        # Load tokenizer
        print("\n1/3 Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        print("2/3 Loading base model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        # Load LoRA weights
        print("3/3 Loading LoRA adapter...")
        self.model = PeftModel.from_pretrained(self.model, self.lora_weights_path)

        self.model.eval()

        print("\n✅ Model loaded successfully!")
        print("=" * 70 + "\n")

    def generate_response(
        self,
        messages: list,
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
            add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the new response
        response = generated_text[len(prompt):].strip()
        return response

    def interactive_mode(self):
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

        self._run_interview()

    def _run_interview(self):
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
        first_question = self.generate_response(messages, max_new_tokens=150)
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
                self._run_interview()
                return

            if not candidate_response:
                print("   (Please provide a response)\n")
                continue

            # Add to conversation
            messages.append({"role": "user", "content": candidate_response})

            # Generate recruiter response
            print(f"\n🤖 Alex (Recruiter):")
            recruiter_response = self.generate_response(messages, max_new_tokens=200)
            print(f"   {recruiter_response}\n")

            # Add to conversation
            messages.append({"role": "assistant", "content": recruiter_response})

            turn += 1
            print("-" * 70 + "\n")

    def batch_test_mode(self):
        """Run batch testing with predefined scenarios."""
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
            response = self.generate_response(messages, max_new_tokens=200)
            print(f"   {response}\n")

            print("=" * 70 + "\n")

    def quality_check(self):
        """Run quality checks on model outputs."""
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
        q1 = self.generate_response(messages, max_new_tokens=150)
        print(f"   {q1}\n")
        messages.append({"role": "assistant", "content": q1})

        print("👤 Candidate:")
        r1 = "I have 6 years of experience in machine learning and AI. I've worked at both startups and large tech companies."
        print(f"   {r1}\n")
        messages.append({"role": "user", "content": r1})

        # Second exchange
        print("🤖 Alex (Follow-up #1):")
        q2 = self.generate_response(messages, max_new_tokens=150)
        print(f"   {q2}\n")
        messages.append({"role": "assistant", "content": q2})

        print("👤 Candidate:")
        r2 = "At my last company, I built a recommendation system that increased engagement by 30%. I used collaborative filtering and deep learning."
        print(f"   {r2}\n")
        messages.append({"role": "user", "content": r2})

        # Third exchange
        print("🤖 Alex (Follow-up #2):")
        q3 = self.generate_response(messages, max_new_tokens=150)
        print(f"   {q3}\n")

        print("\n" + "=" * 70)
        print("Manual Quality Assessment:")
        print("Review the above conversation and check:")
        for i, criterion in enumerate(criteria, 1):
            print(f"  [ ] {criterion}")
        print("=" * 70 + "\n")


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Test fine-tuned recruiter persona model")
    parser.add_argument(
        "--base-model",
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Base model HuggingFace ID"
    )
    parser.add_argument(
        "--lora-weights",
        default="../output/recruiter-persona-llama-3.1-8b/final",
        help="Path to LoRA adapter weights"
    )
    parser.add_argument(
        "--mode",
        choices=["interactive", "batch", "quality"],
        help="Testing mode (if not specified, will prompt)"
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

    # Initialize tester
    tester = RecruiterInferenceTester(args.base_model, args.lora_weights)
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
        tester.interactive_mode()
    elif mode == "batch":
        tester.batch_test_mode()
    elif mode == "quality":
        tester.quality_check()
    else:
        print("\nGoodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
