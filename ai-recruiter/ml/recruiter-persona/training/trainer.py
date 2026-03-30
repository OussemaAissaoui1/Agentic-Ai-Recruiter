"""
Custom Trainer with Interview-Specific Callbacks
Provides specialized training monitoring and callbacks for recruiter persona fine-tuning
"""

from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import os
import json
from typing import Dict, List
import time
import torch


class InterviewMetricsCallback(TrainerCallback):
    """
    Callback to track interview-specific metrics during training.
    Monitors conversation quality and persona consistency.
    """

    def __init__(self):
        super().__init__()
        self.start_time = None
        self.step_times = []

    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training."""
        self.start_time = time.time()
        print("\n" + "=" * 70)
        print("🚀 TRAINING STARTED")
        print("=" * 70)
        print(f"Total training steps: {state.max_steps}")
        print(f"Logging every {args.logging_steps} steps")
        print(f"Evaluating every {args.eval_steps} steps")
        print(f"Saving every {args.save_steps} steps")
        print("=" * 70 + "\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called after logging."""
        if logs is not None:
            # Track time per step
            if "loss" in logs and state.global_step > 0:
                elapsed = time.time() - self.start_time
                steps_per_sec = state.global_step / elapsed
                remaining_steps = state.max_steps - state.global_step
                eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
                eta_hours = eta_seconds / 3600

                # Add custom metrics to logs
                logs["steps_per_sec"] = round(steps_per_sec, 3)
                logs["eta_hours"] = round(eta_hours, 2)

                # Print progress
                if state.global_step % (args.logging_steps * 5) == 0:
                    print(f"\n📊 Progress Update (Step {state.global_step}/{state.max_steps}):")
                    print(f"   Loss: {logs.get('loss', 'N/A'):.4f}")
                    print(f"   Learning Rate: {logs.get('learning_rate', 'N/A'):.2e}")
                    print(f"   Speed: {steps_per_sec:.2f} steps/sec")
                    print(f"   ETA: {eta_hours:.1f} hours\n")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called after evaluation."""
        if metrics is not None:
            print("\n" + "=" * 70)
            print(f"📈 EVALUATION RESULTS (Step {state.global_step})")
            print("=" * 70)
            print(f"Eval Loss: {metrics.get('eval_loss', 'N/A'):.4f}")
            print(f"Eval Runtime: {metrics.get('eval_runtime', 'N/A'):.2f}s")
            print(f"Eval Samples/Sec: {metrics.get('eval_samples_per_second', 'N/A'):.2f}")
            print("=" * 70 + "\n")

    def on_save(self, args, state, control, **kwargs):
        """Called after saving a checkpoint."""
        print(f"\n💾 Checkpoint saved at step {state.global_step}")

    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        elapsed = time.time() - self.start_time
        elapsed_hours = elapsed / 3600

        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETED")
        print("=" * 70)
        print(f"Total time: {elapsed_hours:.2f} hours")
        print(f"Total steps: {state.global_step}")
        print(f"Average speed: {state.global_step / elapsed:.2f} steps/sec")
        print("=" * 70 + "\n")


class EarlyStoppingCallback(TrainerCallback):
    """
    Early stopping callback to stop training when eval loss stops improving.
    Helps prevent overfitting on the recruiter conversation data.
    """
    def __init__(self, early_stopping_patience: int = 3, early_stopping_threshold: float = 0.0):
        """
        Initialize early stopping.

        Args:
            early_stopping_patience: Number of evaluations with no improvement before stopping
            early_stopping_threshold: Minimum change to consider as improvement
        """
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.best_metric = None
        self.patience_counter = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Check if we should stop training."""
        if metrics is None:
            return

        current_metric = metrics.get("eval_loss")
        if current_metric is None:
            return

        # First evaluation
        if self.best_metric is None:
            self.best_metric = current_metric
            print(f"🎯 Initial eval loss: {current_metric:.4f}")
            return

        # Check if improved
        if current_metric < (self.best_metric - self.early_stopping_threshold):
            self.best_metric = current_metric
            self.patience_counter = 0
            print(f"✅ Eval loss improved to {current_metric:.4f}")
        else:
            self.patience_counter += 1
            print(f"⚠️  No improvement. Patience: {self.patience_counter}/{self.early_stopping_patience}")

            if self.patience_counter >= self.early_stopping_patience:
                print(f"\n🛑 Early stopping triggered! Best eval loss: {self.best_metric:.4f}")
                control.should_training_stop = True


class MemoryMonitorCallback(TrainerCallback):
    """
    Monitor GPU memory usage during training.
    Useful for debugging OOM issues on L40 24GB.
    """

    def __init__(self, log_every_n_steps: int = 100):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    def on_step_end(self, args, state, control, **kwargs):
        """Log memory usage periodically."""
        if state.global_step % self.log_every_n_steps == 0:
            try:
                import torch
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                    reserved = torch.cuda.memory_reserved() / 1024**3    # GB
                    max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB

                    print(f"\n💾 GPU Memory (Step {state.global_step}):")
                    print(f"   Allocated: {allocated:.2f} GB")
                    print(f"   Reserved: {reserved:.2f} GB")
                    print(f"   Peak: {max_allocated:.2f} GB")
            except Exception as e:
                # Silently fail if CUDA not available
                pass


class SampleGenerationCallback(TrainerCallback):
    """
    Periodically generate sample conversations to monitor quality during training.
    Helps ensure the model maintains recruiter persona.
    """

    def __init__(
        self,
        tokenizer,
        generation_steps: int = 500,
        max_new_tokens: int = 150,
    ):
        """
        Initialize sample generation callback.

        Args:
            tokenizer: HuggingFace tokenizer
            generation_steps: Generate samples every N steps
            max_new_tokens: Max tokens to generate per sample
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.generation_steps = generation_steps
        self.max_new_tokens = max_new_tokens

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        """Generate sample after each evaluation."""
        if model is None or state.global_step == 0:
            return

        # Only generate every N evaluations
        if state.global_step % self.generation_steps != 0:
            return

        print("\n" + "=" * 70)
        print(f"🎤 SAMPLE GENERATION (Step {state.global_step})")
        print("=" * 70)

        # Test prompt
        messages = [
            {
                "role": "system",
                "content": "You are Alex, a senior recruiter conducting a structured and professional interview. Ask one question at a time, adapt follow-up questions to the candidate's last answer, and keep the tone concise, respectful, and role-relevant. Avoid discriminatory or illegal questions."
            },
            {
                "role": "assistant",
                "content": "Good morning! Thank you for coming in today. Can you start by telling me about your background?"
            },
            {
                "role": "user",
                "content": "I have 5 years of experience in software engineering, primarily working with Python and React."
            }
        ]

        try:
            # Format prompt
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)

            # Generate
            model.eval()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # Decode
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated[len(prompt):].strip()

            print("Candidate: I have 5 years of experience in software engineering...")
            print(f"\nRecruiter: {response}")
            print("=" * 70 + "\n")

        except Exception as e:
            print(f"❌ Generation failed: {e}")
            print("=" * 70 + "\n")

        model.train()
        

def get_default_callbacks(
    tokenizer,
    enable_early_stopping: bool = True,
    enable_memory_monitor: bool = True,
    enable_sample_generation: bool = True,
    early_stopping_patience: int = 3,
    early_stopping_threshold: float = 0.002,
):
    """
    Get default callbacks for recruiter persona training.

    Args:
        tokenizer: HuggingFace tokenizer
        enable_early_stopping: Enable early stopping (recommended)
        enable_memory_monitor: Monitor GPU memory
        enable_sample_generation: Generate samples during training

    Returns:
        List of callbacks
    """
    callbacks = [InterviewMetricsCallback()]

    if enable_early_stopping:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=early_stopping_threshold,
            )
        )

    if enable_memory_monitor and torch.cuda.is_available():
        callbacks.append(MemoryMonitorCallback(log_every_n_steps=100))

    if enable_sample_generation:
        callbacks.append(SampleGenerationCallback(
            tokenizer=tokenizer,
            generation_steps=500,
            max_new_tokens=150,
        ))

    return callbacks


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Custom Trainer Callbacks for Recruiter Persona")
    print("=" * 70)
    print("\nAvailable Callbacks:")
    print("  1. InterviewMetricsCallback - Track progress and ETA")
    print("  2. EarlyStoppingCallback - Prevent overfitting")
    print("  3. MemoryMonitorCallback - Monitor GPU memory")
    print("  4. SampleGenerationCallback - Generate samples during training")
    print("\nUse get_default_callbacks() to get all recommended callbacks")
