#!/usr/bin/env python3
"""
Main Training Script for Recruiter Persona Fine-tuning
Orchestrates the complete training pipeline using QLoRA on Lightning AI L40 24GB
"""

import os
import sys
import yaml
import torch
import wandb
from pathlib import Path
from huggingface_hub import login

# Accept either env var — agents in this repo use HF_TOKEN, the HF docs use
# HUGGINGFACE_TOKEN. Llama-3.1-8B-Instruct is gated so this must work.
hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from trl import DataCollatorForCompletionOnlyLM

# Import custom modules
from lora_config import get_qlora_config, get_lora_config, print_trainable_parameters
from data_loader import RecruiterDataLoader
from trainer import get_default_callbacks
from metrics import (
    compute_metrics,
    preprocess_logits_for_metrics,
    TrainEvalGapCallback,
)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load training configuration from YAML file."""
    print(f"Loading configuration from: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_tokenizer(model_name: str):
    """Setup tokenizer with proper padding configuration.

    Llama-3.x ships a dedicated `<|finetune_right_pad_id|>` token (id 128004).
    Using it instead of eos_token keeps padding and end-of-sequence distinct,
    so the model doesn't confuse "more padding" with "stop generating" at
    inference time.
    """
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    finetune_pad = tokenizer.convert_tokens_to_ids("<|finetune_right_pad_id|>")
    if isinstance(finetune_pad, int) and finetune_pad >= 0:
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
        tokenizer.pad_token_id = finetune_pad
    else:
        # Llama-2 / other models without the dedicated pad token
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def setup_model(config: dict, tokenizer):
    """Load base model + LoRA adapter according to the configured method.

    method=qlora → 4-bit nf4 quantized base + bf16 compute (sub-32GB GPUs)
    method=lora  → bf16 base, no quantization (recommended on ≥40GB GPUs)
    """
    model_name = config["base_model"]
    method = config.get("method", "qlora")

    print(f"\nLoading model: {model_name}")
    print(f"Training method: {method}")

    if method == "qlora":
        bnb_config = get_qlora_config()
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        # Prepare for k-bit training (input/output embedding fp32 cast,
        # disables some inplace ops that fight with quantized layers).
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=config["training"].get("gradient_checkpointing", True)
        )
    elif method == "lora":
        # Plain bf16 LoRA. On a 96 GB GPU we never need to quantize an 8B model
        # — quantization noise on Llama-3 is small but real, no reason to pay it.
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",  # Fastest stable attention on Blackwell
        )
        if config["training"].get("gradient_checkpointing", True):
            # Match the QLoRA path's "use_reentrant=False" hint — required
            # when combining grad-checkpointing with PEFT.
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            model.enable_input_require_grads()
    else:
        raise ValueError(f"Unknown method '{method}'. Expected 'qlora' or 'lora'.")

    model.config.use_cache = False  # Required for gradient checkpointing

    # Apply LoRA
    lora_config = get_lora_config(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["lora_alpha"],
        lora_dropout=config["lora"]["lora_dropout"],
        target_modules=config["lora"]["target_modules"],
        bias=config["lora"]["bias"],
    )

    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    print("\n" + "=" * 70)
    print_trainable_parameters(model)
    print("=" * 70 + "\n")

    return model
def setup_training_args(config: dict) -> TrainingArguments:
    """Setup training arguments from config."""
    train_config = config["training"]
    output_dir = config["output_dir"]

    return TrainingArguments(
        output_dir=output_dir,
        # Learning & Optimization
        learning_rate=train_config["learning_rate"],
        lr_scheduler_type=train_config["lr_scheduler_type"],
        warmup_ratio=train_config["warmup_ratio"],
        weight_decay=train_config["weight_decay"],
        optim=train_config["optim"],
        max_grad_norm=train_config.get("max_grad_norm", 1.0),
        # Batch Configuration
        per_device_train_batch_size=train_config["per_device_train_batch_size"],
        per_device_eval_batch_size=train_config["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
        # Training Duration
        num_train_epochs=train_config.get("num_train_epochs", 3.0),
        max_steps=train_config.get("max_steps", -1),
        # Evaluation & Saving
        eval_strategy=train_config["eval_strategy"],
        eval_steps=train_config["eval_steps"],
        save_strategy=train_config["save_strategy"],
        save_steps=train_config["save_steps"],
        save_total_limit=train_config["save_total_limit"],
        load_best_model_at_end=train_config["load_best_model_at_end"],
        metric_for_best_model=train_config["metric_for_best_model"],
        greater_is_better=train_config.get("greater_is_better", False),
        # Performance
        bf16=train_config["bf16"],
        tf32=train_config["tf32"],
        fp16=train_config.get("fp16", False),
        gradient_checkpointing=train_config["gradient_checkpointing"],
        dataloader_num_workers=train_config["dataloader_num_workers"],
        # Logging
        logging_dir=train_config.get("logging_dir", f"{output_dir}/logs"),
        logging_steps=train_config["logging_steps"],
        logging_first_step=train_config.get("logging_first_step", True),
        report_to=train_config["report_to"],
        # Other
        seed=train_config.get("seed", 42),
        data_seed=train_config.get("data_seed", 42),
        remove_unused_columns=train_config.get("remove_unused_columns", True),
        group_by_length=train_config.get("group_by_length", False),
        ddp_find_unused_parameters=train_config.get("ddp_find_unused_parameters", False),
    )


def setup_wandb(config: dict):
    """Initialize Weights & Biases logging."""
    if config["training"]["report_to"] == "wandb":
        wandb_config = config.get("wandb", {})
        wandb.init(
            project=wandb_config.get("project", "recruiter-persona-finetuning"),
            name=wandb_config.get("name", "training-run"),
            tags=wandb_config.get("tags", []),
            notes=wandb_config.get("notes", ""),
            config=config,
        )


def main():
    """Main training pipeline."""
    print("\n" + "=" * 70)
    print("🎯 Recruiter Persona Fine-tuning - Training Pipeline")
    print("=" * 70 + "\n")

    # 1. Load configuration
    config = load_config("config.yaml")

    # 2. Setup WandB (optional)
    if config["training"]["report_to"] == "wandb":
        print("Initializing Weights & Biases...")
        setup_wandb(config)

    # 3. Setup tokenizer
    tokenizer = setup_tokenizer(config["base_model"])

    # 4. Setup data loader
    print("\nSetting up data loader...")
    data_loader = RecruiterDataLoader(
        train_path=config["train_data"],
        eval_path=config["eval_data"],
        tokenizer=tokenizer,
        max_seq_length=config["training"]["max_seq_length"],
    )

    # 5. Load and prepare datasets
    train_dataset, eval_dataset = data_loader.prepare_datasets()

    # 6. Setup model
    print("\nSetting up model...")
    model = setup_model(config, tokenizer)

    # 7. Setup training arguments
    print("\nSetting up training arguments...")
    training_args = setup_training_args(config)

    # 8. Setup callbacks
    print("\nSetting up custom callbacks...")
    callbacks = get_default_callbacks(
        tokenizer=tokenizer,
        enable_early_stopping=True,
        enable_memory_monitor=False,
        enable_sample_generation=False,
        early_stopping_patience=2,
        early_stopping_threshold=0.002,
    )
    # Adds `train_eval_gap` to the wandb columns on every eval step.
    callbacks.append(TrainEvalGapCallback())

    # 8.5 Multi-turn assistant-only loss masking.
    #
    # CRITICAL: passing only `response_template` was a bug in v2/v3 — the
    # collator masks tokens up to the FIRST `<|start_header_id|>assistant…`
    # marker only. Everything after (including subsequent USER turns) becomes
    # part of the loss, so the model was being trained to also predict
    # candidate replies. That's why earlier checkpoints fluently roleplayed
    # the candidate at inference time.
    #
    # The fix: pass `instruction_template` as well. The collator then masks
    # every USER block between successive assistant turns, leaving ONLY
    # assistant tokens contributing to loss.
    #
    # We pass exact token IDs to avoid the Llama-3 tokenizer occasionally
    # merging the surrounding newlines with the header tokens, which breaks
    # the subset-matching algorithm inside DataCollatorForCompletionOnlyLM.
    instruction_template = "<|start_header_id|>user<|end_header_id|>"
    response_template    = "<|start_header_id|>assistant<|end_header_id|>"
    instruction_template_ids = tokenizer.encode(instruction_template, add_special_tokens=False)
    response_template_ids    = tokenizer.encode(response_template,    add_special_tokens=False)

    data_collator = DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template_ids,
        response_template=response_template_ids,
        tokenizer=tokenizer,
        mlm=False,
    )

    # 9. Initialize trainer
    print("\nInitializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=config["training"]["max_seq_length"],
        packing=False,  # MUST stay false with DataCollatorForCompletionOnlyLM
        callbacks=callbacks,
        # Quantitative eval metrics. See metrics.py for the formulas.
        # preprocess_logits_for_metrics keeps memory bounded by reducing the
        # (B, T, vocab=128k) logits tensor to (B, T-1, 3) before HF concatenates
        # the per-batch outputs across the eval set.
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        # NEFTune was useful in v2/v3 to fight overfitting that came partly
        # from the broken masking. With masking fixed the signal is cleaner and
        # NEFTune just adds noise; turn it off. (Also deprecated as an SFTTrainer
        # kwarg in newer TRL.)
    )

    # 10. Train
    print("\n" + "=" * 70)
    print("🔥 Starting Training...")
    print("=" * 70 + "\n")

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user!")
        print("Saving checkpoint before exit...")
        trainer.save_model(f"{config['output_dir']}/interrupted")
        print("✅ Checkpoint saved!")
        sys.exit(0)

    # 11. Save final model
    print("\n" + "=" * 70)
    print("💾 Saving Final Model...")
    print("=" * 70)
    final_output = f"{config['output_dir']}/final"
    trainer.save_model(final_output)
    tokenizer.save_pretrained(final_output)
    print(f"✅ Model saved to: {final_output}")

    # 12. Final evaluation
    print("\n" + "=" * 70)
    print("📊 Final Evaluation...")
    print("=" * 70)
    eval_results = trainer.evaluate()
    print(f"\nFinal Evaluation Loss: {eval_results['eval_loss']:.4f}")
    print(f"Perplexity: {torch.exp(torch.tensor(eval_results['eval_loss'])):.2f}")

    # Save evaluation results
    eval_output = f"{config['output_dir']}/eval_results.json"
    import json
    with open(eval_output, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"✅ Evaluation results saved to: {eval_output}")

    # 13. Cleanup
    if config["training"]["report_to"] == "wandb":
        wandb.finish()

    print("\n" + "=" * 70)
    print("✅ Training Complete!")
    print("=" * 70)
    print(f"\n📁 Final model: {final_output}")
    print(f"📊 Eval results: {eval_output}")
    print("\n🎉 Ready for inference testing!")
    print("   Run: python ../scripts/test_inference.py\n")


if __name__ == "__main__":
    main()
