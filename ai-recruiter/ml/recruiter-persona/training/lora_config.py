"""
LoRA and QLoRA Configuration for Recruiter Persona Fine-tuning
Provides configuration for efficient parameter-efficient fine-tuning
"""

import torch
from peft import LoraConfig, TaskType
from transformers import BitsAndBytesConfig


def get_qlora_config():
    """
    Get QLoRA (4-bit quantized LoRA) configuration.
    Optimized for L40 24GB GPU - balances quality and memory efficiency.

    Returns:
        BitsAndBytesConfig: 4-bit quantization configuration
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",              # NormalFloat4 quantization
        bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in bfloat16
        bnb_4bit_use_double_quant=True,         # Double quantization for memory
    )


def get_lora_config(
    r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    target_modules: list = None,
    bias: str = "none",
):
    """
    Get LoRA adapter configuration.

    Args:
        r: LoRA rank (16-64). Higher = more expressive but slower.
               Recommended: 32 for good balance
        lora_alpha: LoRA alpha scaling. Typically 2x rank.
               Recommended: 64 (with r=32)
        lora_dropout: Dropout for LoRA layers. Low for conversational tasks.
               Recommended: 0.05
        target_modules: Which modules to apply LoRA to.
               Default: All linear layers for LLaMA architecture
        bias: Bias handling. Options: "none", "all", "lora_only"
               Recommended: "none"

    Returns:
        LoraConfig: LoRA adapter configuration
    """
    if target_modules is None:
        # Default: target all linear layers in LLaMA/Mistral/Qwen architectures
        target_modules = [
            "q_proj",      # Query projection
            "k_proj",      # Key projection
            "v_proj",      # Value projection
            "o_proj",      # Output projection
            "gate_proj",   # Gate projection (MLP)
            "up_proj",     # Up projection (MLP)
            "down_proj",   # Down projection (MLP)
        ]

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
        task_type=TaskType.CAUSAL_LM,
    )


def get_preset_config(preset: str = "balanced"):
    """
    Get preset LoRA configurations for different use cases.

    Args:
        preset: One of "fast", "balanced", "quality"

    Returns:
        tuple: (lora_config, description)
    """
    presets = {
        "fast": {
            "config": get_lora_config(r=16, lora_alpha=32, lora_dropout=0.05),
            "description": "Fast training with lower rank. Good for experimentation.",
            "expected_time": "~3-4 hours",
            "memory_usage": "~16-18GB",
        },
        "balanced": {
            "config": get_lora_config(r=32, lora_alpha=64, lora_dropout=0.05),
            "description": "Balanced quality and speed. Recommended for most cases.",
            "expected_time": "~4-5 hours",
            "memory_usage": "~18-20GB",
        },
        "quality": {
            "config": get_lora_config(r=64, lora_alpha=128, lora_dropout=0.05),
            "description": "Higher rank for best quality. Slower training.",
            "expected_time": "~6-7 hours",
            "memory_usage": "~20-22GB",
        },
    }

    if preset not in presets:
        raise ValueError(f"Unknown preset '{preset}'. Choose from: {list(presets.keys())}")

    return presets[preset]["config"], presets[preset]


def print_trainable_parameters(model):
    """
    Print the number of trainable parameters in the model.
    Useful for understanding how much is being fine-tuned.

    Args:
        model: The PEFT model with LoRA adapters
    """
    trainable_params = 0
    all_param = 0

    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    trainable_pct = 100 * trainable_params / all_param

    print(f"Trainable params: {trainable_params:,} || "
          f"All params: {all_param:,} || "
          f"Trainable%: {trainable_pct:.4f}%")

    return trainable_params, all_param


# Example usage and configuration validation
if __name__ == "__main__":
    print("=" * 70)
    print("LoRA Configuration Presets for Recruiter Persona Fine-tuning")
    print("=" * 70)

    for preset_name in ["fast", "balanced", "quality"]:
        config, info = get_preset_config(preset_name)
        print(f"\n{preset_name.upper()} Preset:")
        print(f"  Description: {info['description']}")
        print(f"  Expected Time: {info['expected_time']}")
        print(f"  Memory Usage: {info['memory_usage']}")
        print(f"  Configuration:")
        print(f"    - Rank (r): {config.r}")
        print(f"    - Alpha: {config.lora_alpha}")
        print(f"    - Dropout: {config.lora_dropout}")
        print(f"    - Target Modules: {', '.join(config.target_modules)}")

    print("\n" + "=" * 70)
    print("QLoRA (4-bit) Configuration:")
    print("=" * 70)
    qlora_config = get_qlora_config()
    print(f"  Load in 4-bit: {qlora_config.load_in_4bit}")
    print(f"  Quantization Type: {qlora_config.bnb_4bit_quant_type}")
    print(f"  Compute Dtype: {qlora_config.bnb_4bit_compute_dtype}")
    print(f"  Double Quantization: {qlora_config.bnb_4bit_use_double_quant}")
    print("\nThis saves ~75% memory compared to full precision fine-tuning!")
