"""Merge LoRA adapters with base model.

Merges the fine-tuned LoRA adapters from training output with the base model
to create a single, standalone model that can be deployed or uploaded to HuggingFace Hub.

Usage:
    # Merge output_v4 final checkpoint (default)
    python merge_adapters.py

    # Merge a specific checkpoint
    python merge_adapters.py --adapter-path ../output_v4/recruiter-persona-llama-3.1-8b/checkpoint-400

    # Specify custom output directory
    python merge_adapters.py --output-dir ./my-merged-model

    # Upload to HuggingFace Hub after merging
    python merge_adapters.py --push-to-hub --hub-model-id your-username/model-name
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapters with base model"
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default="../output_v4/recruiter-persona-llama-3.1-8b/final",
        help="Path to the adapter directory (default: output_v4/final)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model ID (auto-detected from adapter_config.json if not specified)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../output_v4/recruiter-persona-llama-3.1-8b/merged-full",
        help="Output directory for merged model",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Upload merged model to HuggingFace Hub",
    )
    parser.add_argument(
        "--hub-model-id",
        type=str,
        default=None,
        help="HuggingFace Hub model ID (e.g., username/model-name)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for merging (default: auto)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type for model weights (default: bfloat16)",
    )
    return parser.parse_args()


def get_base_model_from_adapter_config(adapter_path: str) -> str:
    """Read base_model_name_or_path from adapter_config.json."""
    import json

    config_path = Path(adapter_path) / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"adapter_config.json not found at {config_path}. "
            f"Please specify --base-model manually."
        )

    with open(config_path) as f:
        config = json.load(f)

    base_model = config.get("base_model_name_or_path")
    if not base_model:
        raise ValueError(
            "base_model_name_or_path not found in adapter_config.json. "
            "Please specify --base-model manually."
        )

    return base_model


def main():
    args = parse_args()

    # Resolve paths
    adapter_path = Path(args.adapter_path).resolve()
    output_dir = Path(args.output_dir).resolve()

    print("=" * 70)
    print("LoRA Adapter → Base Model Merge")
    print("=" * 70)

    # Validate adapter path
    if not adapter_path.exists():
        print(f"❌ Error: Adapter path does not exist: {adapter_path}")
        sys.exit(1)

    if not (adapter_path / "adapter_config.json").exists():
        print(f"❌ Error: adapter_config.json not found in {adapter_path}")
        sys.exit(1)

    # Determine base model
    base_model_id = args.base_model
    if base_model_id is None:
        print(f"\n📖 Reading base model from adapter_config.json...")
        base_model_id = get_base_model_from_adapter_config(adapter_path)

    print(f"\n📋 Configuration:")
    print(f"  Base model:    {base_model_id}")
    print(f"  Adapter path:  {adapter_path}")
    print(f"  Output dir:    {output_dir}")
    print(f"  Device:        {args.device}")
    print(f"  dtype:         {args.dtype}")

    # Determine dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    # Check CUDA availability
    if args.device == "auto":
        device_map = "auto"
        if torch.cuda.is_available():
            print(f"  GPU:           {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version:  {torch.version.cuda}")
        else:
            print("  GPU:           Not available, using CPU")
    else:
        device_map = args.device

    # Step 1: Load base model
    print(f"\n🔄 Step 1/4: Loading base model ({base_model_id})...")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        print(f"✅ Base model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading base model: {e}")
        sys.exit(1)

    # Step 2: Load LoRA adapters
    print(f"\n🔄 Step 2/4: Loading LoRA adapters from {adapter_path}...")
    try:
        model = PeftModel.from_pretrained(
            base_model,
            str(adapter_path),
            device_map=device_map,
        )
        print(f"✅ LoRA adapters loaded successfully")
    except Exception as e:
        print(f"❌ Error loading adapters: {e}")
        sys.exit(1)

    # Step 3: Merge adapters with base model
    print(f"\n🔄 Step 3/4: Merging adapters with base model...")
    try:
        merged_model = model.merge_and_unload()
        print(f"✅ Adapters merged successfully")

        # Print model info
        total_params = sum(p.numel() for p in merged_model.parameters())
        print(f"  Total parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    except Exception as e:
        print(f"❌ Error merging adapters: {e}")
        sys.exit(1)

    # Step 4: Save merged model
    print(f"\n🔄 Step 4/4: Saving merged model to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Save model
        merged_model.save_pretrained(
            str(output_dir),
            safe_serialization=True,
        )

        # Save tokenizer
        print(f"  Saving tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        tokenizer.save_pretrained(str(output_dir))

        print(f"✅ Merged model saved successfully")

        # List saved files
        saved_files = list(output_dir.glob("*"))
        print(f"\n📁 Saved files ({len(saved_files)} files):")
        for file in sorted(saved_files)[:10]:  # Show first 10 files
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"  - {file.name} ({size_mb:.2f} MB)")
        if len(saved_files) > 10:
            print(f"  ... and {len(saved_files) - 10} more files")

        total_size_gb = sum(f.stat().st_size for f in saved_files) / (1024 ** 3)
        print(f"\n  Total size: {total_size_gb:.2f} GB")

    except Exception as e:
        print(f"❌ Error saving merged model: {e}")
        sys.exit(1)

    # Step 5: Upload to HuggingFace Hub (optional)
    if args.push_to_hub:
        if not args.hub_model_id:
            print(f"\n❌ Error: --hub-model-id required when using --push-to-hub")
            sys.exit(1)

        print(f"\n🔄 Step 5/5: Uploading to HuggingFace Hub ({args.hub_model_id})...")
        try:
            merged_model.push_to_hub(args.hub_model_id, private=False)
            tokenizer.push_to_hub(args.hub_model_id, private=False)
            print(f"✅ Model uploaded to https://huggingface.co/{args.hub_model_id}")
        except Exception as e:
            print(f"❌ Error uploading to Hub: {e}")
            print(f"   Make sure you're logged in: huggingface-cli login")
            sys.exit(1)

    print("\n" + "=" * 70)
    print("✅ Merge completed successfully!")
    print("=" * 70)
    print(f"\nMerged model saved at: {output_dir}")
    print(f"\nYou can now use the merged model:")
    print(f"  from transformers import AutoModelForCausalLM, AutoTokenizer")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{output_dir}')")
    print(f"  tokenizer = AutoTokenizer.from_pretrained('{output_dir}')")
    print()
if __name__ == "__main__":
    main()
