"""Push merged model to HuggingFace Hub.

This script uploads the locally merged LoRA model to HuggingFace Hub
so it can be used for inference and deployment.

Usage:
    python push_to_hub.py
    python push_to_hub.py --model-dir ../output_v2/recruiter-persona-llama-3.1-8b/merged-full
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi, login
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Push merged model to HuggingFace Hub"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="../output_v3/recruiter-persona-llama-3.1-8b/merged-full",
        help="Path to the merged model directory",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="oussema2021/fintuned_v3_AiRecruter",
        help="HuggingFace Hub repository ID (username/model-name)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private (default: public)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token (optional, will use cached token if not provided)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model_dir = Path(args.model_dir).resolve()

    print("=" * 70)
    print("Push Merged Model to HuggingFace Hub")
    print("=" * 70)

    # Validate model directory
    if not model_dir.exists():
        print(f"❌ Error: Model directory does not exist: {model_dir}")
        sys.exit(1)

    required_files = ["config.json", "model.safetensors.index.json"]
    for file in required_files:
        if not (model_dir / file).exists():
            print(f"❌ Error: Required file not found: {file}")
            print(f"   This doesn't appear to be a valid merged model directory.")
            sys.exit(1)

    print(f"\n📋 Configuration:")
    print(f"  Model directory: {model_dir}")
    print(f"  Repository ID:   {args.repo_id}")
    print(f"  Visibility:      {'Private' if args.private else 'Public'}")

    # Calculate model size
    total_size_gb = sum(f.stat().st_size for f in model_dir.glob("*")) / (1024 ** 3)
    print(f"  Model size:      {total_size_gb:.2f} GB")

    # Login to HuggingFace Hub
    print(f"\n🔐 Logging in to HuggingFace Hub...")
    try:
        if args.token:
            login(token=args.token)
        else:
            # Will use cached token from `huggingface-cli login`
            login()
        print(f"✅ Successfully logged in")
    except Exception as e:
        print(f"❌ Login failed: {e}")
        print(f"\nPlease run: huggingface-cli login")
        print(f"Or provide a token with: python push_to_hub.py --token YOUR_TOKEN")
        sys.exit(1)

    # Load model and tokenizer
    print(f"\n🔄 Loading model from {model_dir}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            device_map="cpu",  # Keep on CPU for uploading
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        print(f"✅ Model and tokenizer loaded successfully")

        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)

    # Push to Hub
    print(f"\n🚀 Pushing model to {args.repo_id}...")
    print(f"   This may take several minutes depending on your internet speed...")

    try:
        # Push model
        print(f"\n  📤 Uploading model weights...")
        model.push_to_hub(
            args.repo_id,
            private=args.private,
            safe_serialization=True,
        )

        # Push tokenizer
        print(f"  📤 Uploading tokenizer...")
        tokenizer.push_to_hub(
            args.repo_id,
            private=args.private,
        )

        print(f"\n✅ Successfully pushed to HuggingFace Hub!")

    except Exception as e:
        print(f"\n❌ Error pushing to Hub: {e}")
        print(f"\nPossible issues:")
        print(f"  1. Make sure you're logged in: huggingface-cli login")
        print(f"  2. Check your token has write permissions")
        print(f"  3. Verify the repository name is valid")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("✅ Upload completed successfully!")
    print("=" * 70)
    print(f"\nModel available at: https://huggingface.co/{args.repo_id}")
    print(f"\nYou can now use it with:")
    print(f"  from transformers import AutoModelForCausalLM, AutoTokenizer")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{args.repo_id}')")
    print(f"  tokenizer = AutoTokenizer.from_pretrained('{args.repo_id}')")
    print()


if __name__ == "__main__":
    main()
