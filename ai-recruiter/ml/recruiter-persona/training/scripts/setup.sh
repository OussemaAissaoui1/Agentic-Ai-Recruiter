#!/bin/bash
# Setup script for Lightning AI L40 24GB
# Prepares the environment for recruiter persona fine-tuning

set -e

echo "=================================="
echo "🚀 Recruiter Persona Training Setup"
echo "=================================="

# Check if we're on Lightning AI or have CUDA
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "📊 GPU Information:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
else
    echo "⚠️  Warning: No NVIDIA GPU detected!"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip install -q -r ../requirements.txt

# Verify critical packages
echo ""
echo "✅ Verifying installations..."
python -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'  Transformers: {transformers.__version__}')"
python -c "import peft; print(f'  PEFT: {peft.__version__}')"
python -c "import trl; print(f'  TRL: {trl.__version__}')"
python -c "import bitsandbytes; print(f'  BitsAndBytes: {bitsandbytes.__version__}')"
python -c "import yaml; print(f'  PyYAML: {yaml.__version__}')"

# Check CUDA
echo ""
python -c "import torch; print(f'  CUDA Available: {torch.cuda.is_available()}')"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    python -c "import torch; print(f'  CUDA Version: {torch.version.cuda}')"
    python -c "import torch; print(f'  GPU Count: {torch.cuda.device_count()}')"
fi

# Check data files
echo ""
echo "📁 Checking data files..."
TRAIN_FILE="../../data/final/recruiter_llama_train_messages.jsonl"
EVAL_FILE="../../data/final/recruiter_llama_eval_messages.jsonl"

if [ -f "$TRAIN_FILE" ]; then
    TRAIN_COUNT=$(wc -l < "$TRAIN_FILE")
    echo "  ✅ Training data: $TRAIN_COUNT samples"
else
    echo "  ❌ Training data not found: $TRAIN_FILE"
    exit 1
fi

if [ -f "$EVAL_FILE" ]; then
    EVAL_COUNT=$(wc -l < "$EVAL_FILE")
    echo "  ✅ Eval data: $EVAL_COUNT samples"
else
    echo "  ❌ Eval data not found: $EVAL_FILE"
    exit 1
fi

# HuggingFace login check
echo ""
echo "🔐 HuggingFace Authentication:"
echo "  For Llama models, you need to:"
echo "  1. Accept the license: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct"
echo "  2. Login: huggingface-cli login"
echo ""
read -p "Have you completed HuggingFace authentication? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Please complete authentication first:"
    echo "  huggingface-cli login"
    exit 1
fi

# Wandb setup (optional)
echo ""
echo "📈 Weights & Biases (Optional):"
read -p "Do you want to use W&B for experiment tracking? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if command -v wandb &> /dev/null; then
        wandb login
    else
        echo "  wandb not found, installing..."
        pip install -q wandb
        wandb login
    fi
else
    echo "  Skipping W&B setup. Training will log locally only."
    export WANDB_MODE=offline
fi

# Configuration check
echo ""
echo "⚙️  Checking configuration..."
if [ -f "../config.yaml" ]; then
    echo "  ✅ config.yaml found"
    echo ""
    echo "  Current configuration:"
    python -c "
import yaml
with open('../config.yaml') as f:
    config = yaml.safe_load(f)
    print(f\"    Model: {config['base_model']}\")
    print(f\"    Method: {config['method']}\")
    print(f\"    LoRA Rank: {config['lora']['r']}\")
    print(f\"    Learning Rate: {config['training']['learning_rate']}\")
    print(f\"    Epochs: {config['training']['num_train_epochs']}\")
    print(f\"    Batch Size: {config['training']['per_device_train_batch_size']}\")
"
else
    echo "  ❌ config.yaml not found"
    exit 1
fi

# Summary
echo ""
echo "=================================="
echo "✅ Setup Complete!"
echo "=================================="
echo ""
echo "📋 Next Steps:"
echo "  1. Review config.yaml for any final adjustments"
echo "  2. Run training: cd .. && python train.py"
echo "  3. Or use quick start: ./run_training.sh"
echo ""
echo "💡 Estimated training time: ~4-5 hours on L40 24GB"
echo "💰 Estimated cost on Lightning AI: ~\$2-2.50"
echo ""
