#!/bin/bash
# Quick-start training script
# Runs setup checks and starts training

set -e

echo "=" | tr '=' '=' | head -c 70
echo ""
echo "🚀 Recruiter Persona Fine-tuning - Quick Start"
echo "=" | tr '=' '=' | head -c 70
echo ""

# Check if we're in the right directory
if [ ! -f "../train.py" ]; then
    echo "❌ Error: Must run from training/scripts/ directory"
    echo "   Current directory: $(pwd)"
    exit 1
fi

# Quick checks
echo "📋 Pre-flight Checks:"
echo ""

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo "  ✅ GPU detected"
    nvidia-smi --query-gpu=name,memory.free --format=csv,noheader | head -1
else
    echo "  ⚠️  No GPU detected - training will be very slow!"
fi

# Check Python packages
echo ""
echo "  Checking Python packages..."
if python -c "import torch, transformers, peft, trl, bitsandbytes, yaml" 2>/dev/null; then
    echo "  ✅ Required packages installed"
else
    echo "  ❌ Missing packages. Running setup..."
    ./setup.sh
fi

# Check data files
echo ""
TRAIN_FILE="../../data/final/recruiter_llama_train_messages.jsonl"
EVAL_FILE="../../data/final/recruiter_llama_eval_messages.jsonl"

if [ -f "$TRAIN_FILE" ] && [ -f "$EVAL_FILE" ]; then
    echo "  ✅ Data files found"
else
    echo "  ❌ Data files not found"
    echo "     Expected:"
    echo "       - $TRAIN_FILE"
    echo "       - $EVAL_FILE"
    exit 1
fi

# Check config
echo ""
if [ -f "../config.yaml" ]; then
    echo "  ✅ Configuration found"
else
    echo "  ❌ config.yaml not found"
    exit 1
fi

# Show configuration
echo ""
echo "-" | tr '-' '-' | head -c 70
echo ""
echo "⚙️  Training Configuration:"
echo "-" | tr '-' '-' | head -c 70
echo ""

python -c "
import yaml
with open('../config.yaml') as f:
    config = yaml.safe_load(f)
    print(f\"  Model:         {config['base_model']}\")
    print(f\"  Method:        {config['method'].upper()}\")
    print(f\"  LoRA Rank:     {config['lora']['r']}\")
    print(f\"  Learning Rate: {config['training']['learning_rate']}\")
    print(f\"  Epochs:        {config['training']['num_train_epochs']}\")
    print(f\"  Batch Size:    {config['training']['per_device_train_batch_size']} x {config['training']['gradient_accumulation_steps']} = {config['training']['per_device_train_batch_size'] * config['training']['gradient_accumulation_steps']}\")
    print(f\"  Max Seq Len:   {config['training']['max_seq_length']}\")
    print()
    print(f\"  Expected Time: ~{config['hardware']['expected_training_time_hours']} hours\")
    print(f\"  Expected VRAM: {config['hardware']['expected_usage_gb']} GB\")
"

echo ""
echo "-" | tr '-' '-' | head -c 70
echo ""

# Confirm before starting
read -p "Start training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

# Navigate to training directory
cd ..

# Set environment variables (optional)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false

# Start training
echo ""
echo "=" | tr '=' '=' | head -c 70
echo ""
echo "🔥 Starting Training..."
echo "=" | tr '=' '=' | head -c 70
echo ""
echo "💡 Tip: Training will take ~4-5 hours. You can:"
echo "   - Monitor progress in the console"
echo "   - View logs at: output/logs/"
echo "   - Track metrics on W&B (if configured)"
echo ""
echo "   To stop training: Ctrl+C (checkpoint will be saved)"
echo ""
echo "=" | tr '=' '=' | head -c 70
echo ""

# Run training
python train.py

# Training complete
echo ""
echo "=" | tr '=' '=' | head -c 70
echo ""
echo "✅ Training Complete!"
echo "=" | tr '=' '=' | head -c 70
echo ""
echo "📁 Model saved to: output/recruiter-persona-llama-3.1-8b/final"
echo ""
echo "🧪 Next Steps:"
echo "   1. Test the model: python scripts/test_inference.py"
echo "   2. Run quality check: python scripts/test_inference.py --mode quality"
echo "   3. Deploy the model to your application"
echo ""
