# Recruiter Persona Fine-tuning Guide

Complete setup for fine-tuning an LLM to act as a professional recruiter using QLoRA on Lightning AI's L40 24GB GPU.

## 📋 Quick Start

```bash
cd scripts/

# 1. Install dependencies
pip install -r requirements_finetuning.txt

# 2. Login to HuggingFace (required for Llama models)
huggingface-cli login

# 3. (Optional) Login to Weights & Biases
wandb login

# 4. Run training
./run_training.sh

# Or run directly:
python train_recruiter_persona.py
```

## 🎯 Recommended Configuration

### Best Model: **Llama 3.1 8B Instruct**
- **HuggingFace**: `meta-llama/Meta-Llama-3.1-8B-Instruct`
- **Why**: Best instruction-following, excellent conversational abilities
- **Memory**: ~18-20GB VRAM with QLoRA
- **Training Time**: ~4-5 hours for 3 epochs on L40

### Optimal Hyperparameters

**LoRA Configuration:**
```python
r = 32                    # Rank (sweet spot for quality/speed)
lora_alpha = 64          # 2x rank
lora_dropout = 0.05      # Light regularization
target_modules = [       # All linear layers
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]
```

**Training Configuration:**
```python
learning_rate = 2e-4                    # Optimal for conversational tasks
batch_size = 4 (per device)             # Adjust based on memory
gradient_accumulation_steps = 8         # Effective batch size = 32
num_train_epochs = 3                    # Good convergence
max_seq_length = 2048                   # Handles long interviews
optimizer = "paged_adamw_8bit"          # Memory-efficient
lr_scheduler = "cosine"                 # Smooth decay
warmup_ratio = 0.03                     # 3% warmup
```

## 📊 Dataset

- **Training samples**: 7,749 multi-turn interview conversations
- **Eval samples**: 880 conversations
- **Format**: Llama messages format (JSONL)
- **Task**: Recruiter persona conducting professional interviews

## 🚀 Files Overview

### Core Training
- **`train_recruiter_persona.py`** - Main training script with QLoRA
- **`requirements_finetuning.txt`** - All required dependencies
- **`run_training.sh`** - Automated setup and training launcher

### Testing & Inference
- **`test_inference.py`** - Interactive and batch testing
  - Interactive interview mode (chat with model)
  - Batch test mode (predefined test cases)

### Documentation
- **`FINETUNING_CONFIG.md`** - Comprehensive configuration guide
  - All model options (Llama 3.1, Mistral, Qwen)
  - Hyperparameter tuning guide
  - Troubleshooting tips
  - Expected results and metrics

## 💡 Model Alternatives

### 1. Llama 3.1 8B ⭐ (Recommended)
- Best overall quality
- 4-5 hours training
- ~$2 on Lightning AI

### 2. Mistral 7B v0.3
- Fast training (3-4 hours)
- Good performance
- More efficient

### 3. Qwen 2.5 7B
- Excellent at role-playing
- Strong system prompt following
- 3.5-4.5 hours

### 4. Llama 3.2 3B (Fast iteration)
- Quick experiments (1.5-2 hours)
- Lower quality but very fast
- Good for hyperparameter tuning

## 📈 Expected Results

### Training Metrics
- **Initial Loss**: ~2.0-2.5
- **Final Train Loss**: ~0.4-0.8
- **Final Eval Loss**: ~0.5-0.9
- **Gradient Norm**: Should stabilize < 1.0

### Quality Indicators
✅ Maintains recruiter persona consistently
✅ Asks one question at a time
✅ Adapts follow-ups to candidate responses
✅ Professional, respectful tone
✅ Avoids discriminatory questions
✅ Natural conversation flow

## 🔧 Testing Your Model

### Interactive Mode
```bash
python test_inference.py
# Select option 1: Interactive interview
```

### Batch Testing
```bash
python test_inference.py
# Select option 2: Batch test
```

### Custom Testing
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load model
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
model = PeftModel.from_pretrained(base_model, "./recruiter-persona-llama-3.1-8b/final")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

# Generate
messages = [
    {"role": "system", "content": "You are Alex, a senior recruiter..."},
    {"role": "assistant", "content": "Good morning! Can you tell me about yourself?"},
    {"role": "user", "content": "I have 5 years of experience..."}
]
```

## 🛠️ Troubleshooting

### Out of Memory (OOM)
```python
# Reduce batch size
per_device_train_batch_size = 2
gradient_accumulation_steps = 16

# Or reduce sequence length
max_seq_length = 1024

# Or reduce LoRA rank
r = 16
```

### Slow Training
```python
# Enable Flash Attention 2
attn_implementation = "flash_attention_2"

# Reduce logging
logging_steps = 50

# Increase workers
dataloader_num_workers = 4
```

### Poor Quality
```python
# Increase LoRA rank
r = 64
lora_alpha = 128

# Train longer
num_train_epochs = 4-5

# Lower learning rate
learning_rate = 1e-4
```

## 💰 Cost Estimation

**On Lightning AI L40 24GB** (~$0.50/hr):
- Training (3 epochs): ~$2.00-2.50
- Testing/inference: ~$0.10-0.20
- **Total**: ~$2.50

## 📦 Output

After training completes:
```
./recruiter-persona-llama-3.1-8b/
├── final/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors  (~600MB - LoRA weights)
│   └── tokenizer files
├── checkpoint-100/
├── checkpoint-200/
└── logs/
```

## 🔄 Next Steps

1. **Test the model**: `python test_inference.py`
2. **Merge weights** (optional for faster inference):
   ```python
   merged_model = model.merge_and_unload()
   merged_model.save_pretrained("./recruiter-merged")
   ```
3. **Quantize for deployment** (optional):
   - GGUF for CPU (llama.cpp)
   - AWQ/GPTQ for GPU
4. **Deploy**: Integrate into your AI recruiter system

## 📚 Additional Resources

- [PEFT Documentation](https://huggingface.co/docs/peft)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Llama 3.1 Model Card](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)

## ❓ Common Questions

**Q: Can I use a smaller GPU?**
A: With aggressive settings (batch_size=1, r=8, max_len=1024), you might fit on 16GB, but 24GB is recommended.

**Q: How do I know if the model is overfitting?**
A: Monitor eval_loss - if it starts increasing while train_loss decreases, you're overfitting. Use early stopping.

**Q: Should I merge the LoRA weights?**
A: For deployment, merging is faster. For experimentation, keep them separate (easier to swap).

**Q: Can I fine-tune on CPU?**
A: Technically yes, but it would take days/weeks. Not recommended.

## 📝 License

Training code: MIT License
Models: Check respective model licenses (Llama 3.1, Mistral, etc.)
