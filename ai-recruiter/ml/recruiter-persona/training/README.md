# Training Pipeline

Fine-tuning scripts and configurations for the recruiter persona model using QLoRA on Lightning AI L40 24GB.

## 📁 File Structure

| File | Purpose |
|------|---------|
| `train.py` | Main training entrypoint - orchestrates the complete pipeline |
| `config.yaml` | Training hyperparameters and model configuration (YAML) |
| `lora_config.py` | LoRA/QLoRA adapter configuration and presets |
| `trainer.py` | Custom trainer with interview-specific callbacks |
| `data_loader.py` | Training data loading, formatting, and validation |
| `requirements.txt` | Python dependencies for training |
| `scripts/` | Helper scripts for setup, testing, and inference |

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Login to HuggingFace (for Llama models)
```bash
huggingface-cli login
```

### 3. (Optional) Setup Weights & Biases
```bash
wandb login
```

### 4. Configure Training
Edit `config.yaml` to adjust hyperparameters (or use defaults).

### 5. Run Training
```bash
python train.py
```

## ⚙️ Training Approaches

### 1. **QLoRA** (Recommended) ⭐
- **Memory**: ~18-20GB VRAM
- **Training Time**: ~4-5 hours on L40
- **Quality**: Excellent for conversational tasks
- **Method**: 4-bit quantization + LoRA adapters
- **Use Case**: Default for L40 24GB GPU

### 2. **LoRA**
- **Memory**: ~28-32GB VRAM (requires A100 40GB+)
- **Training Time**: ~6-8 hours
- **Quality**: Slightly better than QLoRA
- **Method**: 16-bit + LoRA adapters
- **Use Case**: If you have larger GPU

### 3. **Full Fine-Tuning**
- **Memory**: ~40+ GB VRAM (requires A100 40GB+)
- **Training Time**: ~10-12 hours
- **Quality**: Best, but marginal improvement over LoRA
- **Use Case**: Small models (<3B) or unlimited compute

## 📊 Configuration

The `config.yaml` file contains all training hyperparameters:

```yaml
base_model: "meta-llama/Meta-Llama-3.1-8B-Instruct"
method: "qlora"
output_dir: "./output/recruiter-persona-llama-3.1-8b"

train_data: "../data/final/recruiter_llama_train_messages.jsonl"
eval_data: "../data/final/recruiter_llama_eval_messages.jsonl"

lora:
  r: 32                    # LoRA rank
  lora_alpha: 64          # LoRA alpha (2x rank)
  lora_dropout: 0.05      # Dropout

training:
  learning_rate: 2.0e-4   # Optimal for conversations
  num_train_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 8  # Effective batch = 32
  max_seq_length: 2048
```

### Alternative Models
Edit `base_model` in config.yaml:
- `meta-llama/Meta-Llama-3.1-8B-Instruct` - Best overall (recommended)
- `mistralai/Mistral-7B-Instruct-v0.3` - Faster training
- `Qwen/Qwen2.5-7B-Instruct` - Better role-playing
- `meta-llama/Llama-3.2-3B-Instruct` - Quick experiments

## 🛠️ Custom Components

### LoRA Configuration (`lora_config.py`)
- Preset configurations: `fast`, `balanced`, `quality`
- QLoRA 4-bit quantization setup
- Trainable parameter analysis

```python
from lora_config import get_preset_config
config, info = get_preset_config("balanced")
```

### Data Loader (`data_loader.py`)
- Validates JSONL format
- Applies chat templates
- Analyzes dataset statistics
- Formats conversations for training

```python
from data_loader import RecruiterDataLoader
loader = RecruiterDataLoader(train_path, eval_path, tokenizer)
train_dataset, eval_dataset = loader.prepare_datasets()
```

### Custom Callbacks (`trainer.py`)
- **InterviewMetricsCallback**: Track progress, ETA, loss
- **EarlyStoppingCallback**: Prevent overfitting
- **MemoryMonitorCallback**: Monitor GPU memory usage
- **SampleGenerationCallback**: Generate samples during training

## 📈 Expected Results

### Training Metrics
- **Initial Loss**: ~2.0-2.5
- **Final Training Loss**: ~0.4-0.8
- **Final Eval Loss**: ~0.5-0.9
- **Total Steps**: ~730 (for 3 epochs with 7,749 train samples)

### Resource Usage (L40 24GB)
- **VRAM**: 18-20GB (leaves 4-6GB headroom)
- **Training Time**: 4-5 hours for 3 epochs
- **Cost on Lightning AI**: ~$2-2.50 (at $0.50/hr)

### Quality Indicators
✅ Maintains recruiter persona consistently
✅ Asks one question at a time
✅ Adapts follow-ups to candidate responses
✅ Professional, respectful tone
✅ Avoids discriminatory questions
✅ Natural conversation flow

## 🧪 Testing

After training completes:

```bash
cd scripts/
python test_inference.py
```

Options:
1. **Interactive Mode**: Chat with the fine-tuned model
2. **Batch Test**: Run predefined test cases

## 🔧 Troubleshooting

### Out of Memory (OOM)
Edit `config.yaml`:
```yaml
training:
  per_device_train_batch_size: 2  # Reduce from 4
  gradient_accumulation_steps: 16  # Increase to maintain effective batch
  max_seq_length: 1024  # Reduce if still OOM
```

Or reduce LoRA rank:
```yaml
lora:
  r: 16  # Reduce from 32
  lora_alpha: 32
```

### Slow Training
Enable optimizations in `config.yaml`:
```yaml
training:
  tf32: true
  dataloader_num_workers: 4
```

Or try a faster model:
```yaml
base_model: "mistralai/Mistral-7B-Instruct-v0.3"
```

### Poor Quality
Increase model capacity:
```yaml
lora:
  r: 64  # Increase from 32
  lora_alpha: 128
```

Or train longer:
```yaml
training:
  num_train_epochs: 4-5
```

### Overfitting
Reduce epochs:
```yaml
training:
  num_train_epochs: 2
```

Enable early stopping (already enabled by default in callbacks).

## 📁 Output Structure

After training:
```
output/recruiter-persona-llama-3.1-8b/
├── final/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors  (~600MB)
│   ├── tokenizer files
│   └── special_tokens_map.json
├── checkpoint-100/
├── checkpoint-200/
├── checkpoint-300/
├── logs/
│   └── events.out.tfevents.*
└── eval_results.json
```

## 🔄 Next Steps

1. **Test the model**: `python scripts/test_inference.py`
2. **Merge weights** (optional for deployment):
   ```pythonfrom peft import PeftModel
   model = PeftModel.from_pretrained(base_model, "./output/.../final")
   merged = model.merge_and_unload()
   merged.save_pretrained("./merged-model")
   ```
3. **Deploy**: Integrate into your AI recruiter system

## 📚 Additional Resources

- [Training Configuration Guide](config.yaml)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Llama 3.1 Model Card](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
