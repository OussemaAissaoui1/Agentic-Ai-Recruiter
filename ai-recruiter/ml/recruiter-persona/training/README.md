# Training Pipeline

Fine-tuning scripts and configurations for the recruiter persona model.

## Key Files

| File | Purpose |
|------|---------|
| `train.py` | Main training entrypoint |
| `config.yaml` | Training hyperparameters and model config |
| `lora_config.py` | LoRA/QLoRA adapter configuration |
| `trainer.py` | Custom trainer with interview-specific callbacks |
| `data_loader.py` | Training data loading and collation |

## Training Approaches

1. **Full Fine-Tuning** - For small models with sufficient compute
2. **LoRA** - Low-Rank Adaptation for efficient fine-tuning
3. **QLoRA** - 4-bit quantized LoRA for memory efficiency

## Configuration

```yaml
base_model: meta-llama/Llama-3-8B-Instruct
method: qlora
lora_r: 64
lora_alpha: 128
train_data: data_v2/final/recruiter_llama_train_messages.jsonl
eval_data: data_v2/final/recruiter_llama_eval_messages.jsonl
```
