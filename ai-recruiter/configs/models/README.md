# Model Configurations

Model serving and inference configurations.

## Key Files

| File | Purpose |
|------|---------|
| `recruiter_persona.yaml` | Fine-tuned recruiter model config |
| `embeddings.yaml` | Embedding model for semantic search |
| `asr.yaml` | Speech-to-text model config |
| `tts.yaml` | Text-to-speech model config |
| `vision.yaml` | Computer vision model configs |

## Example

```yaml
# recruiter_persona.yaml
model_id: recruiter-persona-v2
base_model: meta-llama/Llama-3-8B-Instruct
adapter_path: models/recruiter-persona/adapters/v2
quantization: 4bit
max_model_len: 4096
gpu_memory_utilization: 0.9
```
