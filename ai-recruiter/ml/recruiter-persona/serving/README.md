# Model Serving

Inference server configuration for the recruiter persona model.

## Key Files

| File | Purpose |
|------|---------|
| `server.py` | Inference server (vLLM / TGI wrapper) |
| `config.yaml` | Serving configuration |
| `client.py` | Python client for model inference |
| `batch_inference.py` | Batch processing for offline evaluation |

## Serving Options

1. **vLLM** - High-throughput serving with PagedAttention
2. **TGI** - HuggingFace Text Generation Inference
3. **Ollama** - Local development and testing

## Integration

The NLP Agent (`agents/nlp/`) consumes this serving endpoint
for interview question generation.

```python
# Example usage from NLP Agent
from ml.recruiter_persona.serving.client import RecruiterClient

client = RecruiterClient(endpoint="http://localhost:8000")
question = client.generate(context=conversation_history)
```
