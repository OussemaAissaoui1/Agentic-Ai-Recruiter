# Agent Configurations

Per-agent YAML configuration files.

## Key Files

| File | Purpose |
|------|---------|
| `orchestrator.yaml` | Orchestrator timeouts, routing rules |
| `nlp.yaml` | NLP agent model settings, generation params |
| `vision.yaml` | Vision models, frame rate, confidence thresholds |
| `voice.yaml` | ASR model, VAD settings, prosody params |
| `avatar.yaml` | TTS voice, lip sync model, expression mapping |
| `scoring.yaml` | Score weights, competency mappings |

## Example

```yaml
# nlp.yaml
model:
  endpoint: ${MODEL_ENDPOINT}
  timeout_ms: 5000
generation:
  max_tokens: 256
  temperature: 0.7
  top_p: 0.9
```
