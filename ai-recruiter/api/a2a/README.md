# A2A Protocol Definitions

Agent-to-Agent communication schemas and agent cards.

## Key Files

| File | Purpose |
|------|---------|
| `agent_card_schema.json` | JSON Schema for agent cards |
| `task_schema.json` | Task request/response schema |
| `artifact_schema.json` | Task artifact definitions |

## Agent Card Structure

```json
{
    "name": "nlp-agent",
    "description": "Handles interview conversation and question generation",
    "version": "1.0.0",
    "capabilities": ["question_generation", "response_analysis"],
    "endpoint": "http://nlp-agent:8080",
    "protocol": "a2a/1.0"
}
```

## Task Types

| Task Type | Agent | Description |
|-----------|-------|-------------|
| `generate_question` | NLP | Generate next interview question |
| `analyze_response` | NLP | Analyze candidate response |
| `transcribe_audio` | Voice | Convert speech to text |
| `analyze_emotion` | Vision | Analyze facial expressions |
| `generate_avatar` | Avatar | Generate talking head video |
| `compute_score` | Scoring | Calculate evaluation scores |
