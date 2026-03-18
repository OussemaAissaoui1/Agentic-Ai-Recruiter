# Integration Tests

Tests for interactions between components.

## Scope

- Agent-to-agent communication
- Agent-to-memory interactions
- Agent-to-messaging interactions
- Full agent pipelines (e.g., Voice → NLP → Avatar)

## Infrastructure

Uses testcontainers for:
- Redis (memory + messaging)
- Vector DB (Qdrant)

## Example

```python
async def test_voice_to_nlp_pipeline():
    """Test transcript flows from Voice to NLP agent."""
    voice_agent = VoiceAgent()
    nlp_agent = NLPAgent()

    transcript = await voice_agent.transcribe(audio_sample)
    question = await nlp_agent.generate_question(transcript)

    assert question is not None
    assert "?" in question
```
