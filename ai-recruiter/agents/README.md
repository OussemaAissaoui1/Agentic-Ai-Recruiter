# Agents Directory

Contains all AI agents that power the interview system. Each agent is a self-contained module with its own lifecycle, tools, and communication interface.

## Agent Types

| Agent | Purpose | Protocol |
|-------|---------|----------|
| `orchestrator/` | Central state machine coordinating all agents | A2A dispatch |
| `nlp/` | Conversation, question generation, semantic scoring | A2A + Pub/Sub |
| `vision/` | Facial expression and emotion analysis | A2A + Pub/Sub |
| `voice/` | ASR transcription, prosody and stress analysis | A2A + Pub/Sub |
| `avatar/` | Synthetic interviewer face/voice generation | A2A + Pub/Sub |
| `scoring/` | Global candidate evaluation engine | A2A + Pub/Sub |
| `common/` | Shared agent utilities, base classes, decorators | Internal |

## Inter-Agent Data Flows

```
Voice Agent  ──transcript──>  NLP Agent  ──generated text──>  Avatar Agent
Vision Agent ──emotion signals──────────────────────────────>  Avatar Agent
NLP Agent    ──semantic scores──>  Scoring Agent
Vision Agent ──behavioral scores──>  Scoring Agent
Voice Agent  ──prosody scores──>  Scoring Agent
```

## Agent Contract

Each agent MUST implement:
1. `agent_card.json` - A2A discovery metadata
2. `handle_task()` - Process incoming A2A tasks
3. `health_check()` - Liveness/readiness probes
