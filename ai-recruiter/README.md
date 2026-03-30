# AI Recruiter - Agentic Interview System

Production-ready monorepo for an AI-powered recruitment interview platform using multi-agent architecture.

## Architecture Overview

This system implements a **multi-agent orchestration pattern** with:
- **Orchestrator Agent**: LangGraph state machine coordinating all specialized agents
- **NLP Agent**: Conversation management, question generation, semantic analysis
- **Vision Agent**: Facial expression, emotion, and posture analysis
- **Voice Agent**: ASR, prosody analysis, stress detection
- **Avatar Agent**: Synthetic interviewer face/voice generation
- **Scoring Agent**: Objective candidate evaluation engine

## Directory Structure

```
ai-recruiter/
├── agents/           # All AI agents (orchestrator + specialized)
├── core/             # Shared platform components (memory, messaging, state)
├── ml/               # Fine-tuning, evaluation, and model serving
├── api/              # Interface contracts (gRPC, WebSocket, REST, A2A)
├── data/             # Data lifecycle zones
├── configs/          # Configuration templates
├── docs/             # Architecture and technical documentation
├── tests/            # Test suites
└── scripts/          # Operational scripts
```

## Key Design Principles

1. **Agent Autonomy**: Each agent is independently deployable with clear boundaries
2. **Protocol-First**: A2A protocol for agent communication, Pub/Sub for async events
3. **Shared State**: Redis + Vector DB for conversation memory and context
4. **Recruiter Persona**: Fine-tuned interview model (Alex) trained on data_v2

## Getting Started

See `docs/architecture/` for system design and `docs/agents/` for agent specifications.
