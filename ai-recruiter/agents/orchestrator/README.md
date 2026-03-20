# Orchestrator Agent

The central coordination layer implemented as a **LangGraph state machine**.

## Purpose

- Route incoming interview requests to appropriate specialized agents
- Manage interview session state and transitions
- Coordinate multi-agent workflows (e.g., Voice -> NLP -> Avatar pipeline)
- Handle session lifecycle (start, pause, resume, end)
- Aggregate agent outputs for final interview flow

## Key Components

| File | Purpose |
|------|---------|
| `agent.py` | Main orchestrator agent class with LangGraph graph definition |
| `state.py` | Interview state schema (TypedDict for LangGraph) |
| `nodes.py` | Graph nodes for each processing step |
| `edges.py` | Conditional routing logic between nodes |
| `tools.py` | Tool definitions for agent dispatch |
| `agent_card.json` | A2A discovery metadata |

## State Machine Phases

1. **GREETING** - Avatar introduces, collects basic info
2. **QUESTIONING** - NLP generates questions, Voice captures answers
3. **FOLLOW_UP** - Adaptive follow-ups based on responses
4. **CLOSING** - Interview wrap-up and next steps
5. **SCORING** - Final evaluation aggregation

## Dependencies

- Receives: gRPC from API Gateway
- Dispatches to: All specialized agents via A2A
- Publishes: Session events to Message Broker
