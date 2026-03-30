# Architecture Documentation

System design and architectural decisions.

## Key Documents

| Document | Purpose |
|----------|---------|
| `system_overview.md` | High-level architecture diagram and description |
| `data_flows.md` | Inter-agent communication flows |
| `state_management.md` | Session state and memory architecture |
| `protocols.md` | A2A, gRPC, WebSocket protocol details |
| `decisions/` | Architecture Decision Records (ADRs) |

## Architecture Principles

1. **Agent Autonomy** - Each agent independently deployable
2. **Event-Driven** - Async communication via message broker
3. **Shared Nothing** - State externalized to Redis/Vector DB
4. **Contract-First** - APIs defined before implementation
