# Core Platform Components

Shared runtime infrastructure used by all agents.

## Purpose

Provide consistent, reusable platform capabilities:
- **Memory**: Shared state and context management
- **Messaging**: Pub/Sub event distribution
- **State**: Interview session state management
- **Protocols**: A2A and communication protocol implementations

## Modules

| Module | Purpose |
|--------|---------|
| `memory/` | Redis + Vector DB for conversation memory |
| `messaging/` | Redis Streams pub/sub wrapper |
| `state/` | Session state management |
| `protocols/` | A2A, gRPC, WebSocket protocol handlers |

## Design Principles

1. Agents are stateless - all state lives in `memory/`
2. All inter-agent communication goes through `messaging/`
3. Session lifecycle managed by `state/`
