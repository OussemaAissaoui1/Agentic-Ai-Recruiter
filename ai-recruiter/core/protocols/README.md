# Protocols Module

Communication protocol implementations (A2A, gRPC, WebSocket).

## Purpose

Implement standardized communication protocols for the system:
- **A2A**: Agent-to-Agent task dispatch protocol
- **gRPC**: Gateway to Orchestrator communication
- **WebSocket**: Real-time client streaming

## Key Files

| File | Purpose |
|------|---------|
| `a2a/client.py` | A2A protocol client |
| `a2a/server.py` | A2A task handler server |
| `a2a/task.py` | A2A task and artifact definitions |
| `grpc/server.py` | gRPC service implementation |
| `grpc/client.py` | gRPC client wrapper |
| `websocket/handler.py` | WebSocket connection handler |
| `websocket/streaming.py` | Bidirectional streaming utilities |

## A2A Protocol

Follows the A2A (Agent-to-Agent) specification:
- `agent_card.json` - Agent discovery metadata
- `tasks/send` - Task dispatch
- `tasks/sendSubscribe` - Streaming task results
