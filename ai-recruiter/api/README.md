# API Contracts

Interface definitions and contracts by protocol.

## Purpose

Define clear API boundaries between system components:
- External API (clients → gateway)
- Internal API (gateway → orchestrator)
- Inter-agent API (agent → agent)

## Modules

| Module | Purpose |
|--------|---------|
| `grpc/` | Protocol Buffer definitions for gRPC services |
| `websocket/` | WebSocket message schemas |
| `rest/` | OpenAPI specs for REST endpoints |
| `a2a/` | A2A agent card and task schemas |

## Contract-First Development

1. Define contracts in this directory
2. Generate client/server stubs
3. Implement in respective services
