# Common Agent Utilities

Shared code, base classes, and utilities used across all agents.

## Purpose

- Provide consistent abstractions for agent development
- Reduce code duplication across agents
- Enforce common patterns and interfaces

## Key Components

| File | Purpose |
|------|---------|
| `base_agent.py` | Abstract base class all agents inherit from |
| `a2a_client.py` | A2A protocol client for inter-agent communication |
| `pubsub_client.py` | Redis Streams pub/sub wrapper |
| `health.py` | Health check utilities (liveness, readiness) |
| `logging.py` | Structured logging configuration |
| `metrics.py` | Prometheus metrics helpers |
| `decorators.py` | Common decorators (@retry, @trace, @cache) |
| `exceptions.py` | Custom exception hierarchy |
| `schemas.py` | Shared Pydantic models for agent communication |
