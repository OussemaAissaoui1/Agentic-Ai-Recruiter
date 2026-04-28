# core/ — platform primitives

Wire contracts and runtime composition shared by every agent.

```
core/
├── contracts/        # frozen schemas: BaseAgent, A2ATask, AgentCard, HealthStatus
├── runtime/          # AgentRegistry + lifespan composer
└── observability/    # JSON-structured logging
```

## Contracts (`core/contracts/`)

`BaseAgent` is the abstract base that every agent under `agents/` must
implement. It promises five things:

| Method | Why |
|---|---|
| `startup(app_state)` | Cheap init; heavy model loads stay lazy. |
| `shutdown()` | Release resources. |
| `health()` | Non-blocking probe → `HealthStatus`. |
| `card()` | Discovery metadata → `AgentCard`. |
| `handle_task(task)` | A2A entry point — dispatched by the orchestrator. |
| `router` | The FastAPI router mounted at `/api/<name>`. |

`A2ATask` and `A2ATaskResult` are pydantic envelopes. The transport
today is in-process (see `core/runtime/registry.py`). Swapping to gRPC
or Pub/Sub later is a *transport* change — schemas stay frozen.

`HealthState` enum: `loading`, `ready`, `fallback`, `error`. Surfaced at
`/api/health` (aggregated) and `/api/<agent>/health` (per-agent).

## Runtime (`core/runtime/`)

- `AgentRegistry` — holds live `BaseAgent` instances, exposes
  `dispatch(task)` for the orchestrator and `health_all()` for the
  global probe.
- `build_lifespan(registry, startup_order)` — returns the FastAPI
  lifespan that drives ordered startup / shutdown across agents.

## Observability (`core/observability/`)

`configure_logging(level)` once at app boot. `bind_agent_logger(name)`
returns a logger pre-bound with the `agent` field so every JSON line is
attributable.
