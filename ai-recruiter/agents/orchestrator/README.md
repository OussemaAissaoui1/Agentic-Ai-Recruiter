# Orchestrator Agent

Central A2A dispatcher. Today: thin in-process router. Future: LangGraph
state machine driving the full matching → interview → scoring flow.

## Routes (`/api/orchestrator`)

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/health` | Lists registered agents |
| `GET`  | `/agents` | Returns every agent's card |
| `POST` | `/dispatch` | Routes an `A2ATask` envelope to its target agent |

## Dispatcher

`POST /api/orchestrator/dispatch` body is an `A2ATask`:

```json
{
  "task_id": "uuid",
  "sender": "test",
  "target": "matching",
  "capability": "rank",
  "payload": { "job_description": "…", "resumes": [{"id":"a.pdf","text":"…"}] }
}
```

Response is an `A2ATaskResult` (see `core/contracts/a2a.py`).

## Future state machine

Phases: `INTAKE → SCREENING → INVITE → INTERVIEW → SCORING → DECISION`.
Each phase resolves to A2A tasks against the corresponding agent.
LangGraph is wired into pyproject deps; the actual graph is not yet
implemented.
