# State Module

Interview session lifecycle and state management.

## Purpose

Manage interview session state across the distributed system:
- Session creation and initialization
- State transitions (phases of interview)
- Session persistence and recovery
- Multi-tenant session isolation

## Key Files

| File | Purpose |
|------|---------|
| `session.py` | Session model and lifecycle |
| `state_machine.py` | Interview phase state machine |
| `persistence.py` | Session state persistence |
| `recovery.py` | Session recovery from failures |

## Session Phases

```
INIT → GREETING → QUESTIONING → FOLLOW_UP → CLOSING → SCORING → COMPLETED
```

## State Schema

```python
class InterviewState(TypedDict):
    session_id: str
    candidate_id: str
    job_role: str
    phase: InterviewPhase
    questions_asked: list[str]
    answers_received: list[str]
    scores: dict[str, float]
    created_at: datetime
    updated_at: datetime
```
