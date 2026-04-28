# agents/ — multi-agent runtime

Every subdirectory is an independently deployable agent that conforms to
`core.contracts.BaseAgent`. The unified app at `ai-recruiter/main.py`
imports each one, registers it with the in-process `AgentRegistry`, and
mounts its router under `/api/<name>`.

| Agent | Status | What it does | Mount |
|---|---|---|---|
| `matching/` | ready | CV ↔ JD ranker (TAPJFNN+GNN+CAGA) | `/api/matching` |
| `nlp/` | ready | Recruiter persona (vLLM Llama-3.1-8B + Qwen scorer/refiner + Kokoro TTS) | `/api/nlp` |
| `vision/` | fallback | Multimodal behavioral analysis (MediaPipe + HSEmotion + Wav2Vec2) | `/api/vision` + `/ws/vision` |
| `avatar/` | ready | GLB avatar + viseme extraction | `/api/avatar` |
| `scoring/` | ready (stub) | Per-modality score JSONL | `/api/scoring` |
| `voice/` | stub | ASR + prosody (planned) | `/api/voice` |
| `orchestrator/` | ready (stub) | A2A dispatcher | `/api/orchestrator` |

`vision` is in fallback because StudentNet has not been trained yet
(audio uses untrained head). `voice` is intentionally a stub — the
multimodal vocal-stress signal is owned by `vision`.

## Required surface per agent

```
agents/<name>/
├── agent.py               # contains a class implementing BaseAgent
├── api.py                 # build_router(agent) -> APIRouter   (when needed)
├── agent_card.json        # discovery — exposed at /.well-known/agents/<name>
├── README.md              # what it does, contract, fallback modes
└── …                      # implementation files (models/, weights/, …)
```

Cross-agent communication uses `A2ATask` envelopes (see
`core/contracts/a2a.py`). Routes are HTTP and stable; bypassing them
is fine for in-process speed but `handle_task()` is the canonical path
that the orchestrator (today and future) uses.
