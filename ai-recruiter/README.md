# AI Recruiter — Unified Platform

One FastAPI app, three stages, multi-agent under the hood:

1. **Pre-screening** — `agents/matching/` ranks CVs against a JD
   (TAPJFNN + 14-node bipartite GNN) and optionally shortlists with a
   constraint-aware GA.
2. **Live interview** — `agents/nlp/` (vLLM Llama-3.1-8B + Qwen scorer +
   refiner + Kokoro TTS) drives the chat. `agents/vision/` runs the
   candidate's webcam in parallel and streams a 4-D behavioral profile
   (cognitive load / arousal / engagement / confidence).
3. **Reports** — `agents/scoring/` + `agents/vision/` produce JSON / PDF
   reports per session.

```
ai-recruiter/
├── main.py                       # unified FastAPI entry
├── core/                         # contracts (BaseAgent, A2A, Health) + runtime
├── agents/
│   ├── matching/                 # CV ↔ JD ranker (TAPJFNN+GNN+CAGA)
│   ├── nlp/                      # recruiter persona (Alex)
│   ├── vision/                   # multimodal behavioral pipeline (HTTP + /ws/vision)
│   ├── avatar/                   # GLB + viseme extraction
│   ├── scoring/                  # JSONL score collection (stub)
│   ├── voice/                    # ASR + prosody (stub)
│   └── orchestrator/             # A2A dispatcher (stub)
├── apps/
│   ├── shell/                    # static landing page
│   ├── screening/                # Vite/React/TS frontend (matching)
│   ├── interview/                # Vite/React frontend (NLP + camera overlay)
│   ├── behavior/                 # Vite/React frontend (vision dashboard)
│   └── static/                   # build outputs (gitignored)
├── configs/                      # runtime.yaml + per-agent configs
├── docs/runbooks/                # unified_app, fallback_modes, gpu_budget
└── tests/                        # smoke tests
```

## Quick start

```bash
# 1. dependencies (vllm separately because of CUDA pin)
pip install -e .
pip install vllm

# 2. build frontends
bash scripts/build_frontends.sh

# 3. run
HF_TOKEN=hf_xxx uvicorn main:app --host 0.0.0.0 --port 8000

# 4. (optional) warm vLLM
curl -X POST http://localhost:8000/api/nlp/warmup
```

Browse: **http://localhost:8000/**

## What's where

- `main.py` builds an `AgentRegistry`, runs each agent's `startup`, and
  mounts every agent's router at `/api/<name>` plus the vision WebSocket
  at `/ws/vision`. The three Vite builds are served as static dirs under
  `/screening/`, `/interview/`, `/behavior/`.
- Each agent inherits `core.contracts.BaseAgent` and ships an
  `agent_card.json` discoverable at `/.well-known/agents/<name>`.
- Health is aggregated at `/api/health` — it always returns 200 and
  surfaces per-agent `state ∈ {ready, loading, fallback, error}` so you
  can tell which fallback modes are active without reading logs.

## Trained artifacts shipped here

| Where | What |
|---|---|
| `agents/matching/artifacts/checkpoints/` | TAPJFNN + GNN checkpoints |
| `agents/matching/artifacts/lda/` | LDA models for TAPJFNN topic attention |
| `agents/vision/models/` | MediaPipe FaceMesh + Pose `.task` files |
| `agents/vision/weights/` | (empty) StudentNet / visual MLP — fallback mode |
| `model_cache/` | vLLM Llama-3.1-8B fine-tune (NLP) |

The vision agent runs in **fallback mode** out of the box because
StudentNet has not been trained yet — see
`docs/runbooks/fallback_modes.md`.

## Documentation

- `docs/runbooks/unified_app.md` — startup, env vars, route table
- `docs/runbooks/fallback_modes.md` — what happens when artifacts are missing
- `docs/runbooks/gpu_budget.md` — memory split across agents
- Per-agent README under `agents/<name>/README.md`
