# Unified app — runbook

The AI Recruiter platform runs as **one** FastAPI process that mounts every
agent's router, the WebSocket for the vision agent, and the three Vite
sub-apps as static directories.

## Start it

```bash
cd ai-recruiter

# 1. (one-time) install deps
pip install -e .
pip install vllm                       # heavyweight, install separately

# 2. (one-time) build the three frontends
bash scripts/build_frontends.sh

# 3. run
HF_TOKEN=hf_xxx uvicorn main:app --host 0.0.0.0 --port 8000

# 4. (after first request) warm up vLLM (optional but recommended)
curl -X POST http://localhost:8000/api/nlp/warmup
```

Open `http://localhost:8000/` to land on the shell.

## Routes

| Path | Owner | Purpose |
|---|---|---|
| `/` | shell | Static index linking to the three sub-apps |
| `/screening/` | matching | CV ↔ JD ranking UI |
| `/interview/` | nlp + vision | Live chat + camera-driven behavioral overlay |
| `/behavior/` | vision | Standalone behavioral dashboard |
| `/api/matching/*` | matching | rank, extract-signals, health |
| `/api/nlp/*` | nlp | session, chat, stream, tts, warmup |
| `/api/vision/*` | vision | session, mark question, report |
| `/ws/vision` | vision | WebSocket — JPEG frames + PCM in, scores out |
| `/api/avatar/*` | avatar | GLB + viseme extraction |
| `/api/scoring/*` | scoring | record / read per-modality scores |
| `/api/orchestrator/*` | orchestrator | A2A dispatcher |
| `/api/health` | platform | aggregated readiness |
| `/.well-known/agents/{name}` | platform | per-agent card |
| `/docs` | FastAPI | OpenAPI |

## Environment variables

| Var | Default | Purpose |
|---|---|---|
| `AIR_HOST` | `0.0.0.0` | Bind address |
| `AIR_PORT` | `8000` | Bind port |
| `AIR_LOG_LEVEL` | `INFO` | Logging level |
| `AIR_GPU_VLLM_FRACTION` | `0.55` | Slice of GPU memory for vLLM |
| `HF_TOKEN` | unset | Required for `/api/matching/extract-signals` (gated Llama-3.2-3B) |
| `LLM_MODEL` | `meta-llama/Llama-3.2-3B-Instruct` | LLM the matching extractor loads |

See `configs/runtime.yaml` for the full set of knobs.

## Healthcheck contract

`GET /api/health` always returns 200. The body has:

```json
{
  "version": "0.2.0",
  "overall": "ready" | "loading" | "fallback" | "error",
  "agents": { "matching": { "state": "ready", … }, … }
}
```

`overall=ready` means every agent is ready. `fallback` means at least one
agent is running on a documented fallback path (e.g. matching with no GNN
checkpoint; vision with no StudentNet weights). `loading` means at least
one heavy model is still warming up. `error` means at least one agent
failed init irrecoverably.

## Lifespan order

`matching → avatar → vision → voice → scoring → orchestrator → nlp`. Cheap
inits run in parallel; vLLM stays lazy.

## What runs where

| Component | Device | Memory |
|---|---|---|
| vLLM Llama-3.1-8B (NLP) | GPU | ~13 GB at fraction=0.55 on 24 GB cards |
| MediaPipe FaceMesh + Pose + Hands | CPU | ~150 MB |
| HSEmotion EfficientNet-B0 | GPU if free, else CPU | ~50 MB |
| Wav2Vec2-base | GPU if free, else CPU | ~360 MB |
| MiniLM-L6-v2 (matching) | GPU if free, else CPU | ~90 MB |
| TAPJFNN + GNN (matching) | GPU if free, else CPU | <300 MB |
| Qwen2.5-1.5B scorer/refiner | CPU | ~3 GB each |
| Kokoro TTS | CPU | ~200 MB |
