# Fallback modes

Every agent declares an explicit fallback path. Missing artifacts never
crash startup — the agent simply reports `state=fallback` in its health
probe and the unified `/api/health` endpoint surfaces this.

## Matching agent

| Missing artifact | Path on disk | Effect | Health flag |
|---|---|---|---|
| TAPJFNN ckpt | `agents/matching/artifacts/checkpoints/tapjfnn_best.pt` | drops to MiniLM pooled primaries | `tapjfnn_loaded=false` |
| GNN ckpt | `agents/matching/artifacts/checkpoints/gnn_best.pt` | uses TAPJFNN sigmoid (or cosine if also missing) | `gnn_loaded=false` |
| LDA pickles | `agents/matching/artifacts/lda/lda_{jd,resume}.joblib` | TAPJFNN topic attention disabled → cosine | `lda_loaded=false` |
| Llama-3.2-3B (gated) | requires `HF_TOKEN` env | `/api/matching/extract-signals` returns 503 | `llm.has_token=false` |

Scoring fallback chain: **GNN → TAPJFNN → cosine(MiniLM pooled)**. The
zero-shot entity extractor (MiniLM) always runs — the UI shows matching
entities with zero training.

## Vision agent

| Missing artifact | Path on disk | Effect | Health flag |
|---|---|---|---|
| `face_landmarker.task` | `agents/vision/models/` | visual pipeline returns 503 | `visual_loaded=false` |
| `pose_landmarker.task` | `agents/vision/models/` | body pose disabled | (covered by `visual_loaded`) |
| `hand_landmarker.task` | `agents/vision/models/` | gesture analyzer disabled (no fidget detection) | `gesture_loaded=false` |
| `studentnet.pt` | `agents/vision/weights/` | audio uses untrained classification head — probabilities are not meaningful, calibration still works | `studentnet_weights_present=false` |
| `visual_mlp.pt` | `agents/vision/weights/` | visual stress label disabled; 4-D fusion still computes from features | `visual_mlp_weights_present=false` |

The MediaPipe `.task` files ship with the repo. The MLP weights do not
(and StudentNet is not yet trained). The 4-D dimension scores still
compute — they're feature-driven and z-score-normalised against the
candidate's personal baseline (45 s calibration at session start).

## NLP agent

| Missing artifact | Effect | Health flag |
|---|---|---|
| `model_cache/` empty | `/api/nlp/*` returns 503 with structured error | `engine_ready=false` |
| Qwen scorer model | scorer disabled — main LLM still generates questions; no answer analytics | `scorer_ready=false` |
| Qwen refiner model | refiner disabled — questions land without depth pass | `refiner_ready=false` |
| Kokoro TTS | TTS disabled — `/api/nlp/stream` still streams text | `tts_ready=false` |

## Avatar agent

| Missing artifact | Effect |
|---|---|
| `brunette.glb` | `/api/avatar/glb` returns 503; UI falls back to no avatar |

## What "fallback" looks like upstream

`/api/health` example with no GNN ckpt and no StudentNet weights:

```json
{
  "overall": "fallback",
  "agents": {
    "matching": {
      "state": "fallback",
      "fallback_reasons": ["GNN checkpoint missing — falling back to TAPJFNN/cosine"]
    },
    "vision": {
      "state": "fallback",
      "fallback_reasons": ["StudentNet weights missing — audio uses untrained head"]
    },
    "nlp": { "state": "ready" }
  }
}
```
