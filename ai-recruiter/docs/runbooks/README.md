# docs/runbooks/

Operational documentation for running the unified app.

- `unified_app.md` — startup, env vars, route table, lifespan order.
- `fallback_modes.md` — what happens when artifacts (checkpoints,
  weights, gated LLM tokens) are missing.
- `gpu_budget.md` — memory split across vLLM, MediaPipe, Wav2Vec2,
  HSEmotion, MiniLM, GNN.
