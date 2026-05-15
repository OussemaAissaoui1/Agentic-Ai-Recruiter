# artifacts/

Runtime-generated files — gitignored.

- `scoring/` — per-session JSONL written by the scoring agent.
- `uploads/` — temporary CV upload area used by the matching agent.
- `vision/reports/` — PDF + JSON behavioral reports from the vision agent.

Trained model artifacts live with their owning agent under
`agents/<name>/artifacts/` or `agents/<name>/weights/`.
