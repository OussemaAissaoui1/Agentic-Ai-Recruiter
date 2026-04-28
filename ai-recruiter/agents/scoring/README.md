# Scoring Agent

Collects per-modality scores into per-session JSONL, exposes a small
REST surface for posting / reading them. Stub today; final aggregation
returns 501.

## Routes (`/api/scoring`)

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/health` | Readiness |
| `POST` | `/{session_id}/score` | Append `{source, kind, value, payload}` |
| `GET`  | `/{session_id}/scores` | Return all records |
| `POST` | `/{session_id}/final` | 501 — not implemented |

## A2A capabilities

`record_score`, `get_scores`, `compute_final` (501).

## Storage

Append-only JSONL at `artifacts/scoring/<session_id>.jsonl`. One line
per record, `{ts, source, kind, value, payload}`.

## Future

- Weighted aggregation across NLP / vision / voice → final hiring
  recommendation.
- Per-job-role weight presets.
- Cross-session calibration to remove rater drift.
