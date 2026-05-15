# Matching Agent

**Stage 1 — pre-interview screening.** Ranks a batch of CVs against a single JD,
then optionally runs a Constraint-Aware GA shortlist on top.

## What it does

| Step | Component | Source |
|---|---|---|
| 1. Encode JD + CV | MiniLM-L6-v2 | `backend/models/encoder.py` |
| 2. Hierarchical attention (α/β/γ/δ) | TAPJFNN | `backend/models/tapjfnn.py` |
| 3. 14-node bipartite graph + 3-class head | GNN (GCN/GAT/GIN/GraphConv) | `backend/models/gnn.py` |
| 4. Fit score | `sigmoid(logit_GoodFit − logit_NoFit)` | `backend/inference.py::_gnn_scores` |
| 5. *(optional)* Shortlist | Constraint-Aware GA (CAGA) | `backend/models/ga.py` |
| 5a. *(optional)* GA signals | LLM extractor (Llama-3.2-3B) | `backend/agents/cv_extractor.py` |

Implementation tracks three papers — TAPJFNN (Qin et al. 2020), Inductive GNN
(Frazzetto et al. 2025), CAGA (Malini et al. 2026). See the original
`Paper/cv-ranking-system/README.md` for the paper-to-code map.

## HTTP surface (mounted at `/api/matching`)

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/health` | Per-agent readiness + fallback reasons |
| `POST` | `/rank`            | multipart upload of CVs + JD; returns sorted fit scores |
| `POST` | `/extract-signals` | multipart upload; returns per-CV GA-ready metadata |

The `/rank` endpoint accepts `apply_ga_flag=true` plus `metadata_json` and
`constraints_json` form fields to run the GA in the same request.

## A2A capabilities

`rank`, `extract_signals`, `apply_ga`. See `agent_card.json`.

## Trained artifacts

These are checked in under `agents/matching/artifacts/`:

```
artifacts/
├── checkpoints/
│   ├── tapjfnn_best.pt    (~100 MB)
│   └── gnn_best.pt        (~1.5 MB)
├── lda/
│   ├── lda_jd.joblib
│   └── lda_resume.joblib
├── uploads/   (gitignored — temp upload area)
└── logs/      (gitignored)
```

## Fallback chain (when artifacts are missing)

| Available | Score source |
|---|---|
| GNN ckpt (3-class) | `sigmoid(logit_GoodFit − logit_NoFit)` |
| GNN ckpt (2-class, legacy) | `softmax(logits)[:, 1]` |
| TAPJFNN only | TAPJFNN sigmoid prob |
| Nothing | Min-max-normalised cosine similarity (MiniLM pooled) |

The MiniLM zero-shot entity extractor always runs, so the UI shows matching
entities even with zero training. Status surfaces in `GET /health` as
`tapjfnn_loaded`, `gnn_loaded`, `lda_loaded` flags.

## LLM signal extraction

`/extract-signals` requires `HF_TOKEN` env var (gated Llama-3.2-3B). When
unset, the endpoint returns 503 — the rest of the agent keeps working. The
original Paper repo embedded a token in source; that token has been rotated
and the env-var path is the only one supported here.

## Retraining

Use the original training scripts under `backend/scripts/` from the agent
directory (`agents/matching/`):

```bash
python -m agents.matching.backend.scripts.fit_lda --limit 2000
python -m agents.matching.backend.scripts.train_tapjfnn --limit 2000 --epochs 5
python -m agents.matching.backend.scripts.train_gnn --epochs 30 --use_tapjfnn
```

Outputs land in `agents/matching/artifacts/{checkpoints,lda}/` automatically.
