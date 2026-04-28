"""Train the CandidateJobGNN on 14-node per-pair graphs.

Primary node features come from TAPJFNN's g_j / g_r (or raw MiniLM pooled
embeddings if no TAPJFNN checkpoint is available).
Entity node features come from the MiniLM entity extractor.
Edge weights come from the kNN-intersection similarity (Eq. 1).

Optimised for RTX 6000 Pro (Blackwell, 96 GB VRAM):
  * Graphs are precomputed once (encoder / TAPJFNN / extractor are frozen,
    so per-epoch rebuilds in __getitem__ are pure waste) and cached in
    memory, which removes the GPU-bound dataloader bottleneck and lets
    num_workers > 0 actually help.
  * Primary-feature encoding is batched across many examples per forward
    pass instead of one-at-a-time.
  * bf16 autocast on the GNN forward, TF32 matmul, cudnn.benchmark.
  * Much larger default batch size — the 14-node graphs are tiny.
  * Optional torch.compile (--compile) for an extra kick.
"""

from __future__ import annotations

import argparse
import os
from contextlib import nullcontext
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm

from ..config import CHECKPOINT_DIR, ENCODER_CFG, GNN_CFG, TRAIN_CFG, APP_CFG
from ..data.dataset import load_hf_dataset, split_examples, Example
from ..data.entities import EntityExtractor
from ..models.encoder import MiniLMEncoder
from ..models.gnn import BipartiteGraphBuilder, CandidateJobGNN
from ..models.tapjfnn import TAPJFNN
from ..utils.topic_model import load_pair


class CachedGraphDataset(torch.utils.data.Dataset):
    """Indexable list of precomputed PyG Data objects."""

    def __init__(self, graphs: List[Data]) -> None:
        self.graphs = graphs

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Data:
        return self.graphs[idx]


def _collate(batch):
    return Batch.from_data_list(batch)


@torch.no_grad()
def _primary_features_batched(
    examples: List[Example],
    encoder: MiniLMEncoder,
    tapjfnn: TAPJFNN | None,
    lda_j,
    lda_r,
    device: torch.device,
    batch_size: int,
):
    """Return (cand_primary, job_primary) tensors for every example.

    Shapes: (N, D_primary) each. Runs in large forward passes instead of
    one MiniLM / TAPJFNN call per example.
    """
    n = len(examples)
    cand_list: List[torch.Tensor] = []
    job_list: List[torch.Tensor] = []

    use_tapj = tapjfnn is not None and lda_j is not None and lda_r is not None

    for start in tqdm(range(0, n, batch_size), desc="primary features"):
        chunk = examples[start : start + batch_size]
        jobs = [ex.job_text for ex in chunk]
        resumes = [ex.resume_text for ex in chunk]
        if use_tapj:
            tj = torch.as_tensor(lda_j.transform(jobs), dtype=torch.float, device=device)
            tr = torch.as_tensor(lda_r.transform(resumes), dtype=torch.float, device=device)
            out = tapjfnn(jobs, resumes, tj, tr)
            cand_list.append(out["g_r"].detach())
            job_list.append(out["g_j"].detach())
        else:
            # Interleave jobs and resumes so we pay one encoder call per chunk.
            interleaved: List[str] = []
            for j, r in zip(jobs, resumes):
                interleaved.append(j)
                interleaved.append(r)
            pooled = encoder.encode_pooled(interleaved, max_length=256)  # (2B, D)
            pooled = pooled.view(len(chunk), 2, -1).detach()
            job_list.append(pooled[:, 0])
            cand_list.append(pooled[:, 1])

    return torch.cat(cand_list, dim=0), torch.cat(job_list, dim=0)


@torch.no_grad()
def precompute_graphs(
    examples: List[Example],
    encoder: MiniLMEncoder,
    extractor: EntityExtractor,
    builder: BipartiteGraphBuilder,
    tapjfnn: TAPJFNN | None,
    lda_j,
    lda_r,
    device: torch.device,
    encode_batch_size: int,
    store_device: torch.device,
    desc: str,
) -> List[Data]:
    """Build every graph once and return them as a list of Data objects on
    `store_device`. Because the feature-producing modules are frozen, this
    only has to run a single time per split."""
    cand_primaries, job_primaries = _primary_features_batched(
        examples, encoder, tapjfnn, lda_j, lda_r, device, encode_batch_size
    )

    graphs: List[Data] = []
    for idx, ex in enumerate(tqdm(examples, desc=desc)):
        cand_primary = cand_primaries[idx]
        job_primary = job_primaries[idx]
        cand_bag = extractor.extract(ex.resume_text)
        job_bag = extractor.extract(ex.job_text)
        g = builder.build(cand_primary, job_primary, cand_bag, job_bag)
        # Ordinal target (0=No Fit, 1=Potential Fit, 2=Good Fit); training uses
        # this directly so the model learns three levels, not just a yes/no.
        g.y = torch.tensor([ex.label_ordinal], dtype=torch.long)
        if store_device.type != g.x.device.type:
            g = g.to(store_device)
        graphs.append(g)
    return graphs


def _rank_score(logits: torch.Tensor) -> torch.Tensor:
    """Unbounded ranking score: log-odds of Good Fit vs No Fit.

    Training's margin loss operates on this (not on softmax probabilities),
    so gradients don't saturate in the flat softmax region — that's what was
    pinning the score gap near zero under the expected-rank formulation.
    """
    return logits[:, 2] - logits[:, 0]


def _display_score(logits: torch.Tensor) -> torch.Tensor:
    """Squash the unbounded rank score into [0, 1] for reporting / inference."""
    return torch.sigmoid(_rank_score(logits))


def evaluate(model, loader, device, amp_dtype) -> dict:
    model.eval()
    ys, preds, scores = [], [], []
    autocast = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if amp_dtype is not None and device.type == "cuda"
        else nullcontext()
    )
    with torch.no_grad(), autocast:
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            logits = model(batch).float()
            pred = logits.argmax(dim=-1)
            score = _display_score(logits)  # sigmoid(logit_gf - logit_nf), in [0, 1]
            ys.extend(batch.y.cpu().numpy().tolist())
            preds.extend(pred.cpu().numpy().tolist())
            scores.extend(score.cpu().numpy().tolist())
    ys = np.array(ys); preds = np.array(preds); scores = np.array(scores)
    metrics = {
        "accuracy": float(accuracy_score(ys, preds)),
        "macro_f1": float(f1_score(ys, preds, average="macro", zero_division=0)),
    }
    # AUC for separating "Good Fit" from the rest — the thing we really care about.
    is_good = (ys == 2).astype(int)
    if len(np.unique(is_good)) > 1:
        metrics["auc_good_vs_rest"] = float(roc_auc_score(is_good, scores))
    # Pairwise ranking accuracy: over pairs with different ordinal labels, does
    # the model order them correctly? This is the direct "is the gap real?" metric.
    ord_mask = ys[:, None] != ys[None, :]
    ord_correct = (np.sign(scores[:, None] - scores[None, :]) == np.sign(ys[:, None] - ys[None, :]))
    if ord_mask.any():
        metrics["pairwise_rank_acc"] = float(ord_correct[ord_mask].mean())
    # Mean score separation between No Fit and Good Fit — the direct diagnostic
    # for the "gap too narrow" complaint.
    if (ys == 0).any() and (ys == 2).any():
        metrics["score_gap_good_minus_nofit"] = float(scores[ys == 2].mean() - scores[ys == 0].mean())
    return metrics


def _resolve_amp_dtype(name: str) -> torch.dtype | None:
    name = name.lower()
    if name in ("none", "off", "fp32", "float32"):
        return None
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16", "half"):
        return torch.float16
    raise ValueError(f"Unknown precision: {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=TRAIN_CFG.num_epochs)
    ap.add_argument("--batch_size", type=int, default=128,
                    help="Per-step graph batch size. 14-node graphs are tiny, "
                         "but too large a batch starves the optimizer of steps "
                         "on small training splits.")
    ap.add_argument("--encode_batch_size", type=int, default=128,
                    help="Docs per encoder/TAPJFNN forward during precompute.")
    ap.add_argument("--lr", type=float, default=TRAIN_CFG.lr)
    ap.add_argument("--gnn_type", type=str, default=GNN_CFG.gnn_type)
    ap.add_argument("--use_tapjfnn", action="store_true",
                    help="Use TAPJFNN (g_j, g_r) as primary features; else raw MiniLM pooled.")
    ap.add_argument("--precision", type=str, default="bf16",
                    choices=["bf16", "fp16", "fp32", "none"],
                    help="AMP precision for the GNN training loop.")
    ap.add_argument("--num_workers", type=int, default=4,
                    help="Dataloader workers. Safe now that graphs are cached on CPU.")
    ap.add_argument("--cache_device", type=str, default="cpu",
                    choices=["cpu", "cuda"],
                    help="Where to keep precomputed graphs. 'cuda' eliminates "
                         "the H2D copy each step — fits easily in 96 GB.")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the GNN. May trigger recompilation when "
                         "total edge count varies across batches.")
    ap.add_argument("--margin", type=float, default=1.0,
                    help="Target per-rank logit-diff for the pairwise margin loss. "
                         "1.0 means No Fit vs Good Fit (rank diff=2) should differ "
                         "by at least 2.0 in logit space -> ~0.76 gap in sigmoid "
                         "score at inference.")
    ap.add_argument("--lambda_rank", type=float, default=2.0,
                    help="Weight on the pairwise margin ranking loss.")
    args = ap.parse_args()

    # --- Performance knobs (free wins on Blackwell / Ada) -----------------
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store_device = torch.device(args.cache_device if torch.cuda.is_available() else "cpu")
    amp_dtype = _resolve_amp_dtype(args.precision) if device.type == "cuda" else None
    print(f"Device: {device} | cache_device: {store_device} | amp: {amp_dtype}")

    examples = load_hf_dataset(split="train", limit=args.limit, name=APP_CFG.dataset_name)
    train, val, _ = split_examples(
        examples,
        train_ratio=TRAIN_CFG.train_ratio,
        val_ratio=TRAIN_CFG.val_ratio,
        seed=TRAIN_CFG.seed,
    )
    print(f"Train / val: {len(train)} / {len(val)}")

    encoder = MiniLMEncoder(
        model_name=ENCODER_CFG.model_name,
        projection_dim=ENCODER_CFG.projection_dim,
        freeze=True,
    ).to(device).eval()
    extractor = EntityExtractor(encoder)

    tapjfnn = None
    lda_j = lda_r = None
    if args.use_tapjfnn:
        tapjfnn_ckpt = os.path.join(CHECKPOINT_DIR, "tapjfnn_best.pt")
        if os.path.exists(tapjfnn_ckpt):
            tapjfnn = TAPJFNN(encoder=encoder).to(device).eval()
            sd = torch.load(tapjfnn_ckpt, map_location=device)["state_dict"]
            tapjfnn.load_state_dict(sd, strict=False)
            lda_j = load_pair("jd"); lda_r = load_pair("resume")
            if lda_j is None or lda_r is None:
                raise RuntimeError("TAPJFNN requested but LDA models missing.")
        else:
            print("No TAPJFNN checkpoint; falling back to raw MiniLM primary features.")

    primary_dim = ENCODER_CFG.projection_dim
    entity_dim = ENCODER_CFG.projection_dim
    builder = BipartiteGraphBuilder(primary_dim=primary_dim, entity_dim=entity_dim)

    # --- Precompute all graphs ONCE ---------------------------------------
    train_graphs = precompute_graphs(
        train, encoder, extractor, builder, tapjfnn, lda_j, lda_r,
        device=device, encode_batch_size=args.encode_batch_size,
        store_device=store_device, desc="train graphs",
    )
    val_graphs = precompute_graphs(
        val, encoder, extractor, builder, tapjfnn, lda_j, lda_r,
        device=device, encode_batch_size=args.encode_batch_size,
        store_device=store_device, desc="val graphs",
    )

    # Encoder / TAPJFNN are done. Free their VRAM before training.
    del encoder, extractor, tapjfnn
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_ds = CachedGraphDataset(train_graphs)
    val_ds = CachedGraphDataset(val_graphs)

    # When graphs already live on the GPU, workers=0 and pin_memory=False
    # (pinning GPU tensors is invalid; workers would re-copy).
    on_gpu_cache = store_device.type == "cuda"
    loader_kwargs = dict(
        batch_size=args.batch_size,
        collate_fn=_collate,
        num_workers=0 if on_gpu_cache else args.num_workers,
        pin_memory=not on_gpu_cache and device.type == "cuda",
        persistent_workers=(not on_gpu_cache) and args.num_workers > 0,
    )
    train_loader = PyGDataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = PyGDataLoader(val_ds, shuffle=False, **loader_kwargs)

    model = CandidateJobGNN(
        node_feat_dim=builder.node_feat_dim,
        gnn_type=args.gnn_type,
        num_classes=3,
    ).to(device)
    if args.compile:
        model = torch.compile(model, mode="reduce-overhead", dynamic=True)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=TRAIN_CFG.weight_decay)
    sched = CosineAnnealingLR(optim, T_max=args.epochs, eta_min=args.lr * 1e-2)

    # Inverse-frequency class weights over the three ordinal labels.
    counts = [0, 0, 0]
    for ex in train:
        counts[ex.label_ordinal] += 1
    n_train = len(train)
    class_weights = torch.tensor(
        [n_train / (3.0 * c) if c > 0 else 1.0 for c in counts],
        device=device,
    )
    print(f"Class balance (NF/PF/GF): {counts} | weights={class_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Pairwise margin ranking: for every in-batch pair (i, j) with y_i > y_j,
    # enforce score_i - score_j >= margin * (y_i - y_j). This is the mechanism
    # that forces a visible gap between No Fit and Good Fit at inference time.
    margin_per_rank = args.margin
    lambda_rank = args.lambda_rank

    # fp16 needs a grad scaler; bf16 does not.
    use_scaler = amp_dtype is torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    best_score = -float("inf")
    best_path = os.path.join(CHECKPOINT_DIR, "gnn_best.pt")
    patience = 0
    improve_tol = 1e-4

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in tqdm(train_loader, desc=f"epoch {epoch}"):
            batch = batch.to(device, non_blocking=True)
            autocast = (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if amp_dtype is not None and device.type == "cuda"
                else nullcontext()
            )
            with autocast:
                logits = model(batch)
                loss_ce = criterion(logits, batch.y)
                # Ranking score lives in logit space (unbounded), so the
                # margin loss is not clipped by softmax saturation.
                score = _rank_score(logits.float())  # (B,)
                y = batch.y
                diff_y = (y.unsqueeze(0) - y.unsqueeze(1)).float()
                diff_s = score.unsqueeze(0) - score.unsqueeze(1)
                pos_pairs = diff_y > 0
                if pos_pairs.any():
                    required = diff_y[pos_pairs] * margin_per_rank
                    got = diff_s[pos_pairs]
                    loss_rank = F.relu(required - got).mean()
                else:
                    loss_rank = torch.zeros((), device=device)
                loss = loss_ce + lambda_rank * loss_rank

            optim.zero_grad(set_to_none=True)
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CFG.max_grad_norm)
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CFG.max_grad_norm)
                optim.step()
        sched.step()
        metrics = evaluate(model, val_loader, device, amp_dtype)
        print(f"epoch {epoch} val: {metrics}")
        # Primary objective is actually separating Good Fit from the rest by a
        # wide margin, so track AUC + score gap + pairwise rank accuracy.
        score = (
            metrics.get("auc_good_vs_rest", 0.0)
            + 0.5 * metrics.get("pairwise_rank_acc", 0.0)
            + 0.5 * metrics.get("score_gap_good_minus_nofit", 0.0)
        )
        if score > best_score + improve_tol:
            best_score = score
            patience = 0
            # When compiled, unwrap to save the real state dict.
            base_model = getattr(model, "_orig_mod", model)
            torch.save({
                "state_dict": base_model.state_dict(),
                "node_feat_dim": builder.node_feat_dim,
                "gnn_type": args.gnn_type,
                "num_classes": 3,
                "metrics": metrics,
                "use_tapjfnn": args.use_tapjfnn,
            }, best_path)
            print(f"  saved -> {best_path}")
        else:
            patience += 1
            if patience >= TRAIN_CFG.patience:
                print("Early stopping.")
                break


if __name__ == "__main__":
    main()
