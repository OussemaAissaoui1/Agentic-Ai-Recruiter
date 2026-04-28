"""Train TAPJFNN on the HuggingFace fit-classification dataset.

Uses binary BCE on the label_binary target (Good/Potential -> 1, No Fit -> 0).
The trained model is used at inference time to produce (g_j, g_r) document
embeddings that feed the GNN primary nodes.
"""

from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import CHECKPOINT_DIR, ENCODER_CFG, TAPJFNN_CFG, TRAIN_CFG, APP_CFG
from ..data.dataset import ResumeJobDataset, collate_batch, load_hf_dataset, split_examples
from ..models.tapjfnn import TAPJFNN
from ..utils.topic_model import load_pair


def _topic_tensors(pair_jd, pair_resume, jobs: List[str], resumes: List[str], device):
    if pair_jd is None or pair_resume is None:
        raise RuntimeError(
            "LDA models not found. Run `python -m backend.scripts.fit_lda` first."
        )
    tj = torch.tensor(pair_jd.transform(jobs), dtype=torch.float, device=device)
    tr = torch.tensor(pair_resume.transform(resumes), dtype=torch.float, device=device)
    return tj, tr


def evaluate(model, loader, lda_j, lda_r, device) -> dict:
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            tj, tr = _topic_tensors(lda_j, lda_r, batch["job_texts"], batch["resume_texts"], device)
            out = model(batch["job_texts"], batch["resume_texts"], tj, tr)
            ps.extend(out["probs"].cpu().numpy().tolist())
            ys.extend(batch["labels_binary"].numpy().tolist())
    ys = np.array(ys); ps = np.array(ps)
    preds = (ps > 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(ys, preds)),
        "f1": float(f1_score(ys, preds, zero_division=0)),
    }
    if len(np.unique(ys)) > 1:
        metrics["auc"] = float(roc_auc_score(ys, ps))
    else:
        metrics["auc"] = 0.5
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=TRAIN_CFG.num_epochs)
    ap.add_argument("--batch_size", type=int, default=TRAIN_CFG.batch_size)
    ap.add_argument("--lr", type=float, default=TRAIN_CFG.lr)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    examples = load_hf_dataset(split="train", limit=args.limit, name=APP_CFG.dataset_name)
    train, val, _ = split_examples(
        examples,
        train_ratio=TRAIN_CFG.train_ratio,
        val_ratio=TRAIN_CFG.val_ratio,
        seed=TRAIN_CFG.seed,
    )
    print(f"Train / val: {len(train)} / {len(val)}")

    train_loader = DataLoader(
        ResumeJobDataset(train), batch_size=args.batch_size,
        shuffle=True, collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        ResumeJobDataset(val), batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_batch,
    )

    model = TAPJFNN().to(device)
    lda_j = load_pair("jd")
    lda_r = load_pair("resume")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=TRAIN_CFG.weight_decay)
    sched = CosineAnnealingLR(optim, T_max=args.epochs, eta_min=args.lr * 1e-2)
    # Class-imbalance weight from the training split
    pos = sum(1 for e in train if e.label_binary == 1)
    neg = len(train) - pos
    pos_weight = torch.tensor([max(neg / max(pos, 1), 1.0)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_f1 = 0.0
    best_path = os.path.join(CHECKPOINT_DIR, "tapjfnn_best.pt")
    patience = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for batch in pbar:
            tj, tr = _topic_tensors(lda_j, lda_r, batch["job_texts"], batch["resume_texts"], device)
            labels = batch["labels_binary"].to(device)
            out = model(batch["job_texts"], batch["resume_texts"], tj, tr)
            loss = criterion(out["logits"], labels)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CFG.max_grad_norm)
            optim.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        sched.step()

        metrics = evaluate(model, val_loader, lda_j, lda_r, device)
        print(f"epoch {epoch} val: {metrics}")
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            patience = 0
            torch.save({"state_dict": model.state_dict(), "metrics": metrics}, best_path)
            print(f"  saved -> {best_path}")
        else:
            patience += 1
            if patience >= TRAIN_CFG.patience:
                print("Early stopping.")
                break


if __name__ == "__main__":
    main()

