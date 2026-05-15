"""
Custom evaluation metrics for the recruiter persona trainer.

What you get on every eval step (wandb columns of the same name):
  - eval_loss                  : cross-entropy over assistant tokens (HF default)
  - eval_perplexity            : exp(eval_loss)
  - eval_token_accuracy_top1   : fraction of positions where argmax matches label
  - eval_token_accuracy_top5   : fraction where the label is in the top-5
  - eval_mean_nll              : per-token NLL averaged over assistant tokens
                                 (same number as eval_loss, included for symmetry)
  - eval_nll_p50/p95/p99       : NLL percentiles — catches "fat tail" tokens the
                                 mean hides
  - eval_loss_turn_{0..4}      : loss broken down by assistant-turn index
                                 (turn 0 = first recruiter reply, etc.)
  - eval_loss_turn_5plus       : aggregate loss for turns 5+
  - eval_train_eval_gap        : eval_loss minus the latest train loss seen
                                 (overfit indicator; written by a callback —
                                  not by this module)

How:
  preprocess_logits_for_metrics() runs on every eval batch and reduces the
  (B, T, V) logits tensor (potentially 30+ GB at V=128k) into a (B, T-1, 3)
  summary tensor: [argmax, top5-hit, per-token-NLL]. Memory stays bounded.

  compute_metrics() then aggregates across the full eval set.

Limitation:
  Predictive entropy H(p) requires materializing the full softmax, which is
  too expensive on Llama-3's 128k vocab at the batch size we use. If you need
  entropy, run a separate small-batch pass — see scripts/probe_entropy.py
  (not implemented; entropy is omitted from the live training loop).
"""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from transformers import TrainerCallback


def preprocess_logits_for_metrics(
    logits: Any,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Shrink (B, T, V) logits → (B, T-1, 3) summary.

    Channels along the last axis:
      0 → top-1 argmax prediction id
      1 → top-5 hit indicator (1.0 if true label is in top-5, else 0.0)
      2 → per-token NLL, with masked positions still populated (filtered later
          using the label tensor)
    """
    if isinstance(logits, tuple):
        # Some model heads return (logits, hidden_states, …); we only want logits.
        logits = logits[0]

    # Causal-LM shift: position t in shifted_logits predicts position t+1 in labels.
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    B, T_minus_1, V = shift_logits.shape

    # Top-1: cheap argmax over vocab.
    top1 = shift_logits.argmax(dim=-1)

    # Top-5: gather top-5 ids and check if the true label is among them.
    # Use clamp(min=0) so -100 labels don't index out of range; they're masked
    # out downstream by compute_metrics.
    safe_labels = shift_labels.clamp(min=0)
    top5_idx = shift_logits.topk(5, dim=-1).indices  # (B, T-1, 5)
    top5_hit = (top5_idx == safe_labels.unsqueeze(-1)).any(dim=-1).float()

    # Per-token NLL via cross_entropy with reduction='none'.
    # We cast to float32 — bf16 cross_entropy occasionally returns NaN on extreme logits.
    nll = F.cross_entropy(
        shift_logits.reshape(-1, V).float(),
        shift_labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).reshape(B, T_minus_1)

    # Stack into (B, T-1, 3). This is what gets returned to HF's evaluation loop
    # in place of the full logits — small enough to concatenate over the full
    # eval set without OOM.
    return torch.stack([top1.float(), top5_hit, nll], dim=-1)


def compute_metrics(eval_pred) -> Dict[str, float]:
    """Aggregate per-token tensors into scalar metrics.

    eval_pred.predictions has shape (N, T-1, 3) — the output of
    preprocess_logits_for_metrics, concatenated across batches.
    eval_pred.label_ids has shape (N, T) — the raw labels (with -100 masks).
    """
    preds = eval_pred.predictions
    labels = eval_pred.label_ids

    # Convert from torch tensor if needed (HF passes numpy for compute_metrics).
    if hasattr(preds, "numpy"):
        preds = preds.numpy()
    if hasattr(labels, "numpy"):
        labels = labels.numpy()
    preds = np.asarray(preds)
    labels = np.asarray(labels)

    # Align with the shift inside preprocess_logits_for_metrics.
    shift_labels = labels[..., 1:]
    top1 = preds[..., 0].astype(np.int64)
    top5_hit = preds[..., 1].astype(np.float32)
    nll = preds[..., 2].astype(np.float32)

    valid = shift_labels != -100
    n_valid = int(valid.sum())
    if n_valid == 0:
        # No assistant tokens at all — sanity check fired in masking.
        return {
            "token_accuracy_top1": 0.0,
            "token_accuracy_top5": 0.0,
            "mean_nll":            float("nan"),
            "perplexity":          float("nan"),
            "n_valid_tokens":      0,
        }

    top1_correct = (top1 == shift_labels) & valid
    top5_correct = (top5_hit > 0.5) & valid
    nll_valid = nll[valid]
    mean_nll = float(nll_valid.mean())

    metrics: Dict[str, float] = {
        "token_accuracy_top1": float(top1_correct.sum()) / n_valid,
        "token_accuracy_top5": float(top5_correct.sum()) / n_valid,
        "mean_nll":            mean_nll,
        "perplexity":          float(math.exp(min(mean_nll, 50.0))),  # cap to avoid overflow
        "nll_p50":             float(np.percentile(nll_valid, 50)),
        "nll_p95":             float(np.percentile(nll_valid, 95)),
        "nll_p99":             float(np.percentile(nll_valid, 99)),
        "n_valid_tokens":      n_valid,
    }

    # Per-turn loss — detect turn boundaries as transitions from masked to
    # unmasked positions along the time dimension. Each transition marks the
    # start of a new assistant turn.
    if valid.ndim == 2 and valid.shape[1] > 0:
        prev = np.concatenate(
            [np.zeros((valid.shape[0], 1), dtype=bool), valid[:, :-1]], axis=1
        )
        is_turn_start = valid & ~prev                  # (N, T-1)
        turn_idx = np.cumsum(is_turn_start, axis=1) - 1  # 0-indexed per row

        for t in range(5):
            m = (turn_idx == t) & valid
            if m.any():
                metrics[f"loss_turn_{t}"] = float(nll[m].mean())

        m_late = (turn_idx >= 5) & valid
        if m_late.any():
            metrics["loss_turn_5plus"] = float(nll[m_late].mean())

    return metrics


class TrainEvalGapCallback(TrainerCallback):
    """Callback that writes `train_eval_gap = eval_loss - last_train_loss`
    after each eval. Lets you watch overfitting directly in wandb.
    """

    def __init__(self):
        super().__init__()
        self.last_train_loss: float | None = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        if "loss" in logs:
            self.last_train_loss = float(logs["loss"])
        if "eval_loss" in logs and self.last_train_loss is not None:
            logs["train_eval_gap"] = float(logs["eval_loss"]) - self.last_train_loss
