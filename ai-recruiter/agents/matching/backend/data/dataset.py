"""HuggingFace resume/JD dataset loader.

Dataset: cnamuangtoun/resume-job-description-fit
Columns: resume_text, job_description_text, label
label is one of: 'No Fit', 'Potential Fit', 'Good Fit' (3-way).

We map this to a binary target (1 if label != 'No Fit') for the paper-1
BCE head and keep the ordinal label separate for the paper-2 CORAL variant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from ..config import APP_CFG


@dataclass
class Example:
    job_text: str
    resume_text: str
    label_ordinal: int  # 0,1,2
    label_binary: int   # 1 only if ordinal == 2 (Good Fit)


class ResumeJobDataset(Dataset):
    def __init__(self, examples: List[Example]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        ex = self.examples[idx]
        return {
            "job_text": ex.job_text,
            "resume_text": ex.resume_text,
            "label_ordinal": ex.label_ordinal,
            "label_binary": ex.label_binary,
        }


def _row_to_example(row: Dict) -> Optional[Example]:
    job = (row.get("job_description_text") or "").strip()
    resume = (row.get("resume_text") or "").strip()
    label_str = (row.get("label") or "").strip()
    if not job or not resume or label_str not in APP_CFG.label_map:
        return None
    ordinal = APP_CFG.label_map[label_str]
    return Example(
        job_text=job,
        resume_text=resume,
        label_ordinal=ordinal,
        label_binary=1 if ordinal == 2 else 0,
    )


def load_hf_dataset(
    split: str = "train",
    limit: Optional[int] = None,
    name: str = None,
) -> List[Example]:
    """Load and decode the HF resume/JD fit dataset."""
    name = name or APP_CFG.dataset_name
    ds = load_dataset(name, split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    out: List[Example] = []
    for row in ds:
        ex = _row_to_example(row)
        if ex is not None:
            out.append(ex)
    return out


def collate_batch(batch: List[Dict]) -> Dict:
    return {
        "job_texts": [b["job_text"] for b in batch],
        "resume_texts": [b["resume_text"] for b in batch],
        "labels_ordinal": torch.tensor([b["label_ordinal"] for b in batch], dtype=torch.long),
        "labels_binary": torch.tensor([b["label_binary"] for b in batch], dtype=torch.float),
    }


def split_examples(
    examples: List[Example],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Example], List[Example], List[Example]]:
    import random
    rng = random.Random(seed)
    idx = list(range(len(examples)))
    rng.shuffle(idx)
    n = len(idx)
    t = int(n * train_ratio)
    v = int(n * (train_ratio + val_ratio))
    train = [examples[i] for i in idx[:t]]
    val = [examples[i] for i in idx[t:v]]
    test = [examples[i] for i in idx[v:]]
    return train, val, test
