"""Entity extraction for the GNN graph builder.

The Frazzetto et al. paper used GPT-4 with iterative human-in-the-loop prompts
to extract six entity categories from resumes and JDs. We reproduce that
behaviour without calling an external LLM by:

    1) Segmenting the document with line-level granularity (preserving the
       structural cues PDFs encode as newlines and bullets), then producing
       sub-fragments by punctuation so a "Python, Java, AWS" line yields
       both the parent line and the three items.
    2) Embedding every phrase with the same MiniLM encoder used by TAPJFNN.
    3) Computing cosine similarity against a MiniLM embedding of a natural
       language description for each of the six categories. Each category
       has its own threshold (config.ENTITY_THRESHOLDS) and we additionally
       require a margin between the best and second-best category to drop
       phrases that straddle two categories (e.g. "computer science").

This stays within the paper's spirit (category is defined by a natural
language description, not a fixed keyword list).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List

import torch

from ..config import (
    ENTITY_CATEGORIES,
    ENTITY_PER_CATEGORY_CAP,
    ENTITY_PROTOTYPES,
    ENTITY_THRESHOLDS,
    ENTITY_TOP2_MARGIN,
)
from ..models.encoder import MiniLMEncoder


# Junk patterns to drop before classification.
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_PHONE_RE = re.compile(r"\+?\d[\d\s\-().]{6,}\d")
_ALPHA_RE = re.compile(r"[A-Za-z]")


@dataclass
class EntityBag:
    """Extracted entity phrases grouped by category with their MiniLM
    embeddings. Fields match the six categories from paper 2."""

    phrases: Dict[str, List[str]] = field(default_factory=dict)
    embeddings: Dict[str, torch.Tensor] = field(default_factory=dict)

    def get_embeddings(self, category: str, device: torch.device, dim: int) -> torch.Tensor:
        """Return (N, D) embeddings for a category (zero tensor if missing)."""
        emb = self.embeddings.get(category)
        if emb is None or emb.numel() == 0:
            return torch.zeros(0, dim, device=device)
        return emb.to(device)


class EntityExtractor:
    """MiniLM-based six-way zero-shot entity classifier."""

    def __init__(self, encoder: MiniLMEncoder, threshold: float = 0.28) -> None:
        self.encoder = encoder
        # Fallback threshold for any category not in ENTITY_THRESHOLDS.
        self.threshold = threshold
        self._prototype_vecs: torch.Tensor | None = None

    @torch.no_grad()
    def _prototypes(self) -> torch.Tensor:
        if self._prototype_vecs is None:
            descriptions = [ENTITY_PROTOTYPES[c] for c in ENTITY_CATEGORIES]
            vecs = self.encoder.encode_pooled(descriptions, max_length=96)
            self._prototype_vecs = torch.nn.functional.normalize(vecs, dim=-1)
        return self._prototype_vecs

    @staticmethod
    def _clean(text: str) -> str:
        text = _URL_RE.sub(" ", text)
        text = _EMAIL_RE.sub(" ", text)
        text = _PHONE_RE.sub(" ", text)
        return text

    @staticmethod
    def _candidate_phrases(text: str) -> List[str]:
        """Multi-granularity segmentation.

        We emit:
          - whole lines (after whitespace normalisation), so list-style
            "Python, Java, AWS" survives as a hard-skill blob;
          - sub-fragments split on bullets / commas / pipes / slashes, so the
            same line also contributes "Python", "Java", "AWS" individually;
          - sentence splits inside long lines.

        The classifier later decides whether each phrase is an entity and of
        which type, so over-generating candidates is intentional — duplicates
        are filtered by lower-cased key.
        """
        if not text:
            return []
        text = EntityExtractor._clean(text)
        out: List[str] = []
        seen: set[str] = set()

        def add(p: str) -> None:
            p = p.strip(" \t.:;,()[]{}|*•-—_/").strip()
            if not p:
                return
            if not _ALPHA_RE.search(p):
                return  # purely numeric / dates / symbols
            tokens = p.split()
            n = len(tokens)
            if n < 1 or n > 14:
                return
            if len(p) < 3 or len(p) > 140:
                return
            key = re.sub(r"\s+", " ", p.lower())
            if key in seen:
                return
            seen.add(key)
            out.append(p)

        for raw_line in re.split(r"[\r\n]+", text):
            line = re.sub(r"[ \t]+", " ", raw_line).strip()
            if not line:
                continue
            add(line)
            for frag in re.split(r"[,;•|·]|\s/\s|\s—\s|\s-\s", line):
                add(frag)
            for frag in re.split(r"\.\s+", line):
                add(frag)

        return out[:300]  # cap for latency

    @torch.no_grad()
    def extract(self, text: str) -> EntityBag:
        phrases = self._candidate_phrases(text)
        if not phrases:
            return EntityBag()

        phrase_vecs = self.encoder.encode_pooled(phrases, max_length=32)
        phrase_vecs_n = torch.nn.functional.normalize(phrase_vecs, dim=-1)
        sims = phrase_vecs_n @ self._prototypes().t()  # (N, 6)

        # Top-2 for the margin filter.
        top2_vals, top2_idx = sims.topk(2, dim=-1)
        best_sim = top2_vals[:, 0]
        second_sim = top2_vals[:, 1]
        best_idx = top2_idx[:, 0]

        per_cat_phrases: Dict[str, List[str]] = {c: [] for c in ENTITY_CATEGORIES}
        per_cat_embs: Dict[str, List[torch.Tensor]] = {c: [] for c in ENTITY_CATEGORIES}

        for i, phrase in enumerate(phrases):
            cat = ENTITY_CATEGORIES[best_idx[i].item()]
            thr = ENTITY_THRESHOLDS.get(cat, self.threshold)
            if best_sim[i].item() < thr:
                continue
            if (best_sim[i] - second_sim[i]).item() < ENTITY_TOP2_MARGIN:
                continue
            per_cat_phrases[cat].append(phrase)
            per_cat_embs[cat].append(phrase_vecs[i].unsqueeze(0))

        bag = EntityBag()
        for cat in ENTITY_CATEGORIES:
            if not per_cat_phrases[cat]:
                continue
            kept_phrases = per_cat_phrases[cat][:ENTITY_PER_CATEGORY_CAP]
            kept_embs = per_cat_embs[cat][:ENTITY_PER_CATEGORY_CAP]
            bag.phrases[cat] = kept_phrases
            bag.embeddings[cat] = torch.cat(kept_embs, dim=0)
        return bag
