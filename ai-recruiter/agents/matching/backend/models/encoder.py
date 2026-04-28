"""MiniLM word-level encoder.

Replacement for the BiLSTM word-level encoder in TAPJFNN (Qin et al. Eqs. 5-6).
TAPJFNN's downstream modules consume a sequence of per-token hidden vectors
(one per word in a requirement / experience), so we MUST return token-level
hidden states, not a pooled sentence vector.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class MiniLMEncoder(nn.Module):
    """Tokenise a batch of text segments and return per-token hidden states."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        projection_dim: int = 400,
        freeze: bool = False,
    ) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.hidden_dim = self.backbone.config.hidden_size
        self.projection_dim = projection_dim

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.proj = nn.Sequential(
            nn.Linear(self.hidden_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def encode(self, texts: List[str], max_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a list of strings.

        Returns:
            tokens:   (B, L, D) per-token projected hidden states.
            mask:     (B, L)    attention mask; 1 = real token.
        """
        if len(texts) == 0:
            return (
                torch.zeros(0, 0, self.projection_dim, device=self.device),
                torch.zeros(0, 0, dtype=torch.long, device=self.device),
            )
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)
        out = self.backbone(**batch)
        tokens = self.proj(out.last_hidden_state)
        return tokens, batch["attention_mask"]

    def encode_pooled(self, texts: List[str], max_length: int = 256) -> torch.Tensor:
        """Return mean-pooled sentence embeddings (B, D). Used for entity
        embeddings and the candidate/job primary-node features in the GNN."""
        tokens, mask = self.encode(texts, max_length=max_length)
        if tokens.numel() == 0:
            return torch.zeros(0, self.projection_dim, device=self.device)
        mask_f = mask.unsqueeze(-1).float()
        pooled = (tokens * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-6)
        return pooled
