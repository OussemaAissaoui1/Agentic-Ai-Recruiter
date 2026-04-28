"""TAPJFNN with MiniLM word encoder.

Implements the four hierarchical attention levels from Qin et al. 2020
"An Enhanced Neural Network Approach to Person-Job Fit in Talent
Recruitment". Equation numbers in comments refer to the journal version.

Replacement vs. original paper:
    Word-level BiLSTM (Eqs. 5-6)  ->  MiniLM last_hidden_state + linear proj
    Everything else (Eqs. 7-17)   ->  implemented exactly as described

Hierarchy:
    j_{l,*}  -> MiniLM   -> h^J_{l,*}
    h^J_{l,*}             -> α word-attn (Eq. 7-8)         -> s^J_l
    {s^J_l}               -> BiLSTM (Eq. 9)                -> {c^J_t}
    {c^J_t}, z^J          -> β topic-attn (Eq. 10-11)      -> g^J
    r_{l,*}  -> MiniLM   -> h^R_{l,*}
    h^R_{l,*}, s^J_k      -> γ ability-attn (Eq. 12-13)   -> s^R_{l,k}
    s^R_l = sum_k s^R_{l,k} (Eq. 14)
    {s^R_l}               -> BiLSTM (Eq. 15)               -> {c^R_t}
    {c^R_t}, g^J, z^J, z^R -> δ exp-attn (Eq. 16)          -> g^R
    Head (Eq. 17)                                          -> y_hat
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ENCODER_CFG, TAPJFNN_CFG
from ..data.segmentation import split_experiences, split_requirements
from .encoder import MiniLMEncoder


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """x: (B, L, D), mask: (B, L). Returns (B, D)."""
    mask_f = mask.unsqueeze(-1).float()
    return (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-6)


class WordAlphaAttention(nn.Module):
    """Eq. 7-8. Shared across all requirements and experiences."""

    def __init__(self, dim: int, attn_dim: int):
        super().__init__()
        self.W = nn.Linear(dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # tokens: (N, L, D). Returns (N, D)
        e = self.v(torch.tanh(self.W(tokens))).squeeze(-1)  # (N, L)
        e = e.masked_fill(mask == 0, -1e9)
        a = F.softmax(e, dim=-1)
        return torch.sum(a.unsqueeze(-1) * tokens, dim=1)


class TopicBetaAttention(nn.Module):
    """Eq. 9-11. BiLSTM over {s^J_l}, then topic-conditioned attention."""

    def __init__(self, dim: int, attn_dim: int, topic_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(dim, dim // 2, batch_first=True, bidirectional=True)
        self.W_c = nn.Linear(dim, attn_dim)
        self.W_z = nn.Linear(topic_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(
        self,
        s_j: torch.Tensor,           # (B, P, D)
        mask: torch.Tensor,          # (B, P)
        topic_j: torch.Tensor,       # (B, Tj)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pack-pad in practice but P is small so we skip packing.
        c, _ = self.lstm(s_j)        # (B, P, D)
        scored = torch.tanh(self.W_c(c) + self.W_z(topic_j).unsqueeze(1))
        e = self.v(scored).squeeze(-1)  # (B, P)
        e = e.masked_fill(mask == 0, -1e9)
        b = F.softmax(e, dim=-1)
        g_j = torch.sum(b.unsqueeze(-1) * c, dim=1)   # (B, D)
        return g_j, c


class AbilityGammaAttention(nn.Module):
    """Eq. 12-13. For every experience l and every requirement k, attend over
    the tokens of experience l conditioned on the requirement representation
    s^J_k. Returns s^R_{l,k}."""

    def __init__(self, dim: int, attn_dim: int):
        super().__init__()
        self.W_s = nn.Linear(dim, attn_dim)
        self.U = nn.Linear(dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(
        self,
        exp_tokens: torch.Tensor,    # (B, Q, Le, D)
        exp_mask: torch.Tensor,      # (B, Q, Le)
        s_j: torch.Tensor,           # (B, P, D)
        req_mask: torch.Tensor,      # (B, P)
    ) -> torch.Tensor:
        B, Q, Le, D = exp_tokens.shape
        P = s_j.size(1)
        # Compute W_s * s^J_k for every k: (B, P, A)
        ws = self.W_s(s_j)
        # Compute U * h^R_{l,t} for every l,t: (B, Q, Le, A)
        uh = self.U(exp_tokens)
        # Broadcast to (B, Q, P, Le, A)
        scored = torch.tanh(uh.unsqueeze(2) + ws.unsqueeze(1).unsqueeze(3))
        e = self.v(scored).squeeze(-1)  # (B, Q, P, Le)
        # Mask padded tokens and padded requirements
        tok_mask = exp_mask.unsqueeze(2).expand(-1, -1, P, -1)
        e = e.masked_fill(tok_mask == 0, -1e9)
        # Softmax jointly over experiences l (Q) and tokens (Le) per
        # requirement k, as in Eq. 13. Reshape to (B, P, Q*Le).
        e_t = e.permute(0, 2, 1, 3).contiguous().view(B, P, Q * Le)
        a = F.softmax(e_t, dim=-1).view(B, P, Q, Le)
        a = a.permute(0, 2, 1, 3).contiguous()  # (B, Q, P, Le)
        # Weighted sum over tokens t=Le: (B, Q, P, D)
        # a: (B,Q,P,t)  tokens: (B,Q,t,D)  -> result (B,Q,P,D)
        s_rlk = torch.einsum("bqpt,bqtd->bqpd", a, exp_tokens)
        # Mask out padded requirements so Eq. 14 sum ignores them
        rmask = req_mask.unsqueeze(1).unsqueeze(-1).float()  # (B,1,P,1)
        s_rlk = s_rlk * rmask
        return s_rlk  # (B, Q, P, D)


class TopicDeltaAttention(nn.Module):
    """Eq. 15-16. BiLSTM over {s^R_l}, then attention conditioned on g^J, z^J, z^R."""

    def __init__(self, dim: int, attn_dim: int, topic_jd_dim: int, topic_resume_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(dim, dim // 2, batch_first=True, bidirectional=True)
        self.W_g = nn.Linear(dim, attn_dim, bias=False)
        self.U_c = nn.Linear(dim, attn_dim)
        self.M_zj = nn.Linear(topic_jd_dim, attn_dim, bias=False)
        self.V_zr = nn.Linear(topic_resume_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(
        self,
        s_r: torch.Tensor,        # (B, Q, D)
        mask: torch.Tensor,       # (B, Q)
        g_j: torch.Tensor,        # (B, D)
        topic_j: torch.Tensor,    # (B, Tj)
        topic_r: torch.Tensor,    # (B, Tr)
    ) -> torch.Tensor:
        c, _ = self.lstm(s_r)  # (B, Q, D)
        scored = torch.tanh(
            self.U_c(c)
            + self.W_g(g_j).unsqueeze(1)
            + self.M_zj(topic_j).unsqueeze(1)
            + self.V_zr(topic_r).unsqueeze(1)
        )
        e = self.v(scored).squeeze(-1)
        e = e.masked_fill(mask == 0, -1e9)
        d = F.softmax(e, dim=-1)
        return torch.sum(d.unsqueeze(-1) * c, dim=1)


class TAPJFNN(nn.Module):
    """Full person-job fit model with MiniLM word encoder."""

    def __init__(
        self,
        encoder: Optional[MiniLMEncoder] = None,
        dim: int = ENCODER_CFG.projection_dim,
        attn_dim: int = TAPJFNN_CFG.attention_dim,
        topic_jd_dim: int = TAPJFNN_CFG.topic_dim_jd,
        topic_resume_dim: int = TAPJFNN_CFG.topic_dim_resume,
        dropout: float = TAPJFNN_CFG.dropout,
    ) -> None:
        super().__init__()
        self.encoder = encoder or MiniLMEncoder(
            model_name=ENCODER_CFG.model_name,
            projection_dim=dim,
            freeze=ENCODER_CFG.freeze,
        )
        self.dim = dim
        self.topic_jd_dim = topic_jd_dim
        self.topic_resume_dim = topic_resume_dim

        self.alpha = WordAlphaAttention(dim, attn_dim)
        self.beta = TopicBetaAttention(dim, attn_dim, topic_jd_dim)
        self.gamma = AbilityGammaAttention(dim, attn_dim)
        self.delta = TopicDeltaAttention(dim, attn_dim, topic_jd_dim, topic_resume_dim)

        # Eq. 17 prediction head.
        # Concat: [g_j, g_r, g_j - g_r, g_j * g_r, z_j, z_r, proj_z_j * proj_z_r]
        self.z_j_proj = nn.Linear(topic_jd_dim, attn_dim)
        self.z_r_proj = nn.Linear(topic_resume_dim, attn_dim)
        head_in = dim * 4 + topic_jd_dim + topic_resume_dim + attn_dim
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(head_in, attn_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(attn_dim, 1),
        )
        self.dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------ utils
    def _encode_segments(
        self,
        segments_per_doc: List[List[str]],
        max_tokens: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode a batch of variable-length segment lists.

        Returns:
            tokens:    (B, S_max, L_max, D)
            tok_mask:  (B, S_max, L_max)   - which tokens are real
            seg_mask:  (B, S_max)          - which segments are real
        """
        B = len(segments_per_doc)
        S_max = max(1, max(len(s) for s in segments_per_doc))
        flat: List[str] = []
        seg_lengths: List[int] = []
        for segs in segments_per_doc:
            seg_lengths.append(len(segs))
            flat.extend(segs)
        if not flat:
            dev = self.encoder.device
            return (
                torch.zeros(B, S_max, 1, self.dim, device=dev),
                torch.zeros(B, S_max, 1, dtype=torch.long, device=dev),
                torch.zeros(B, S_max, dtype=torch.long, device=dev),
            )
        tokens_flat, mask_flat = self.encoder.encode(flat, max_length=max_tokens)
        L_max = tokens_flat.size(1)
        dev = tokens_flat.device
        tokens = torch.zeros(B, S_max, L_max, self.dim, device=dev)
        tok_mask = torch.zeros(B, S_max, L_max, dtype=torch.long, device=dev)
        seg_mask = torch.zeros(B, S_max, dtype=torch.long, device=dev)
        cursor = 0
        for i, n in enumerate(seg_lengths):
            if n == 0:
                continue
            tokens[i, :n] = tokens_flat[cursor:cursor + n]
            tok_mask[i, :n] = mask_flat[cursor:cursor + n]
            seg_mask[i, :n] = 1
            cursor += n
        return tokens, tok_mask, seg_mask

    # ------------------------------------------------------------- forward
    def forward(
        self,
        job_texts: List[str],
        resume_texts: List[str],
        topic_j: torch.Tensor,       # (B, Tj)
        topic_r: torch.Tensor,       # (B, Tr)
    ) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        reqs_per_doc = [split_requirements(t, max_items=TAPJFNN_CFG.max_requirements)
                        for t in job_texts]
        exps_per_doc = [split_experiences(t, max_items=TAPJFNN_CFG.max_experiences)
                        for t in resume_texts]
        # Guarantee at least one segment so shapes are well-defined.
        reqs_per_doc = [r if r else [t[:200] or "."] for r, t in zip(reqs_per_doc, job_texts)]
        exps_per_doc = [e if e else [t[:1000] or "."] for e, t in zip(exps_per_doc, resume_texts)]

        j_tokens, j_tok_mask, j_seg_mask = self._encode_segments(
            reqs_per_doc, ENCODER_CFG.max_tokens_requirement
        )
        r_tokens, r_tok_mask, r_seg_mask = self._encode_segments(
            exps_per_doc, ENCODER_CFG.max_tokens_experience
        )
        B, P, Lj, D = j_tokens.shape
        Q, Le = r_tokens.size(1), r_tokens.size(2)

        # --- α over words in each requirement (Eq. 7-8)
        s_j = self.alpha(
            j_tokens.view(B * P, Lj, D),
            j_tok_mask.view(B * P, Lj),
        ).view(B, P, D)
        s_j = s_j * j_seg_mask.unsqueeze(-1)  # zero out padded requirements

        # --- β topic-conditioned attention over requirements (Eq. 9-11)
        g_j, _ = self.beta(s_j, j_seg_mask, topic_j)

        # --- γ ability-aware attention over resume tokens per requirement k (Eq. 12-13)
        # Work experience by experience to bound memory.
        s_rlk = self.gamma(r_tokens, r_tok_mask, s_j, j_seg_mask)  # (B, Q, P, D)

        # --- sum over requirement index k (Eq. 14)
        s_r = s_rlk.sum(dim=2)  # (B, Q, D)
        s_r = s_r * r_seg_mask.unsqueeze(-1)

        # --- δ topic-conditioned attention over experiences (Eq. 15-16)
        g_r = self.delta(s_r, r_seg_mask, g_j, topic_j, topic_r)

        # --- prediction head (Eq. 17)
        diff = g_j - g_r
        prod = g_j * g_r
        topic_inter = self.z_j_proj(topic_j) * self.z_r_proj(topic_r)
        feat = torch.cat(
            [g_j, g_r, diff, prod, topic_j, topic_r, topic_inter], dim=-1
        )
        logits = self.head(self.dropout(feat)).squeeze(-1)

        return {
            "logits": logits,
            "probs": torch.sigmoid(logits),
            "g_j": g_j,
            "g_r": g_r,
        }
