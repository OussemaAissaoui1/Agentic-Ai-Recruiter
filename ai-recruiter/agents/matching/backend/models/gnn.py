"""GNN for candidate-job matching.

Re-implementation of Frazzetto et al. 2025 "Graph Neural Networks for
Candidate-Job Matching: An Inductive Learning Approach" (DSE).

Graph schema (Sect. 4.1, Fig. 1):
    - 2 primary nodes: candidate, job_description
    - 6 entity nodes per side = 12 entity nodes total
    - Total 14 nodes, minimum 15 edges:
        self-loop on each primary (2)
        cand - jd (1)
        primary - its entity stars (12)
        + 0-6 weighted cross entity edges

Edge weights (Eq. 1): kNN-intersection similarity raised to the 1/p power.
    sim(cand_cat, jd_cat) = (shared_subset_count / (|E_c| + |E_j|))^(1/p)

Node features (Eq. 2): entity nodes use DeepSets-style concat(mean, sum, max)
of MiniLM embeddings of the extracted entity phrases. Primary nodes use
MiniLM mean-pool of the full document (optionally concatenated with
psychometric features).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import (
    GATConv,
    GCNConv,
    GINConv,
    GraphConv,
    GraphNorm,
    JumpingKnowledge,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

from ..config import ENCODER_CFG, ENTITY_CATEGORIES, GNN_CFG

# NOTE: we intentionally do NOT import EntityBag to avoid a circular import
# (entities.py imports MiniLMEncoder; models/__init__.py imports gnn).
# Duck-typing below: any object exposing .get_embeddings(cat, device, dim) works.


def _pool_entity(embeddings: torch.Tensor, dim: int) -> torch.Tensor:
    """DeepSets concat(mean, sum, max). embeddings: (N, D) -> (3*D,)."""
    if embeddings.numel() == 0:
        return torch.zeros(3 * dim, device=embeddings.device if embeddings.is_cuda else None)
    mean = embeddings.mean(dim=0)
    sum_ = embeddings.sum(dim=0)
    max_ = embeddings.max(dim=0).values
    return torch.cat([mean, sum_, max_], dim=0)


def _knn_intersection_similarity(
    cand_emb: torch.Tensor,
    jd_emb: torch.Tensor,
    k: int,
    sharpen_p: int,
) -> float:
    """Eq. 1 from the paper.

    For every pair (vc, vj) of entity instances in a given category, check
    whether the k-NN neighbourhood of vc (within the combined instance set)
    is a subset of the k-NN neighbourhood of vj. Count the number of ordered
    pairs that satisfy the subset condition and normalise by
    (|E_c| + |E_j|); raise to 1/sharpen_p.
    """
    Nc, Nj = cand_emb.size(0), jd_emb.size(0)
    if Nc == 0 or Nj == 0:
        return 0.0
    all_emb = F.normalize(torch.cat([cand_emb, jd_emb], dim=0), dim=-1)
    sim = all_emb @ all_emb.t()
    # Exclude self from neighbour sets.
    sim.fill_diagonal_(-float("inf"))
    k_eff = max(1, min(k, all_emb.size(0) - 1))
    # Top-k neighbour sets (indices). For each node i -> neighbour set N_i.
    nbrs = sim.topk(k_eff, dim=-1).indices  # (N, k_eff)
    # Build bitmasks
    N = all_emb.size(0)
    mask = torch.zeros(N, N, dtype=torch.bool, device=cand_emb.device)
    idx = torch.arange(N, device=cand_emb.device).unsqueeze(-1).expand(-1, k_eff)
    mask[idx.flatten(), nbrs.flatten()] = True
    # For each (i in cand, j in jd) check N_i subset of N_j
    cand_mask = mask[:Nc]                          # (Nc, N)
    jd_mask = mask[Nc:]                            # (Nj, N)
    # subset(i, j) = not any (in N_i and not in N_j)
    # Broadcast: (Nc, 1, N) & (1, Nj, N)
    violation = cand_mask.unsqueeze(1) & ~jd_mask.unsqueeze(0)  # (Nc, Nj, N)
    is_subset = ~violation.any(dim=-1)                           # (Nc, Nj)
    count = is_subset.sum().item()
    raw = count / (Nc + Nj)
    return float(raw ** (1.0 / sharpen_p))


class BipartiteGraphBuilder:
    """Build one 14-node graph per candidate-job pair."""

    def __init__(
        self,
        primary_dim: int,
        entity_dim: int,
        k: int = GNN_CFG.k_neighbours,
        sharpen_p: int = GNN_CFG.sharpening_p,
    ):
        self.primary_dim = primary_dim
        self.entity_dim = entity_dim  # dim of a single entity embedding (projection_dim)
        self.entity_feat_dim = 3 * entity_dim  # mean + sum + max
        self.k = k
        self.sharpen_p = sharpen_p
        # Final node feature dim: pad primary to entity_feat_dim for stack().
        self.node_feat_dim = max(primary_dim, self.entity_feat_dim)

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(0) >= self.node_feat_dim:
            return x[: self.node_feat_dim]
        return F.pad(x, (0, self.node_feat_dim - x.size(0)))

    def build(
        self,
        cand_primary: torch.Tensor,   # (D_primary,)
        job_primary: torch.Tensor,    # (D_primary,)
        cand_bag,                     # EntityBag (duck-typed)
        job_bag,                      # EntityBag (duck-typed)
    ) -> Data:
        device = cand_primary.device
        node_features: List[torch.Tensor] = []
        # 0: candidate, 1: job
        node_features.append(self._pad(cand_primary))
        node_features.append(self._pad(job_primary))

        cand_entity_idx: Dict[str, int] = {}
        job_entity_idx: Dict[str, int] = {}
        for cat in ENTITY_CATEGORIES:
            cand_entity_idx[cat] = len(node_features)
            node_features.append(
                _pool_entity(cand_bag.get_embeddings(cat, device, self.entity_dim), self.entity_dim)
            )
            job_entity_idx[cat] = len(node_features)
            node_features.append(
                _pool_entity(job_bag.get_embeddings(cat, device, self.entity_dim), self.entity_dim)
            )

        x = torch.stack(node_features, dim=0)  # (14, node_feat_dim)

        edges: List[List[int]] = []
        weights: List[float] = []

        def _add(a: int, b: int, w: float):
            edges.append([a, b])
            edges.append([b, a])
            weights.append(w)
            weights.append(w)

        # Self-loops on primary nodes (Sect. 4.1)
        edges.append([0, 0]); weights.append(1.0)
        edges.append([1, 1]); weights.append(1.0)
        # Direct candidate - job edge
        _add(0, 1, 1.0)
        # Star topology: primary <-> its entity nodes
        for cat in ENTITY_CATEGORIES:
            _add(0, cand_entity_idx[cat], 1.0)
            _add(1, job_entity_idx[cat], 1.0)
        # Cross entity-to-entity edges (same category) with kNN-intersection weight
        for cat in ENTITY_CATEGORIES:
            w = _knn_intersection_similarity(
                cand_bag.get_embeddings(cat, device, self.entity_dim),
                job_bag.get_embeddings(cat, device, self.entity_dim),
                k=self.k,
                sharpen_p=self.sharpen_p,
            )
            if w > 0.0:
                _add(cand_entity_idx[cat], job_entity_idx[cat], w)

        edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
        edge_weight = torch.tensor(weights, dtype=torch.float, device=device)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_weight)


# -------------------------------------------------------- GNN backbone
class _ConvBlock(nn.Module):
    """One conv + GraphNorm + LeakyReLU + dropout (paper Sect. 5.2.1-2)."""

    def __init__(
        self,
        conv: nn.Module,
        hidden: int,
        dropout: float,
        uses_edge_weight: bool,
    ):
        super().__init__()
        self.conv = conv
        self.norm = GraphNorm(hidden)
        self.act = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(dropout)
        self.uses_edge_weight = uses_edge_weight

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor],
        batch: torch.Tensor,
    ) -> torch.Tensor:
        if self.uses_edge_weight and edge_weight is not None:
            x = self.conv(x, edge_index, edge_weight)
        else:
            x = self.conv(x, edge_index)
        x = self.norm(x, batch)
        x = self.act(x)
        x = self.dropout(x)
        return x


def _make_conv(kind: str, in_dim: int, out_dim: int):
    kind = kind.upper()
    if kind == "GCN":
        return GCNConv(in_dim, out_dim, add_self_loops=False), True
    if kind == "GAT":
        return GATConv(in_dim, out_dim, heads=1, add_self_loops=False), False
    if kind == "GIN":
        mlp = nn.Sequential(nn.Linear(in_dim, out_dim), nn.LeakyReLU(0.1), nn.Linear(out_dim, out_dim))
        return GINConv(mlp, train_eps=True), False
    if kind == "GRAPHCONV":
        return GraphConv(in_dim, out_dim), True
    raise ValueError(f"Unknown GNN type: {kind}")


class CandidateJobGNN(nn.Module):
    """Pre-aggregation + conv stack + JK + concat readout + deep readout head."""

    def __init__(
        self,
        node_feat_dim: int,
        hidden: int = GNN_CFG.hidden_channels,
        num_layers: int = GNN_CFG.num_layers,
        gnn_type: str = GNN_CFG.gnn_type,
        dropout: float = GNN_CFG.dropout,
        deep_readout_layers: int = GNN_CFG.deep_readout_layers,
        use_jk: bool = GNN_CFG.jumping_knowledge,
        num_classes: int = GNN_CFG.num_classes,
    ):
        super().__init__()
        # Pre-aggregation projection (paper uses 7 independent linears; we
        # share one linear since all our nodes live in the same space — the
        # paper's split is required only when primary/entity features have
        # different native dims, which they do not in our implementation).
        self.pre = nn.Linear(node_feat_dim, hidden)

        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            conv, uses_w = _make_conv(gnn_type, hidden, hidden)
            self.blocks.append(_ConvBlock(conv, hidden, dropout, uses_w))

        self.use_jk = use_jk
        if use_jk:
            self.jk = JumpingKnowledge(mode="cat")
            jk_out = hidden * num_layers
        else:
            self.jk = None
            jk_out = hidden

        # Concat(sum, mean, max) readout
        readout_in = 3 * jk_out
        layers: List[nn.Module] = []
        d = readout_in
        for _ in range(deep_readout_layers):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Dropout(dropout))
            d = hidden
        layers.append(nn.Linear(d, num_classes))
        self.head = nn.Sequential(*layers)

    def forward(self, batch: Batch) -> torch.Tensor:
        x = self.pre(batch.x)
        edge_index = batch.edge_index
        edge_weight = getattr(batch, "edge_attr", None)
        bvec = batch.batch
        xs = []
        for block in self.blocks:
            x = block(x, edge_index, edge_weight, bvec)
            xs.append(x)
        if self.use_jk:
            x = self.jk(xs)
        pooled = torch.cat(
            [global_mean_pool(x, bvec), global_max_pool(x, bvec), global_add_pool(x, bvec)],
            dim=-1,
        )
        return self.head(pooled)  # (B, num_classes)
