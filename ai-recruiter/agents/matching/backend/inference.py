"""End-to-end ranking pipeline: JD + CVs -> fit scores -> CAGA filter."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

from .config import (
    APP_CFG,
    CHECKPOINT_DIR,
    ENCODER_CFG,
    GA_CFG,
    GNN_CFG,
    TAPJFNN_CFG,
)
from .data.entities import EntityExtractor
from .models.encoder import MiniLMEncoder

log = logging.getLogger("agents.matching.inference")
from .models.ga import ConstraintAwareGA, GACandidate, GAConstraints
from .models.gnn import BipartiteGraphBuilder, CandidateJobGNN
from .models.tapjfnn import TAPJFNN
from .utils.topic_model import load_pair


@dataclass
class RankedCandidate:
    id: str
    fit_score: float
    matching_entities: Dict[str, List[str]]
    resume_entities: Dict[str, List[str]]
    job_entities: Dict[str, List[str]]


class RankingPipeline:
    """Single-process pipeline used by the FastAPI layer and CLI."""

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.encoder = MiniLMEncoder(
            model_name=ENCODER_CFG.model_name,
            projection_dim=ENCODER_CFG.projection_dim,
            freeze=False,
        ).to(self.device)
        self.extractor = EntityExtractor(self.encoder)

        self.lda_j = load_pair("jd")
        self.lda_r = load_pair("resume")

        tapjfnn_ckpt = os.path.join(CHECKPOINT_DIR, "tapjfnn_best.pt")
        self.tapjfnn: Optional[TAPJFNN] = None
        self.tapjfnn_load_error: Optional[str] = None
        if os.path.exists(tapjfnn_ckpt) and self.lda_j and self.lda_r:
            try:
                sd = torch.load(tapjfnn_ckpt, map_location=self.device)["state_dict"]
                self.tapjfnn = TAPJFNN(encoder=self.encoder).to(self.device)
                self.tapjfnn.load_state_dict(sd, strict=False)
                self.tapjfnn.eval()
            except (RuntimeError, OSError, EOFError, KeyError) as e:
                # Corrupt or partially-written checkpoint — fall through to the
                # GNN/cosine path instead of crashing the whole pipeline at init.
                self.tapjfnn = None
                self.tapjfnn_load_error = f"{type(e).__name__}: {e}"
                log.warning("tapjfnn checkpoint failed to load (%s); skipping", self.tapjfnn_load_error)

        gnn_ckpt = os.path.join(CHECKPOINT_DIR, "gnn_best.pt")
        self.builder = BipartiteGraphBuilder(
            primary_dim=ENCODER_CFG.projection_dim,
            entity_dim=ENCODER_CFG.projection_dim,
        )
        self.gnn: Optional[CandidateJobGNN] = None
        self.gnn_num_classes: int = 2
        self.gnn_load_error: Optional[str] = None
        if os.path.exists(gnn_ckpt):
            try:
                ck = torch.load(gnn_ckpt, map_location=self.device)
                self.gnn_num_classes = int(ck.get("num_classes", 2))
                self.gnn = CandidateJobGNN(
                    node_feat_dim=ck.get("node_feat_dim", self.builder.node_feat_dim),
                    gnn_type=ck.get("gnn_type", GNN_CFG.gnn_type),
                    num_classes=self.gnn_num_classes,
                ).to(self.device)
                self.gnn.load_state_dict(ck["state_dict"])
                self.gnn.eval()
            except (RuntimeError, OSError, EOFError, KeyError) as e:
                self.gnn = None
                self.gnn_load_error = f"{type(e).__name__}: {e}"
                log.warning("gnn checkpoint failed to load (%s); skipping", self.gnn_load_error)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _primary_embeddings(self, job_text: str, resume_texts: List[str]):
        """Return (cand_primaries, job_primary). Shapes (N, D) and (D,)."""
        if self.tapjfnn is not None and self.lda_j is not None and self.lda_r is not None:
            tj = torch.tensor(self.lda_j.transform([job_text] * len(resume_texts)),
                              dtype=torch.float, device=self.device)
            tr = torch.tensor(self.lda_r.transform(resume_texts),
                              dtype=torch.float, device=self.device)
            out = self.tapjfnn([job_text] * len(resume_texts), resume_texts, tj, tr)
            cands = out["g_r"].detach()       # (N, D)
            job = out["g_j"][0].detach()      # (D,)
            tapjfnn_probs = out["probs"].detach().cpu().numpy()
            return cands, job, tapjfnn_probs
        pooled = self.encoder.encode_pooled([job_text] + resume_texts, max_length=256)
        return pooled[1:], pooled[0], None

    @torch.no_grad()
    def _gnn_scores(self, cand_primaries, job_primary, cand_bags, job_bag) -> Optional[np.ndarray]:
        if self.gnn is None:
            return None
        graphs = []
        for cand_p, cand_bag in zip(cand_primaries, cand_bags):
            graphs.append(self.builder.build(cand_p, job_primary, cand_bag, job_bag))
        batch = Batch.from_data_list(graphs).to(self.device)
        logits = self.gnn(batch)
        if self.gnn_num_classes == 3:
            # Log-odds of Good Fit vs No Fit, squashed to [0, 1]. Training uses
            # the raw logit diff directly, so the gap is not crushed by softmax
            # saturation.
            score = torch.sigmoid(logits[:, 2] - logits[:, 0])
        else:
            score = F.softmax(logits, dim=-1)[:, 1]
        return score.cpu().numpy()

    # ------------------------------------------------------------------ run
    def rank(
        self,
        job_text: str,
        resumes: List[Dict],
    ) -> List[RankedCandidate]:
        """resumes: list of {"id": str, "text": str}."""
        ids = [r["id"] for r in resumes]
        texts = [r["text"] for r in resumes]
        cand_primaries, job_primary, tapjfnn_probs = self._primary_embeddings(job_text, texts)
        cand_bags = [self.extractor.extract(t) for t in texts]
        job_bag = self.extractor.extract(job_text)
        gnn_probs = self._gnn_scores(cand_primaries, job_primary, cand_bags, job_bag)

        # Combine scores: GNN if available else TAPJFNN else cosine similarity.
        if gnn_probs is not None:
            fit = gnn_probs
            scorer = "gnn"
        elif tapjfnn_probs is not None:
            fit = tapjfnn_probs
            scorer = "tapjfnn"
        else:
            # Untrained fallback: cosine similarity between job and resume primaries
            jn = F.normalize(job_primary.unsqueeze(0), dim=-1)
            cn = F.normalize(cand_primaries, dim=-1)
            fit = (cn @ jn.t()).squeeze(-1).cpu().numpy()
            fit = (fit - fit.min()) / (fit.max() - fit.min() + 1e-6)
            scorer = "cosine_fallback"
        log.info(
            "rank: n=%d scorer=%s tapjfnn=%s gnn=%s lda=%s",
            len(resumes), scorer,
            self.tapjfnn is not None,
            self.gnn is not None,
            self.lda_j is not None and self.lda_r is not None,
        )

        out: List[RankedCandidate] = []
        for i, rid in enumerate(ids):
            matching = self._matching_entities(cand_bags[i].phrases, job_bag.phrases)
            out.append(RankedCandidate(
                id=rid,
                fit_score=float(fit[i]),
                matching_entities=matching,
                resume_entities=cand_bags[i].phrases,
                job_entities=job_bag.phrases,
            ))
        out.sort(key=lambda c: c.fit_score, reverse=True)
        return out

    @staticmethod
    def _matching_entities(a: Dict[str, List[str]], b: Dict[str, List[str]]) -> Dict[str, List[str]]:
        matches: Dict[str, List[str]] = {}
        for cat in a:
            if cat not in b:
                continue
            sa = {s.lower() for s in a[cat]}
            sb = {s.lower() for s in b[cat]}
            inter = sorted(sa & sb)
            if inter:
                matches[cat] = inter
        return matches


def apply_ga(
    ranked: List[RankedCandidate],
    ga_inputs: Dict[str, Dict[str, float]],
    constraints: Dict,
) -> Dict:
    """Apply CAGA on the given ranked candidates.

    ga_inputs: {candidate_id: {cost, gender_female, interview_score,
                               cultural_fit, experience_score, salary_alignment}}
    constraints: {budget, min_female_ratio, min_fit_threshold, role_requirements}
    """
    ga_candidates: List[GACandidate] = []
    # Every candidate is "Software Engineer" per the app spec.
    role = "software_engineer"
    for r in ranked:
        meta = ga_inputs.get(r.id, {})
        ga_candidates.append(GACandidate(
            id=r.id,
            skills_match=r.fit_score,  # ML output is the skills-match proxy
            interview_score=float(meta.get("interview_score", 0.0)),
            cultural_fit=float(meta.get("cultural_fit", 0.0)),
            experience_score=float(meta.get("experience_score", 0.0)),
            salary_alignment=float(meta.get("salary_alignment", 0.0)),
            cost=float(meta.get("cost", 0.0)),
            gender_female=int(meta.get("gender_female", 0)),
            role=role,
        ))
    ga = ConstraintAwareGA(
        candidates=ga_candidates,
        constraints=GAConstraints(
            budget=float(constraints.get("budget", 0.0)),
            min_female_ratio=float(constraints.get("min_female_ratio", 0.0)),
            role_requirements=constraints.get("role_requirements", {}) or {},
            min_fit_threshold=float(constraints.get("min_fit_threshold", 0.0)),
        ),
    )
    z, fitness, history = ga.run()
    selected_ids = [ga_candidates[i].id for i in np.where(z == 1)[0]]
    return {
        "selected_ids": selected_ids,
        "fitness": fitness,
        "history": history,
    }
