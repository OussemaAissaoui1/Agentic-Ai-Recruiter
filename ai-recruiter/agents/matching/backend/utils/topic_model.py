"""LDA topic models used by TAPJFNN's multi-topic attention (Qin et al.
Sect. 4.2). The paper pre-trains two LDAs offline — one on job postings and
one on resumes — and feeds the fixed topic distribution per document in as
a conditioning vector for the β / δ attention scores.

We persist these with joblib so training + inference share weights.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import joblib
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from ..config import LDA_DIR


@dataclass
class LDAPair:
    vectorizer: CountVectorizer
    lda: LatentDirichletAllocation
    num_topics: int

    def transform(self, docs: List[str]) -> np.ndarray:
        if not docs:
            return np.zeros((0, self.num_topics), dtype=np.float32)
        X = self.vectorizer.transform(docs)
        return self.lda.transform(X).astype(np.float32)


def _fit(docs: List[str], num_topics: int, seed: int = 42) -> LDAPair:
    vec = CountVectorizer(
        max_df=0.95,
        min_df=3,
        stop_words="english",
        max_features=20000,
    )
    X = vec.fit_transform(docs)
    lda = LatentDirichletAllocation(
        n_components=num_topics,
        learning_method="online",
        batch_size=256,
        max_iter=20,
        random_state=seed,
        n_jobs=-1,
    )
    lda.fit(X)
    return LDAPair(vectorizer=vec, lda=lda, num_topics=num_topics)


def fit_and_save(
    job_docs: List[str],
    resume_docs: List[str],
    num_topics_jd: int,
    num_topics_resume: int,
    seed: int = 42,
) -> None:
    os.makedirs(LDA_DIR, exist_ok=True)
    jd = _fit(job_docs, num_topics_jd, seed)
    rs = _fit(resume_docs, num_topics_resume, seed)
    joblib.dump(jd, os.path.join(LDA_DIR, "lda_jd.joblib"))
    joblib.dump(rs, os.path.join(LDA_DIR, "lda_resume.joblib"))


def load_pair(kind: str) -> Optional[LDAPair]:
    path = os.path.join(LDA_DIR, f"lda_{kind}.joblib")
    if not os.path.exists(path):
        return None
    return joblib.load(path)
