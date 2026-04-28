"""Centralised configuration for the CV Ranking System.

Values here reflect the three papers we implement:
    - Qin et al. 2020 (TAPJFNN)
    - Frazzetto et al. 2025 (GNN, DSE s41019-025-00293-y)
    - Malini et al. 2026 (Constraint-Aware GA, ETASR 13543)

The MiniLM encoder replaces the BiLSTM word-level encoder from TAPJFNN.
"""

import os
from dataclasses import dataclass, field
from typing import List

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
ARTIFACT_DIR = os.path.join(ROOT_DIR, "artifacts")
CHECKPOINT_DIR = os.path.join(ARTIFACT_DIR, "checkpoints")
LDA_DIR = os.path.join(ARTIFACT_DIR, "lda")
LOG_DIR = os.path.join(ARTIFACT_DIR, "logs")
UPLOAD_DIR = os.path.join(ARTIFACT_DIR, "uploads")
for d in (ARTIFACT_DIR, CHECKPOINT_DIR, LDA_DIR, LOG_DIR, UPLOAD_DIR):
    os.makedirs(d, exist_ok=True)


ENTITY_CATEGORIES: List[str] = [
    "soft_skills",
    "hard_skills",
    "education",
    "field_of_education",
    "industry_sector",
    "role",
]

# Natural-language descriptions used by the MiniLM zero-shot entity classifier.
# These stand in for the GPT-4 prompt the paper used. They describe *what* each
# category is and include short exemplar lists so the prototype embedding sits
# closer to typical CV phrasing.
ENTITY_PROTOTYPES: dict = {
    "soft_skills": "interpersonal and behavioural traits such as teamwork, "
                   "leadership, communication, problem solving, adaptability, "
                   "attention to detail, collaboration, ownership, time management, "
                   "critical thinking",
    "hard_skills": "technical skills, tools and technologies such as Python, Java, "
                   "C++, TensorFlow, PyTorch, Docker, Kubernetes, AWS, GCP, SQL, "
                   "React, Node.js, machine learning, deep learning, data engineering",
    "education": "academic degrees, qualifications and certifications such as "
                 "Bachelor of Science, Master's degree, PhD, MBA, AWS certification, "
                 "engineering diploma, granted by universities or institutions",
    "field_of_education": "field of study or academic discipline such as computer "
                          "science, software engineering, electrical engineering, "
                          "applied mathematics, physics, statistics, data science, "
                          "business administration",
    "industry_sector": "industry or business domain such as fintech, banking, "
                       "healthcare, retail, e-commerce, telecommunications, "
                       "consulting, manufacturing, education technology, gaming",
    "role": "job titles and functional roles such as software engineer, backend "
            "developer, frontend developer, data scientist, machine learning "
            "engineer, tech lead, product manager, DevOps engineer, research engineer",
}

# Per-category cosine thresholds. Soft skills are abstract → looser; field/role
# are dense and easily confused with hard_skills → tighter.
ENTITY_THRESHOLDS: dict = {
    "soft_skills": 0.26,
    "hard_skills": 0.30,
    "education": 0.30,
    "field_of_education": 0.32,
    "industry_sector": 0.32,
    "role": 0.30,
}

# Margin between best and second-best category. Filters out ambiguous phrases
# like "computer science" that pull both hard_skills and field_of_education.
ENTITY_TOP2_MARGIN: float = 0.04

# Per-category cap to bound graph size and inference latency on long CVs.
ENTITY_PER_CATEGORY_CAP: int = 50


@dataclass
class EncoderConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    # MiniLM-L6 hidden size
    hidden_dim: int = 384
    # Paper TAPJFNN downstream dim (BiLSTM 200+200). We project MiniLM -> 400 so
    # downstream module shapes match the paper.
    projection_dim: int = 400
    max_tokens_requirement: int = 32
    max_tokens_experience: int = 320
    freeze: bool = False


@dataclass
class TapjfnnConfig:
    """Hierarchical attention sizes from Qin et al. Sect. 6.2."""
    attention_dim: int = 200
    topic_dim_jd: int = 50
    topic_dim_resume: int = 150
    max_requirements: int = 18
    max_experiences: int = 18
    dropout: float = 0.2
    num_classes: int = 2


@dataclass
class GnnConfig:
    """Graph + GNN hyperparameters from Frazzetto et al."""
    k_neighbours: int = 10
    sharpening_p: int = 4
    psychometric_dim: int = 18  # unused in our pipeline; kept for future extension
    use_psychometric: bool = False
    hidden_channels: int = 128
    num_layers: int = 3
    dropout: float = 0.3
    gnn_type: str = "GCN"  # one of: GCN, GAT, GIN, GraphConv
    deep_readout_layers: int = 2
    jumping_knowledge: bool = True
    num_classes: int = 2
    pos_class_weight: float = 1.0


@dataclass
class GaConfig:
    """CAGA hyperparameters (Malini et al.). Paper does not fix numeric values;
    these are reasonable defaults consistent with the convergence behaviour
    reported in Fig. 2 of the paper."""
    population_size: int = 100
    num_generations: int = 50
    crossover_rate: float = 0.8
    mutation_rate: float = 0.0  # set to 1/m at runtime
    tournament_k: int = 3
    elitism: int = 2
    # Fit-score weights, w1..w5 in Eq. 2. Must sum to 1 (Eq. 9).
    w_skills: float = 0.35
    w_interview: float = 0.15
    w_cultural: float = 0.15
    w_experience: float = 0.25
    w_salary: float = 0.10


@dataclass
class TrainingConfig:
    batch_size: int = 16
    num_epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-5
    max_grad_norm: float = 1.0
    patience: int = 5
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    seed: int = 42


@dataclass
class AppConfig:
    dataset_name: str = "cnamuangtoun/resume-job-description-fit"
    # Label map derived from the actual dataset columns (label column is string).
    label_map: dict = field(default_factory=lambda: {
        "No Fit": 0,
        "Potential Fit": 1,
        "Good Fit": 2,
    })


ENCODER_CFG = EncoderConfig()
TAPJFNN_CFG = TapjfnnConfig()
GNN_CFG = GnnConfig()
GA_CFG = GaConfig()
TRAIN_CFG = TrainingConfig()
APP_CFG = AppConfig()
