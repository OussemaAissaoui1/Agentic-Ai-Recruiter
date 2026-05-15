"""JD Generation agent — graph-RAG over a Neo4j employee/role/skill graph
with a rejection-feedback loop.

Public API:
    >>> from agents.jd_generation import JDGenerationAgent
    >>> from agents.jd_generation import generate_jd, reject_jd, approve_jd, get_jd

The agent is mounted at /api/jd/* by main.py. Modules inside the parent
project can also call the service functions directly without HTTP.
"""

from __future__ import annotations

from .agent import JDGenerationAgent
from .service import (
    RejectionError,
    approve_jd,
    generate_jd,
    generate_jd_stream,
    get_jd,
    reject_jd,
)

__version__ = "0.1.0"

__all__ = [
    "JDGenerationAgent",
    "RejectionError",
    "approve_jd",
    "generate_jd",
    "generate_jd_stream",
    "get_jd",
    "reject_jd",
]
