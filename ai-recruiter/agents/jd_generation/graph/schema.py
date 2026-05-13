"""Constraints + indexes for the JD graph.

`apply_schema(client)` is idempotent — uses `IF NOT EXISTS`. Run at agent
startup (best-effort) and as the first step of `seed.py`.
"""

from __future__ import annotations

import logging
from typing import List

from .client import Neo4jClient

log = logging.getLogger("agents.jd_generation.graph")

_CONSTRAINTS: List[str] = [
    # Unique id constraints — one per node label.
    "CREATE CONSTRAINT employee_id_unique IF NOT EXISTS "
    "FOR (n:Employee) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT role_id_unique IF NOT EXISTS "
    "FOR (n:Role) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT skill_id_unique IF NOT EXISTS "
    "FOR (n:Skill) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT team_id_unique IF NOT EXISTS "
    "FOR (n:Team) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT education_id_unique IF NOT EXISTS "
    "FOR (n:Education) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT priorcompany_id_unique IF NOT EXISTS "
    "FOR (n:PriorCompany) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT jd_id_unique IF NOT EXISTS "
    "FOR (n:JobDescription) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT rejection_id_unique IF NOT EXISTS "
    "FOR (n:RejectionReason) REQUIRE n.id IS UNIQUE",
]

_INDEXES: List[str] = [
    "CREATE INDEX role_family_idx IF NOT EXISTS FOR (n:Role) ON (n.role_family)",
    "CREATE INDEX role_level_idx IF NOT EXISTS FOR (n:Role) ON (n.level)",
    "CREATE INDEX skill_name_idx IF NOT EXISTS FOR (n:Skill) ON (n.name)",
    "CREATE INDEX rejection_created_idx IF NOT EXISTS "
    "FOR (n:RejectionReason) ON (n.created_at)",
    "CREATE INDEX jd_role_family_idx IF NOT EXISTS "
    "FOR (n:JobDescription) ON (n.role_family)",
    "CREATE INDEX jd_status_idx IF NOT EXISTS FOR (n:JobDescription) ON (n.status)",
]


def apply_schema(client: Neo4jClient) -> dict:
    """Apply all constraints + indexes. Idempotent. Returns a count summary."""
    applied = 0
    for stmt in _CONSTRAINTS + _INDEXES:
        client.write(stmt)
        applied += 1
    log.info("jd_generation.schema applied stmts=%d", applied)
    return {"statements_applied": applied}
