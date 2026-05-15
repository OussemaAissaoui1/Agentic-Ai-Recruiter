"""Cypher queries for the JD graph.

Each function takes a `Neo4jClient` plus parameters and returns plain dicts /
lists ready to surface back to the LLM. Keep the SQL-like surface here — the
tool wrappers in `tools/retrieval.py` add LLM-tool descriptors and call these.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .client import Neo4jClient

# ---------------------------------------------------------------------------
# find_role_family — fuzzy resolve a free-text role title
# ---------------------------------------------------------------------------
def find_role_family(
    client: Neo4jClient, role_title: str, level: Optional[str] = None
) -> Dict[str, Any]:
    """Resolve a free-text role title to a (role_family, level) tuple.

    Strategy: case-insensitive substring match against `Role.title`, then
    `Role.role_family`. If level is provided, prefer roles at that level.
    Returns the best match plus alternatives.
    """
    title = (role_title or "").strip().lower()
    if not title:
        return {"matched": None, "alternatives": [], "reason": "empty role_title"}

    rows = client.read(
        """
        MATCH (r:Role)
        WITH r,
             toLower(r.title) AS t,
             toLower(r.role_family) AS f
        WHERE t CONTAINS $needle OR f CONTAINS $needle OR $needle CONTAINS f
        RETURN r.id AS id, r.title AS title, r.level AS level,
               r.role_family AS role_family, r.department AS department
        """,
        needle=title,
    )

    if not rows:
        return {"matched": None, "alternatives": [], "reason": "no role matched"}

    if level:
        level_norm = level.strip().lower()
        scored = sorted(
            rows,
            key=lambda r: (
                0 if (r["level"] or "").lower() == level_norm else 1,
                -len(r["role_family"] or ""),
            ),
        )
    else:
        scored = rows

    return {
        "matched":      scored[0],
        "alternatives": scored[1:5],
        "reason":       "ok",
    }


# ---------------------------------------------------------------------------
# get_cohort — current holders of a (role_family, level)
# ---------------------------------------------------------------------------
def get_cohort(
    client: Neo4jClient, role_family: str, level: str, k: int = 20
) -> Dict[str, Any]:
    rows = client.read(
        """
        MATCH (e:Employee)-[:HOLDS]->(r:Role)
        WHERE r.role_family = $role_family AND r.level = $level
          AND e.status = 'active'
        OPTIONAL MATCH (e)-[:BELONGS_TO]->(t:Team)
        RETURN e.id AS employee_id,
               e.tenure_years AS tenure_years,
               e.hire_date AS hire_date,
               t.id AS team_id,
               t.name AS team_name
        ORDER BY e.tenure_years DESC
        LIMIT $k
        """,
        role_family=role_family, level=level, k=k,
    )
    return {
        "role_family":   role_family,
        "level":         level,
        "size":          len(rows),
        "employee_ids":  [r["employee_id"] for r in rows],
        "members":       rows,
    }


# ---------------------------------------------------------------------------
# get_skill_distribution — universal / distinctive / long_tail bucketing
# ---------------------------------------------------------------------------
def get_skill_distribution(
    client: Neo4jClient, employee_ids: List[str]
) -> Dict[str, Any]:
    """For a cohort, return skills bucketed by frequency.

    Thresholds (from spec):
        universal:   > 80% cohort frequency
        distinctive: 40-80%
        long_tail:   < 40%
    """
    if not employee_ids:
        return {"cohort_size": 0, "universal": [], "distinctive": [], "long_tail": []}

    rows = client.read(
        """
        MATCH (e:Employee)-[h:HAS_SKILL]->(s:Skill)
        WHERE e.id IN $ids
        WITH s, count(DISTINCT e) AS holders, avg(h.proficiency) AS avg_prof
        RETURN s.id AS skill_id,
               s.name AS name,
               s.category AS category,
               holders,
               avg_prof
        ORDER BY holders DESC
        """,
        ids=employee_ids,
    )

    n = len(employee_ids)
    universal: List[Dict[str, Any]] = []
    distinctive: List[Dict[str, Any]] = []
    long_tail: List[Dict[str, Any]] = []
    for r in rows:
        share = r["holders"] / n
        bucket = {"skill_id": r["skill_id"], "name": r["name"], "category": r["category"],
                  "frequency": round(share, 3), "holders": r["holders"],
                  "avg_proficiency": round(r["avg_prof"], 2)}
        if share > 0.80:
            universal.append(bucket)
        elif share >= 0.40:
            distinctive.append(bucket)
        else:
            long_tail.append(bucket)
    return {
        "cohort_size": n,
        "universal":   universal,
        "distinctive": distinctive,
        "long_tail":   long_tail[:30],  # cap noise
    }


# ---------------------------------------------------------------------------
# get_career_paths_into — paths in the PROGRESSES_TO graph terminating here
# ---------------------------------------------------------------------------
def get_career_paths_into(
    client: Neo4jClient, role_family: str, depth: int = 3
) -> Dict[str, Any]:
    """Career paths leading into the given role family.

    Walks the `PROGRESSES_TO` graph backward up to `depth` hops and returns
    the most-frequent paths plus the immediate predecessor role mix.
    """
    depth = max(1, min(int(depth), 5))
    rows = client.read(
        f"""
        MATCH path = (prev:Role)-[:PROGRESSES_TO*1..{depth}]->(cur:Role)
        WHERE cur.role_family = $role_family
        WITH [r IN nodes(path) | {{
                id:          r.id,
                title:       r.title,
                level:       r.level,
                role_family: r.role_family
             }}] AS steps,
             [rel IN relationships(path) | rel.frequency] AS freqs
        RETURN steps, freqs
        ORDER BY size(steps) DESC, reduce(s = 0, f IN freqs | s + coalesce(f,0)) DESC
        LIMIT 20
        """,
        role_family=role_family,
    )
    immediate = client.read(
        """
        MATCH (prev:Role)-[p:PROGRESSES_TO]->(cur:Role)
        WHERE cur.role_family = $role_family
        RETURN prev.role_family AS role_family, prev.level AS level,
               sum(p.frequency) AS frequency
        ORDER BY frequency DESC
        """,
        role_family=role_family,
    )
    return {
        "role_family":           role_family,
        "depth":                 depth,
        "paths":                 rows,
        "immediate_predecessors": immediate,
    }


# ---------------------------------------------------------------------------
# get_education_distribution
# ---------------------------------------------------------------------------
def get_education_distribution(
    client: Neo4jClient, employee_ids: List[str]
) -> Dict[str, Any]:
    if not employee_ids:
        return {"cohort_size": 0, "by_field": [], "by_degree": [],
                "by_tier": [], "self_taught_pct": 0.0}

    by_field = client.read(
        """
        MATCH (e:Employee)-[:GRADUATED_FROM]->(d:Education)
        WHERE e.id IN $ids
        RETURN d.field AS field, count(*) AS n
        ORDER BY n DESC
        """,
        ids=employee_ids,
    )
    by_degree = client.read(
        """
        MATCH (e:Employee)-[:GRADUATED_FROM]->(d:Education)
        WHERE e.id IN $ids
        RETURN d.degree AS degree, count(*) AS n
        ORDER BY n DESC
        """,
        ids=employee_ids,
    )
    by_tier = client.read(
        """
        MATCH (e:Employee)-[:GRADUATED_FROM]->(d:Education)
        WHERE e.id IN $ids
        RETURN d.institution_tier AS tier, count(*) AS n
        ORDER BY n DESC
        """,
        ids=employee_ids,
    )
    with_edu = client.read(
        """
        MATCH (e:Employee)
        WHERE e.id IN $ids
        OPTIONAL MATCH (e)-[:GRADUATED_FROM]->(d:Education)
        WITH e, count(d) AS edu_count
        RETURN sum(CASE WHEN edu_count = 0 THEN 1 ELSE 0 END) AS no_edu,
               count(e) AS total
        """,
        ids=employee_ids,
    )[0]
    total = max(1, with_edu["total"])
    return {
        "cohort_size":     with_edu["total"],
        "by_field":        by_field,
        "by_degree":       by_degree,
        "by_tier":         by_tier,
        "self_taught_pct": round(100.0 * with_edu["no_edu"] / total, 1),
    }


# ---------------------------------------------------------------------------
# get_prior_industries
# ---------------------------------------------------------------------------
def get_prior_industries(
    client: Neo4jClient, employee_ids: List[str]
) -> Dict[str, Any]:
    if not employee_ids:
        return {"cohort_size": 0, "by_industry": [], "by_size_bucket": []}

    by_industry = client.read(
        """
        MATCH (e:Employee)-[:WORKED_AT]->(c:PriorCompany)
        WHERE e.id IN $ids
        RETURN c.industry AS industry, count(*) AS n
        ORDER BY n DESC
        """,
        ids=employee_ids,
    )
    by_size = client.read(
        """
        MATCH (e:Employee)-[:WORKED_AT]->(c:PriorCompany)
        WHERE e.id IN $ids
        RETURN c.size_bucket AS size_bucket, count(*) AS n
        ORDER BY n DESC
        """,
        ids=employee_ids,
    )
    return {
        "cohort_size":     len(employee_ids),
        "by_industry":     by_industry,
        "by_size_bucket":  by_size,
    }


# ---------------------------------------------------------------------------
# get_team_signals
# ---------------------------------------------------------------------------
def get_team_signals(client: Neo4jClient, team_id: str) -> Dict[str, Any]:
    rows = client.read(
        """
        MATCH (t:Team {id: $team_id})
        OPTIONAL MATCH (e:Employee)-[:BELONGS_TO]->(t)
        OPTIONAL MATCH (e)-[:HOLDS]->(r:Role)
        RETURN t.id AS team_id, t.name AS name, t.department AS department, t.size AS size,
               collect(DISTINCT {
                   employee_id: e.id, level: e.level,
                   role_family: r.role_family, role_title: r.title
               }) AS members
        """,
        team_id=team_id,
    )
    if not rows:
        return {"team_id": team_id, "found": False}
    row = rows[0]
    members = [m for m in row["members"] if m.get("employee_id")]
    by_level: Dict[str, int] = {}
    by_family: Dict[str, int] = {}
    for m in members:
        by_level[m["level"] or "unknown"] = by_level.get(m["level"] or "unknown", 0) + 1
        by_family[m["role_family"] or "unknown"] = by_family.get(m["role_family"] or "unknown", 0) + 1
    return {
        "team_id":     row["team_id"],
        "name":        row["name"],
        "department":  row["department"],
        "size":        row["size"],
        "found":       True,
        "by_level":    by_level,
        "by_family":   by_family,
        "members":     members,
    }


# ---------------------------------------------------------------------------
# get_past_jds
# ---------------------------------------------------------------------------
def get_past_jds(
    client: Neo4jClient, role_family: str,
    outcome: str = "good_hire", k: int = 3,
) -> Dict[str, Any]:
    rows = client.read(
        """
        MATCH (j:JobDescription)
        WHERE j.role_family = $role_family
          AND ( ($outcome = 'any')
             OR (coalesce(j.hire_outcome, '') = $outcome) )
        RETURN j.id AS id,
               j.text AS text,
               j.status AS status,
               j.hire_outcome AS hire_outcome,
               j.created_at AS created_at
        ORDER BY j.created_at DESC
        LIMIT $k
        """,
        role_family=role_family, outcome=outcome, k=k,
    )
    return {"role_family": role_family, "outcome": outcome, "jds": rows}


# ---------------------------------------------------------------------------
# get_past_rejections — feedback-loop core
# ---------------------------------------------------------------------------
def get_past_rejections(
    client: Neo4jClient, role_family: str, k: int = 10
) -> Dict[str, Any]:
    """Past rejections for this role family, most recent first.

    The agent runner calls this DETERMINISTICALLY (never delegated to the
    model). The returned `reasons` are placed in the model's context so it
    can address each one in the new draft.
    """
    rows = client.read(
        """
        MATCH (j:JobDescription)-[:REJECTED_FOR]->(r:RejectionReason)
        WHERE j.role_family = $role_family
        RETURN j.id AS jd_id,
               j.text AS jd_snippet,
               r.id AS reason_id,
               r.text AS reason,
               r.categories AS categories,
               r.created_at AS created_at
        ORDER BY r.created_at DESC
        LIMIT $k
        """,
        role_family=role_family, k=k,
    )
    return {"role_family": role_family, "count": len(rows), "reasons": rows}


# ---------------------------------------------------------------------------
# store_rejection
# ---------------------------------------------------------------------------
def store_rejection(
    client: Neo4jClient,
    jd_id: str,
    reason_id: str,
    reason_text: str,
    categories: List[str],
    created_at: str,
) -> Dict[str, Any]:
    """Persist a RejectionReason + REJECTED_FOR edge; flip JD status."""
    rows = client.write(
        """
        MATCH (j:JobDescription {id: $jd_id})
        MERGE (r:RejectionReason {id: $reason_id})
        SET r.text = $reason_text,
            r.categories = $categories,
            r.created_at = $created_at
        MERGE (j)-[:REJECTED_FOR]->(r)
        SET j.status = 'rejected'
        RETURN j.id AS jd_id, r.id AS reason_id
        """,
        jd_id=jd_id, reason_id=reason_id, reason_text=reason_text,
        categories=categories, created_at=created_at,
    )
    if not rows:
        return {"ok": False, "reason": f"JD {jd_id} not found"}
    return {"ok": True, **rows[0]}


# ---------------------------------------------------------------------------
# JobDescription CRUD — used by service.py
# ---------------------------------------------------------------------------
def create_jd(
    client: Neo4jClient,
    jd_id: str,
    text: str,
    role_family: str,
    created_at: str,
) -> Dict[str, Any]:
    client.write(
        """
        MERGE (j:JobDescription {id: $jd_id})
        SET j.text = $text,
            j.status = 'draft',
            j.role_family = $role_family,
            j.created_at = $created_at,
            j.hire_outcome = null
        WITH j
        MATCH (r:Role {id: 'role_' + $role_family + '_senior'})
        MERGE (j)-[:GENERATED_FOR_ROLE_FAMILY]->(r)
        """,
        jd_id=jd_id, text=text, role_family=role_family, created_at=created_at,
    )
    return {"jd_id": jd_id, "status": "draft"}


def approve_jd_node(client: Neo4jClient, jd_id: str) -> Dict[str, Any]:
    rows = client.write(
        """
        MATCH (j:JobDescription {id: $jd_id})
        SET j.status = 'approved'
        RETURN j.id AS jd_id
        """,
        jd_id=jd_id,
    )
    return {"ok": bool(rows), "jd_id": jd_id}


def get_jd_node(client: Neo4jClient, jd_id: str) -> Optional[Dict[str, Any]]:
    rows = client.read(
        """
        MATCH (j:JobDescription {id: $jd_id})
        OPTIONAL MATCH (j)-[:REJECTED_FOR]->(r:RejectionReason)
        RETURN j.id AS id, j.text AS text, j.status AS status,
               j.role_family AS role_family, j.created_at AS created_at,
               j.hire_outcome AS hire_outcome,
               collect(DISTINCT {
                   id: r.id, text: r.text, categories: r.categories,
                   created_at: r.created_at
               }) AS rejections
        """,
        jd_id=jd_id,
    )
    if not rows:
        return None
    row = rows[0]
    row["rejections"] = [r for r in row["rejections"] if r.get("id")]
    return row


# ---------------------------------------------------------------------------
# Read-only Cypher escape hatch
# ---------------------------------------------------------------------------
_WRITE_RE = re.compile(
    r"(?i)(^|[\s;])(create|merge|delete|set|remove|drop|detach|load\s+csv|call\s+\{?\s*create)\b"
)


class CypherWriteRejected(RuntimeError):
    """Raised when the read-only `cypher_query` tool sees a write keyword."""


def cypher_query(client: Neo4jClient, query: str) -> Dict[str, Any]:
    """Run an arbitrary READ-ONLY Cypher query.

    Rejects any write/admin keyword via regex. The match is whitespace- and
    comment-tolerant. Statements with multiple semicolons are rejected as a
    safety net (one query at a time).
    """
    q = (query or "").strip()
    if not q:
        return {"ok": False, "reason": "empty query"}
    # Strip simple line comments to keep the regex tight
    stripped = "\n".join(line.split("//", 1)[0] for line in q.splitlines())
    if stripped.count(";") > 1:
        return {"ok": False, "reason": "multiple statements not allowed"}
    if _WRITE_RE.search(stripped):
        return {"ok": False, "reason": "write operations not allowed via cypher_query"}
    try:
        rows = client.read(stripped.rstrip(";"))
    except Exception as e:  # noqa: BLE001 — surface DB errors to the agent
        return {"ok": False, "reason": f"query failed: {type(e).__name__}: {e}"}
    return {"ok": True, "rows": rows, "row_count": len(rows)}
