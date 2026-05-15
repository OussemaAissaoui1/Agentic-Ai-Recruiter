"""Synthetic data seed loader.

Reads `data/synthetic_employees.json` and loads it into Neo4j. Idempotent
via `MERGE`. Schema is applied first.

Safety:
    - Refuses to run when `JD_PROD_GUARD=1`. Set this in any production-like
      environment.
    - `--clear` wipes the entire graph before seeding (dev only).

CLI:
    python -m agents.jd_generation.graph.seed [--clear] [--quiet]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.jd_generation.config import load_jd_config
from agents.jd_generation.graph.client import Neo4jClient, Neo4jNotConfigured, get_client
from agents.jd_generation.graph.schema import apply_schema

log = logging.getLogger("agents.jd_generation.seed")

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "synthetic_employees.json"


def _load_payload(path: Path = DATA_PATH) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _clear(client: Neo4jClient) -> None:
    """Wipe the graph. Dev only — guarded by `JD_PROD_GUARD`."""
    client.write("MATCH (n) DETACH DELETE n")


def _seed_nodes(client: Neo4jClient, payload: Dict[str, Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}

    # Skills
    skills = payload.get("skills", [])
    if skills:
        client.write(
            """
            UNWIND $rows AS row
            MERGE (s:Skill {id: row.id})
            SET s.name = row.name,
                s.category = row.category,
                s.esco_id = row.esco_id
            """,
            rows=skills,
        )
    counts["Skill"] = len(skills)

    # Roles
    roles = payload.get("roles", [])
    if roles:
        client.write(
            """
            UNWIND $rows AS row
            MERGE (r:Role {id: row.id})
            SET r.title = row.title,
                r.level = row.level,
                r.role_family = row.role_family,
                r.department = row.department
            """,
            rows=roles,
        )
    counts["Role"] = len(roles)

    # Teams
    teams = payload.get("teams", [])
    if teams:
        client.write(
            """
            UNWIND $rows AS row
            MERGE (t:Team {id: row.id})
            SET t.name = row.name,
                t.department = row.department,
                t.size = row.size
            """,
            rows=teams,
        )
    counts["Team"] = len(teams)

    # Employees (the simple core fields — edges + nested entities below)
    employees = payload.get("employees", [])
    if employees:
        client.write(
            """
            UNWIND $rows AS row
            MERGE (e:Employee {id: row.id})
            SET e.hire_date = row.hire_date,
                e.tenure_years = row.tenure_years,
                e.level = row.level,
                e.status = row.status
            """,
            rows=employees,
        )
    counts["Employee"] = len(employees)

    # Education and PriorCompany are nested inside employees — collect uniques
    edu_rows: List[Dict[str, Any]] = []
    seen_edu: set = set()
    co_rows: List[Dict[str, Any]] = []
    seen_co: set = set()
    for emp in employees:
        for edu in emp.get("education", []):
            if edu.get("id") and edu["id"] not in seen_edu:
                edu_rows.append(edu)
                seen_edu.add(edu["id"])
        for co in emp.get("prior_companies", []):
            if co.get("id") and co["id"] not in seen_co:
                co_rows.append(co)
                seen_co.add(co["id"])

    if edu_rows:
        client.write(
            """
            UNWIND $rows AS row
            MERGE (e:Education {id: row.id})
            SET e.degree = row.degree,
                e.field = row.field,
                e.institution_tier = row.institution_tier
            """,
            rows=edu_rows,
        )
    counts["Education"] = len(edu_rows)

    if co_rows:
        client.write(
            """
            UNWIND $rows AS row
            MERGE (c:PriorCompany {id: row.id})
            SET c.name = row.name,
                c.industry = row.industry,
                c.size_bucket = row.size_bucket
            """,
            rows=co_rows,
        )
    counts["PriorCompany"] = len(co_rows)

    # Past JDs
    past_jds = payload.get("past_jds", [])
    if past_jds:
        client.write(
            """
            UNWIND $rows AS row
            MERGE (j:JobDescription {id: row.id})
            SET j.text = row.text,
                j.status = row.status,
                j.role_family = row.role_family,
                j.hire_outcome = row.hire_outcome,
                j.created_at = row.created_at
            """,
            rows=past_jds,
        )
    counts["JobDescription"] = len(past_jds)

    # RejectionReason nodes (one per rejected JD)
    rejection_rows = [
        {
            "id":          jd["rejection"]["id"],
            "jd_id":       jd["id"],
            "text":        jd["rejection"]["text"],
            "categories":  jd["rejection"]["categories"],
            "created_at":  jd.get("created_at"),
        }
        for jd in past_jds
        if "rejection" in jd
    ]
    if rejection_rows:
        client.write(
            """
            UNWIND $rows AS row
            MERGE (r:RejectionReason {id: row.id})
            SET r.text = row.text,
                r.categories = row.categories,
                r.created_at = row.created_at
            """,
            rows=rejection_rows,
        )
        client.write(
            """
            UNWIND $rows AS row
            MATCH (j:JobDescription {id: row.jd_id})
            MATCH (r:RejectionReason {id: row.id})
            MERGE (j)-[:REJECTED_FOR]->(r)
            """,
            rows=rejection_rows,
        )
    counts["RejectionReason"] = len(rejection_rows)

    return counts


def _seed_edges(client: Neo4jClient, payload: Dict[str, Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    employees = payload.get("employees", [])

    # HOLDS / PREVIOUSLY_HELD
    holds_rows: List[Dict[str, Any]] = []
    prev_rows: List[Dict[str, Any]] = []
    for emp in employees:
        for r in emp.get("roles_held", []):
            row = {
                "emp_id":  emp["id"],
                "role_id": r["role_id"],
                "start":   r.get("start"),
                "end":     r.get("end"),
            }
            if r.get("current"):
                holds_rows.append(row)
            else:
                prev_rows.append(row)

    if holds_rows:
        client.write(
            """
            UNWIND $rows AS row
            MATCH (e:Employee {id: row.emp_id})
            MATCH (r:Role     {id: row.role_id})
            MERGE (e)-[h:HOLDS]->(r)
            SET h.start_date = row.start, h.end_date = row.end
            """,
            rows=holds_rows,
        )
    counts["HOLDS"] = len(holds_rows)

    if prev_rows:
        client.write(
            """
            UNWIND $rows AS row
            MATCH (e:Employee {id: row.emp_id})
            MATCH (r:Role     {id: row.role_id})
            MERGE (e)-[h:PREVIOUSLY_HELD {start_date: row.start, end_date: row.end}]->(r)
            """,
            rows=prev_rows,
        )
    counts["PREVIOUSLY_HELD"] = len(prev_rows)

    # HAS_SKILL
    has_skill_rows = [
        {"emp_id": emp["id"], "skill_id": s["skill_id"], "proficiency": s["proficiency"]}
        for emp in employees
        for s in emp.get("skills", [])
    ]
    if has_skill_rows:
        client.write(
            """
            UNWIND $rows AS row
            MATCH (e:Employee {id: row.emp_id})
            MATCH (s:Skill    {id: row.skill_id})
            MERGE (e)-[r:HAS_SKILL]->(s)
            SET r.proficiency = row.proficiency
            """,
            rows=has_skill_rows,
        )
    counts["HAS_SKILL"] = len(has_skill_rows)

    # BELONGS_TO
    belongs_rows = [
        {"emp_id": emp["id"], "team_id": emp["team_id"]}
        for emp in employees
        if emp.get("team_id")
    ]
    if belongs_rows:
        client.write(
            """
            UNWIND $rows AS row
            MATCH (e:Employee {id: row.emp_id})
            MATCH (t:Team     {id: row.team_id})
            MERGE (e)-[:BELONGS_TO]->(t)
            """,
            rows=belongs_rows,
        )
    counts["BELONGS_TO"] = len(belongs_rows)

    # GRADUATED_FROM
    grad_rows = [
        {"emp_id": emp["id"], "edu_id": edu["id"]}
        for emp in employees
        for edu in emp.get("education", [])
        if edu.get("id")
    ]
    if grad_rows:
        client.write(
            """
            UNWIND $rows AS row
            MATCH (e:Employee {id: row.emp_id})
            MATCH (d:Education {id: row.edu_id})
            MERGE (e)-[:GRADUATED_FROM]->(d)
            """,
            rows=grad_rows,
        )
    counts["GRADUATED_FROM"] = len(grad_rows)

    # WORKED_AT
    worked_rows = [
        {"emp_id": emp["id"], "co_id": co["id"]}
        for emp in employees
        for co in emp.get("prior_companies", [])
        if co.get("id")
    ]
    if worked_rows:
        client.write(
            """
            UNWIND $rows AS row
            MATCH (e:Employee     {id: row.emp_id})
            MATCH (c:PriorCompany {id: row.co_id})
            MERGE (e)-[:WORKED_AT]->(c)
            """,
            rows=worked_rows,
        )
    counts["WORKED_AT"] = len(worked_rows)

    # JobDescription -> role family — GENERATED_FOR_ROLE_FAMILY uses a Role
    # node as the family marker. We pick the senior-level role of the family
    # as the anchor; downstream queries traverse by `Role.role_family`.
    past_jds = payload.get("past_jds", [])
    gen_rows = [
        {"jd_id": jd["id"], "role_id": f"role_{jd['role_family']}_senior"}
        for jd in past_jds
    ]
    if gen_rows:
        client.write(
            """
            UNWIND $rows AS row
            MATCH (j:JobDescription {id: row.jd_id})
            MATCH (r:Role           {id: row.role_id})
            MERGE (j)-[:GENERATED_FOR_ROLE_FAMILY]->(r)
            """,
            rows=gen_rows,
        )
    counts["GENERATED_FOR_ROLE_FAMILY"] = len(gen_rows)

    # Derived edges — Role REQUIRES Skill, Role PROGRESSES_TO Role,
    # Skill ADJACENT_TO Skill — built from observed employee data.
    _build_derived_edges(client, payload)

    return counts


def _build_derived_edges(client: Neo4jClient, payload: Dict[str, Any]) -> None:
    """Compute Role REQUIRES Skill (weighted) and Role PROGRESSES_TO Role.

    Run after HOLDS and HAS_SKILL are populated. Keeps the graph
    self-consistent with the underlying employee data.
    """
    # REQUIRES: weight = avg proficiency among current holders of the role
    client.write(
        """
        MATCH (e:Employee)-[h:HOLDS]->(r:Role)
        MATCH (e)-[s:HAS_SKILL]->(sk:Skill)
        WITH r, sk, avg(s.proficiency) AS weight, count(*) AS holders
        WHERE holders >= 2
        MERGE (r)-[req:REQUIRES]->(sk)
        SET req.weight = weight
        """
    )

    # PROGRESSES_TO: count of employees who held role_a then role_b
    client.write(
        """
        MATCH (e:Employee)-[:PREVIOUSLY_HELD]->(prev:Role)
        MATCH (e)-[:HOLDS]->(cur:Role)
        WITH prev, cur, count(*) AS freq
        MERGE (prev)-[p:PROGRESSES_TO]->(cur)
        SET p.frequency = freq
        """
    )

    # ADJACENT_TO: skills that co-occur in employees of the same role family.
    # Strength = number of co-occurring employees, normalized by min of the
    # two skills' total employee counts (Jaccard-style).
    client.write(
        """
        MATCH (e:Employee)-[:HAS_SKILL]->(s1:Skill)
        MATCH (e)-[:HAS_SKILL]->(s2:Skill)
        WHERE s1.id < s2.id
        WITH s1, s2, count(DISTINCT e) AS co
        WHERE co >= 5
        MATCH (e1:Employee)-[:HAS_SKILL]->(s1)
        WITH s1, s2, co, count(DISTINCT e1) AS s1_total
        MATCH (e2:Employee)-[:HAS_SKILL]->(s2)
        WITH s1, s2, co, s1_total, count(DISTINCT e2) AS s2_total
        WITH s1, s2, toFloat(co) / toFloat(CASE WHEN s1_total < s2_total
                                                 THEN s1_total
                                                 ELSE s2_total END) AS strength
        WHERE strength >= 0.30
        MERGE (s1)-[adj:ADJACENT_TO]->(s2)
        SET adj.strength = strength
        """
    )


def _prod_guard_check() -> Optional[str]:
    if os.environ.get("JD_PROD_GUARD", "0").strip() not in {"", "0", "false", "False"}:
        return "JD_PROD_GUARD is set — refusing to seed (production guard)."
    return None


def seed(clear: bool = False, quiet: bool = False) -> Dict[str, Any]:
    """Programmatic entrypoint. Returns a summary dict."""
    err = _prod_guard_check()
    if err:
        raise RuntimeError(err)

    cfg = load_jd_config()
    if not cfg.neo4j.is_configured:
        raise Neo4jNotConfigured(
            "JD_NEO4J_URI / JD_NEO4J_USER / JD_NEO4J_PASSWORD must be set"
        )

    client = get_client(cfg)
    schema_summary = apply_schema(client)
    if clear:
        if not quiet:
            log.info("clearing graph...")
        _clear(client)

    payload = _load_payload()
    node_counts = _seed_nodes(client, payload)
    edge_counts = _seed_edges(client, payload)

    summary = {
        "started_at":  datetime.now(timezone.utc).isoformat(),
        "schema":      schema_summary,
        "nodes":       node_counts,
        "edges":       edge_counts,
        "meta":        payload.get("_meta", {}),
    }
    if not quiet:
        log.info("seed done: %s", summary)
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="agents.jd_generation.graph.seed")
    parser.add_argument("--clear", action="store_true",
                        help="wipe the graph before seeding (dev only)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    try:
        summary = seed(clear=args.clear, quiet=args.quiet)
    except Neo4jNotConfigured as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 3

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
