"""Graph dump for the 3D explorer.

Returns a D3/force-graph-3d-shaped `{nodes, links}` JSON payload. The
frontend uses `id` as the node id, `type` as the node label (employee /
role / skill / ...), and a few standard fields for tooltips. Anything
extra goes under `props` so the frontend can opt into showing it.

Filtering:
    filter syntax: "role_family:backend" or "level:senior" or "team_id:team_platform"
    multiple filters: "role_family:backend,level:senior"
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from .client import Neo4jClient

log = logging.getLogger("agents.jd_generation.graph.dump")

# Node-type colors are picked by the frontend; we just label types here so the
# frontend has stable strings to switch on.
NODE_TYPE_BY_LABEL = {
    "Employee":         "employee",
    "Role":             "role",
    "Skill":            "skill",
    "Team":             "team",
    "Education":        "education",
    "PriorCompany":     "prior_company",
    "JobDescription":   "job_description",
    "RejectionReason":  "rejection",
}


def _parse_filter(s: Optional[str]) -> Dict[str, str]:
    """`'role_family:backend,level:senior'` -> `{'role_family':'backend','level':'senior'}`"""
    if not s:
        return {}
    out: Dict[str, str] = {}
    for piece in s.split(","):
        if ":" in piece:
            k, v = piece.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def _cohort_ids(client: Neo4jClient, filters: Dict[str, str]) -> Tuple[List[str], List[str]]:
    """Return (employee_ids, role_ids) honouring the filter."""
    role_family = filters.get("role_family")
    level = filters.get("level")
    team_id = filters.get("team_id")

    where_clauses: List[str] = []
    if role_family:
        where_clauses.append("r.role_family = $role_family")
    if level:
        where_clauses.append("r.level = $level")
    if team_id:
        where_clauses.append("t.id = $team_id")

    if not where_clauses:
        # No filter: take the whole graph
        emp_rows = client.read("MATCH (e:Employee) RETURN e.id AS id")
        role_rows = client.read("MATCH (r:Role) RETURN r.id AS id")
        return [r["id"] for r in emp_rows], [r["id"] for r in role_rows]

    optional_team = "OPTIONAL MATCH (e)-[:BELONGS_TO]->(t:Team)" if team_id else ""
    where = " AND ".join(where_clauses)

    rows = client.read(
        f"""
        MATCH (e:Employee)-[:HOLDS]->(r:Role)
        {optional_team}
        WHERE {where}
        RETURN DISTINCT e.id AS emp_id, r.id AS role_id
        """,
        role_family=role_family, level=level, team_id=team_id,
    )
    emp_ids = sorted({row["emp_id"] for row in rows})
    role_ids = sorted({row["role_id"] for row in rows})
    return emp_ids, role_ids


def dump(
    client: Neo4jClient,
    filter: Optional[str] = None,
    include_rejections: bool = True,
) -> Dict[str, Any]:
    """Build the {nodes, links} payload."""
    filters = _parse_filter(filter)
    emp_ids, role_ids = _cohort_ids(client, filters)

    nodes: Dict[str, Dict[str, Any]] = {}
    links: List[Dict[str, Any]] = []

    def add(node_id: str, label: str, name: str, props: Dict[str, Any]) -> None:
        if node_id and node_id not in nodes:
            nodes[node_id] = {
                "id":    node_id,
                "type":  NODE_TYPE_BY_LABEL.get(label, label.lower()),
                "label": name,
                "props": props,
            }

    # ---- Employees ----------------------------------------------------------
    if emp_ids:
        rows = client.read(
            """
            MATCH (e:Employee) WHERE e.id IN $ids
            RETURN e.id AS id, e.level AS level, e.tenure_years AS tenure,
                   e.hire_date AS hire_date, e.status AS status
            """,
            ids=emp_ids,
        )
        for r in rows:
            add(r["id"], "Employee", f"{r['id']} ({r['level']})",
                {"level": r["level"], "tenure_years": r["tenure"],
                 "hire_date": r["hire_date"], "status": r["status"]})

    # ---- Roles --------------------------------------------------------------
    if role_ids:
        rows = client.read(
            """
            MATCH (r:Role) WHERE r.id IN $ids
            RETURN r.id AS id, r.title AS title, r.level AS level,
                   r.role_family AS role_family, r.department AS department
            """,
            ids=role_ids,
        )
        for r in rows:
            add(r["id"], "Role", r["title"],
                {"level": r["level"], "role_family": r["role_family"],
                 "department": r["department"]})

    # ---- Skills + HAS_SKILL edges ------------------------------------------
    if emp_ids:
        rows = client.read(
            """
            MATCH (e:Employee)-[h:HAS_SKILL]->(s:Skill)
            WHERE e.id IN $ids
            RETURN e.id AS emp_id, s.id AS skill_id, s.name AS name,
                   s.category AS category, h.proficiency AS proficiency
            """,
            ids=emp_ids,
        )
        for r in rows:
            add(r["skill_id"], "Skill", r["name"], {"category": r["category"]})
            links.append({
                "source":      r["emp_id"],
                "target":      r["skill_id"],
                "type":        "HAS_SKILL",
                "proficiency": r["proficiency"],
            })

    # ---- HOLDS edges --------------------------------------------------------
    if emp_ids:
        rows = client.read(
            """
            MATCH (e:Employee)-[h:HOLDS]->(r:Role)
            WHERE e.id IN $ids
            RETURN e.id AS emp_id, r.id AS role_id, h.start_date AS start, h.end_date AS end
            """,
            ids=emp_ids,
        )
        for r in rows:
            links.append({"source": r["emp_id"], "target": r["role_id"], "type": "HOLDS",
                          "start": r["start"], "end": r["end"]})

    # ---- Teams + BELONGS_TO -----------------------------------------------
    if emp_ids:
        rows = client.read(
            """
            MATCH (e:Employee)-[:BELONGS_TO]->(t:Team)
            WHERE e.id IN $ids
            RETURN e.id AS emp_id, t.id AS team_id, t.name AS name,
                   t.department AS department, t.size AS size
            """,
            ids=emp_ids,
        )
        for r in rows:
            add(r["team_id"], "Team", r["name"],
                {"department": r["department"], "size": r["size"]})
            links.append({"source": r["emp_id"], "target": r["team_id"], "type": "BELONGS_TO"})

    # ---- PROGRESSES_TO (role -> role) — independent of filter for context ---
    rows = client.read(
        """
        MATCH (a:Role)-[p:PROGRESSES_TO]->(b:Role)
        WHERE a.id IN $role_ids OR b.id IN $role_ids
        RETURN a.id AS src, b.id AS dst, p.frequency AS frequency
        """,
        role_ids=role_ids,
    )
    for r in rows:
        if r["src"] in nodes or r["dst"] in nodes:
            links.append({"source": r["src"], "target": r["dst"],
                          "type": "PROGRESSES_TO", "frequency": r["frequency"]})

    # ---- REQUIRES (role -> skill) -----------------------------------------
    if role_ids:
        rows = client.read(
            """
            MATCH (r:Role)-[req:REQUIRES]->(s:Skill)
            WHERE r.id IN $ids
            RETURN r.id AS role_id, s.id AS skill_id, req.weight AS weight
            """,
            ids=role_ids,
        )
        for r in rows:
            if r["skill_id"] in nodes:
                links.append({"source": r["role_id"], "target": r["skill_id"],
                              "type": "REQUIRES", "weight": r["weight"]})

    # ---- Rejections ----------------------------------------------------------
    if include_rejections:
        # Match all JD nodes for the filtered role families. If no filter,
        # match all JDs.
        role_family = filters.get("role_family")
        rows = client.read(
            """
            MATCH (j:JobDescription)
            WHERE $rf IS NULL OR j.role_family = $rf
            OPTIONAL MATCH (j)-[:REJECTED_FOR]->(r:RejectionReason)
            RETURN j.id AS jd_id, j.role_family AS rf, j.status AS status,
                   r.id AS rej_id, r.text AS rej_text, r.categories AS rej_cats
            """,
            rf=role_family,
        )
        for r in rows:
            jd_id = r["jd_id"]
            add(jd_id, "JobDescription",
                f"JD ({r['rf']}, {r['status']})",
                {"role_family": r["rf"], "status": r["status"]})
            # Anchor the JD to the senior role of its family so the layout
            # has a stable position.
            anchor = f"role_{r['rf']}_senior"
            if anchor in nodes:
                links.append({"source": jd_id, "target": anchor,
                              "type": "GENERATED_FOR_ROLE_FAMILY"})
            if r["rej_id"]:
                add(r["rej_id"], "RejectionReason",
                    (r["rej_text"] or "")[:60],
                    {"categories": r["rej_cats"] or [],
                     "full_text": r["rej_text"]})
                links.append({"source": jd_id, "target": r["rej_id"], "type": "REJECTED_FOR"})

    return {
        "nodes":      list(nodes.values()),
        "links":      links,
        "node_count": len(nodes),
        "link_count": len(links),
        "filter":     filters,
    }
