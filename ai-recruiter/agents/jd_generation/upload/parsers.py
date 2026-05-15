"""Upload parsers + applier.

Two accepted formats:

1. JSON — a single file matching `UploadPayload`. Shape is identical to the
   shipped `synthetic_employees.json`.
2. CSV bundle (.zip) — one CSV per entity. File names are case-insensitive:
       employees.csv, roles.csv, skills.csv, teams.csv,
       holds.csv, has_skill.csv,
       (optional) past_jds.csv, education.csv, prior_companies.csv,
       worked_at.csv, graduated_from.csv

In CSV mode, edges live in their own files (holds.csv etc.) because flatness
plays nicer with Excel/Sheets exports. JSON mode keeps edges nested under
each employee.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import zipfile
from typing import Any, Dict, List, Tuple

from pydantic import ValidationError

from agents.jd_generation.config import JDConfig
from agents.jd_generation.graph.client import Neo4jClient
from agents.jd_generation.graph.seed import _seed_edges, _seed_nodes

from .schema import (
    EducationEntry,
    EmployeeEntry,
    EmployeeRoleHeld,
    EmployeeSkill,
    PastJDEntry,
    PriorCompanyEntry,
    RoleEntry,
    SkillEntry,
    TeamEntry,
    UploadPayload,
)

log = logging.getLogger("agents.jd_generation.upload")


class UploadError(ValueError):
    """Raised when an upload can't be parsed or fails validation."""


# ---------------------------------------------------------------------------
# Format detection + parsing
# ---------------------------------------------------------------------------
def parse_payload(file_bytes: bytes, filename: str) -> UploadPayload:
    ext = (filename or "").rsplit(".", 1)[-1].lower()
    if ext == "json":
        return _parse_json(file_bytes)
    if ext == "zip":
        return _parse_csv_bundle(file_bytes)
    if ext == "csv":
        # A single CSV is too ambiguous — surface the requirement loudly.
        raise UploadError(
            "single CSVs aren't supported — zip them into a bundle with "
            "employees.csv, roles.csv, skills.csv (and edge CSVs)"
        )
    raise UploadError(f"unsupported upload extension: .{ext}")


def _parse_json(file_bytes: bytes) -> UploadPayload:
    try:
        raw = json.loads(file_bytes.decode("utf-8"))
    except json.JSONDecodeError as e:
        raise UploadError(f"invalid JSON: {e}") from e
    if not isinstance(raw, dict):
        raise UploadError("top-level JSON must be an object")
    # Tolerate the shipped synthetic file's `_meta` key
    raw.pop("_meta", None)
    try:
        return UploadPayload.model_validate(raw)
    except ValidationError as e:
        raise UploadError(f"validation failed: {e}") from e


def _parse_csv_bundle(file_bytes: bytes) -> UploadPayload:
    try:
        zf = zipfile.ZipFile(io.BytesIO(file_bytes))
    except zipfile.BadZipFile as e:
        raise UploadError(f"invalid zip: {e}") from e

    members = {name.lower(): name for name in zf.namelist()}

    def _read_csv(key: str) -> List[Dict[str, str]]:
        for ext in ("csv", "tsv"):
            candidate = members.get(f"{key}.{ext}") or members.get(f"./{key}.{ext}")
            if candidate:
                with zf.open(candidate) as f:
                    text = io.TextIOWrapper(f, encoding="utf-8-sig", newline="")
                    delim = "," if ext == "csv" else "\t"
                    reader = csv.DictReader(text, delimiter=delim)
                    return [{k: (v or "").strip() for k, v in row.items()}
                            for row in reader]
        return []

    skills_rows    = _read_csv("skills")
    roles_rows     = _read_csv("roles")
    teams_rows     = _read_csv("teams")
    employees_rows = _read_csv("employees")
    holds_rows     = _read_csv("holds")
    has_skill_rows = _read_csv("has_skill")
    education_rows = _read_csv("education")
    prior_co_rows  = _read_csv("prior_companies")
    worked_at_rows = _read_csv("worked_at")
    graduated_rows = _read_csv("graduated_from")
    past_jd_rows   = _read_csv("past_jds")

    if not employees_rows:
        raise UploadError("zip must contain employees.csv (was missing or empty)")

    # ---- Top-level entities ------------------------------------------------
    skills = [SkillEntry(**r) for r in skills_rows] if skills_rows else []
    roles  = [RoleEntry(**r) for r in roles_rows] if roles_rows else []
    teams  = [TeamEntry(**(_coerce_size(r))) for r in teams_rows] if teams_rows else []

    # ---- Employees + nested edges -----------------------------------------
    holds_by_emp:    Dict[str, List[EmployeeRoleHeld]] = {}
    skills_by_emp:   Dict[str, List[EmployeeSkill]]    = {}
    edu_by_emp:      Dict[str, List[EducationEntry]]   = {}
    prior_by_emp:    Dict[str, List[PriorCompanyEntry]] = {}

    for r in holds_rows:
        emp_id = r.get("emp_id") or r.get("employee_id")
        if not emp_id:
            continue
        holds_by_emp.setdefault(emp_id, []).append(EmployeeRoleHeld(
            role_id=r["role_id"], start=r.get("start") or None,
            end=r.get("end") or None,
            current=(r.get("current", "true").lower() in {"true", "1", "yes"}),
        ))

    for r in has_skill_rows:
        emp_id = r.get("emp_id") or r.get("employee_id")
        if not emp_id:
            continue
        prof = int(r.get("proficiency") or 3)
        skills_by_emp.setdefault(emp_id, []).append(
            EmployeeSkill(skill_id=r["skill_id"], proficiency=prof)
        )

    edu_lookup = {r["id"]: EducationEntry(**r) for r in education_rows} if education_rows else {}
    for r in graduated_rows:
        emp_id = r.get("emp_id") or r.get("employee_id")
        edu_id = r.get("edu_id") or r.get("education_id")
        if emp_id and edu_id and edu_id in edu_lookup:
            edu_by_emp.setdefault(emp_id, []).append(edu_lookup[edu_id])

    co_lookup = {r["id"]: PriorCompanyEntry(**r) for r in prior_co_rows} if prior_co_rows else {}
    for r in worked_at_rows:
        emp_id = r.get("emp_id") or r.get("employee_id")
        co_id = r.get("co_id") or r.get("company_id")
        if emp_id and co_id and co_id in co_lookup:
            prior_by_emp.setdefault(emp_id, []).append(co_lookup[co_id])

    employees: List[EmployeeEntry] = []
    for r in employees_rows:
        emp_id = r["id"]
        tenure = r.get("tenure_years")
        employees.append(EmployeeEntry(
            id=emp_id,
            hire_date=r.get("hire_date") or None,
            tenure_years=float(tenure) if tenure else None,
            level=r.get("level", "mid"),  # type: ignore[arg-type]
            status=r.get("status", "active") or "active",
            team_id=r.get("team_id") or None,
            roles_held=holds_by_emp.get(emp_id, []),
            skills=skills_by_emp.get(emp_id, []),
            education=edu_by_emp.get(emp_id, []),
            prior_companies=prior_by_emp.get(emp_id, []),
        ))

    past_jds = [PastJDEntry(**(_coerce_status(r))) for r in past_jd_rows] if past_jd_rows else []

    try:
        return UploadPayload(
            skills=skills, roles=roles, teams=teams,
            employees=employees, past_jds=past_jds,
        )
    except ValidationError as e:
        raise UploadError(f"validation failed: {e}") from e


def _coerce_size(row: Dict[str, str]) -> Dict[str, Any]:
    out = dict(row)
    if out.get("size"):
        try:
            out["size"] = int(out["size"])
        except ValueError:
            out["size"] = None
    return out


def _coerce_status(row: Dict[str, str]) -> Dict[str, Any]:
    out = dict(row)
    if out.get("status") not in {"draft", "approved", "rejected"}:
        out["status"] = "approved"
    if out.get("hire_outcome") not in {"good_hire", "bad_hire", "not_filled", None}:
        out["hire_outcome"] = None
    return out


# ---------------------------------------------------------------------------
# Apply to Neo4j
# ---------------------------------------------------------------------------
def _payload_to_seed_dict(payload: UploadPayload) -> Dict[str, Any]:
    """Convert UploadPayload to the dict shape `seed.py._seed_nodes` consumes."""
    return {
        "skills":    [s.model_dump() for s in payload.skills],
        "roles":     [r.model_dump() for r in payload.roles],
        "teams":     [t.model_dump() for t in payload.teams],
        "employees": [e.model_dump() for e in payload.employees],
        "past_jds":  [j.model_dump() for j in payload.past_jds],
    }


def _replace_preserving_rejections(client: Neo4jClient) -> None:
    """Wipe everything EXCEPT RejectionReason nodes and their REJECTED_FOR edges.

    Rejections are learning artifacts — losing them defeats the whole point
    of the rejection-feedback loop, so even a hard replace keeps them.
    """
    # Step 1: collect the JD ids that are linked to RejectionReason — we
    # also preserve those JD nodes (without their text? — keep the text too,
    # so historic context survives).
    rows = client.read(
        "MATCH (j:JobDescription)-[:REJECTED_FOR]->(:RejectionReason) "
        "RETURN DISTINCT j.id AS id"
    )
    preserved_jd_ids = [r["id"] for r in rows]

    # Step 2: delete everything except RejectionReason + linked JDs
    client.write(
        """
        MATCH (n)
        WHERE NOT (n:RejectionReason)
          AND NOT (n:JobDescription AND n.id IN $preserved)
        DETACH DELETE n
        """,
        preserved=preserved_jd_ids,
    )


def handle_upload(
    client: Neo4jClient,
    file_bytes: bytes,
    filename: str,
    mode: str,
    cfg: JDConfig,
) -> Dict[str, Any]:
    """Parse the upload, validate, write to Neo4j, return a summary."""
    if cfg.upload.prod_guard and mode == "replace":
        raise UploadError("JD_PROD_GUARD is set — replace mode refused")

    payload = parse_payload(file_bytes, filename)

    if mode == "replace":
        _replace_preserving_rejections(client)

    seed_dict = _payload_to_seed_dict(payload)
    node_counts = _seed_nodes(client, seed_dict)
    edge_counts = _seed_edges(client, seed_dict)

    return {
        "ok":        True,
        "mode":      mode,
        "counts":    payload.counts(),
        "nodes":     node_counts,
        "edges":     edge_counts,
        "warnings":  payload.referential_warnings(),
    }
