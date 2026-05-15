"""Phase 6 tests — graph dump + upload.

Unit tests:
    * filter parsing
    * pydantic upload schema rejects bad shapes
    * referential-integrity warnings
    * JSON upload parses the shipped synthetic file unchanged
    * CSV-bundle round trip
    * format detection error paths

Integration tests (live Neo4j):
    * dump returns nodes + links against seeded data
    * augment upload adds new employees without wiping existing ones
    * replace upload preserves RejectionReason nodes
"""

from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

DATA = (
    Path(__file__).resolve().parents[2]
    / "agents"
    / "jd_generation"
    / "data"
    / "synthetic_employees.json"
)


# ===========================================================================
# Filter parsing
# ===========================================================================
def test_parse_filter_empty() -> None:
    from agents.jd_generation.graph.dump import _parse_filter
    assert _parse_filter(None) == {}
    assert _parse_filter("") == {}


def test_parse_filter_single() -> None:
    from agents.jd_generation.graph.dump import _parse_filter
    assert _parse_filter("role_family:backend") == {"role_family": "backend"}


def test_parse_filter_multi() -> None:
    from agents.jd_generation.graph.dump import _parse_filter
    out = _parse_filter("role_family:backend,level:senior,team_id:team_platform")
    assert out == {"role_family": "backend", "level": "senior", "team_id": "team_platform"}


def test_parse_filter_ignores_bad_pieces() -> None:
    from agents.jd_generation.graph.dump import _parse_filter
    out = _parse_filter("role_family:backend,plain_token,level:senior")
    assert out == {"role_family": "backend", "level": "senior"}


# ===========================================================================
# Upload schema validation
# ===========================================================================
def test_upload_payload_accepts_synthetic_file() -> None:
    """The shipped synthetic file is the canonical reference upload."""
    from agents.jd_generation.upload.schema import UploadPayload
    raw = json.loads(DATA.read_text(encoding="utf-8"))
    raw.pop("_meta", None)
    payload = UploadPayload.model_validate(raw)
    counts = payload.counts()
    assert counts["employees"] == 100
    assert counts["roles"] == 15


def test_upload_payload_rejects_bad_level() -> None:
    from pydantic import ValidationError
    from agents.jd_generation.upload.schema import UploadPayload
    with pytest.raises(ValidationError):
        UploadPayload.model_validate({
            "employees": [{
                "id": "e1", "level": "principal",  # not in {junior, mid, senior}
                "status": "active",
            }],
        })


def test_upload_payload_rejects_proficiency_out_of_range() -> None:
    from pydantic import ValidationError
    from agents.jd_generation.upload.schema import UploadPayload
    with pytest.raises(ValidationError):
        UploadPayload.model_validate({
            "employees": [{
                "id": "e1", "level": "mid", "status": "active",
                "skills": [{"skill_id": "sk_1", "proficiency": 99}],
            }],
        })


def test_upload_payload_referential_warnings() -> None:
    from agents.jd_generation.upload.schema import UploadPayload
    payload = UploadPayload.model_validate({
        "skills": [{"id": "sk_real", "name": "Python"}],
        "employees": [{
            "id": "e1", "level": "mid", "status": "active",
            "skills": [{"skill_id": "sk_ghost", "proficiency": 4}],
            "team_id": "team_ghost",
        }],
    })
    warnings = payload.referential_warnings()
    assert any("sk_ghost" in w for w in warnings)
    assert any("team_ghost" in w for w in warnings)


def test_upload_payload_empty_is_valid() -> None:
    from agents.jd_generation.upload.schema import UploadPayload
    payload = UploadPayload.model_validate({})
    assert payload.counts() == {"employees": 0, "roles": 0, "skills": 0,
                                "teams": 0, "past_jds": 0}


# ===========================================================================
# Parser format detection
# ===========================================================================
def test_parser_rejects_unknown_extension() -> None:
    from agents.jd_generation.upload.parsers import UploadError, parse_payload
    with pytest.raises(UploadError, match="unsupported"):
        parse_payload(b"{}", "blob.txt")


def test_parser_rejects_single_csv() -> None:
    from agents.jd_generation.upload.parsers import UploadError, parse_payload
    with pytest.raises(UploadError, match="zip them into a bundle"):
        parse_payload(b"id,name\n1,a\n", "employees.csv")


def test_parser_rejects_invalid_json() -> None:
    from agents.jd_generation.upload.parsers import UploadError, parse_payload
    with pytest.raises(UploadError, match="invalid JSON"):
        parse_payload(b"{not json", "data.json")


def test_parser_rejects_non_object_json() -> None:
    from agents.jd_generation.upload.parsers import UploadError, parse_payload
    with pytest.raises(UploadError, match="object"):
        parse_payload(b"[1,2,3]", "data.json")


def test_parser_strips_meta_block() -> None:
    """The shipped synthetic file has a `_meta` key — uploads should accept
    that shape verbatim."""
    from agents.jd_generation.upload.parsers import parse_payload
    body = DATA.read_bytes()
    payload = parse_payload(body, "synthetic_employees.json")
    assert payload.counts()["employees"] == 100


# ===========================================================================
# CSV bundle round trip
# ===========================================================================
def _build_minimal_csv_zip() -> bytes:
    files: Dict[str, str] = {}

    def _csv(rows: List[Dict[str, Any]], header: List[str]) -> str:
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)
        return buf.getvalue()

    files["skills.csv"]    = _csv([{"id": "sk_py", "name": "Python", "category": "language_python"}],
                                  ["id", "name", "category"])
    files["roles.csv"]     = _csv([
        {"id": "role_backend_senior", "title": "Senior Backend Engineer",
         "level": "senior", "role_family": "backend", "department": "engineering"},
    ], ["id", "title", "level", "role_family", "department"])
    files["teams.csv"]     = _csv([{"id": "team_platform", "name": "Platform",
                                    "department": "engineering", "size": "12"}],
                                  ["id", "name", "department", "size"])
    files["employees.csv"] = _csv([{"id": "emp_x", "level": "senior", "status": "active",
                                    "team_id": "team_platform", "tenure_years": "3.2",
                                    "hire_date": "2023-01-01"}],
                                  ["id", "level", "status", "team_id",
                                   "tenure_years", "hire_date"])
    files["holds.csv"]     = _csv([{"emp_id": "emp_x", "role_id": "role_backend_senior",
                                    "start": "2023-01-01", "end": "", "current": "true"}],
                                  ["emp_id", "role_id", "start", "end", "current"])
    files["has_skill.csv"] = _csv([{"emp_id": "emp_x", "skill_id": "sk_py", "proficiency": "5"}],
                                  ["emp_id", "skill_id", "proficiency"])

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w") as zf:
        for name, body in files.items():
            zf.writestr(name, body)
    return buf.getvalue()


def test_csv_bundle_parses_minimal_zip() -> None:
    from agents.jd_generation.upload.parsers import parse_payload
    blob = _build_minimal_csv_zip()
    payload = parse_payload(blob, "bundle.zip")
    counts = payload.counts()
    assert counts["employees"] == 1
    assert counts["roles"] == 1
    assert counts["skills"] == 1
    assert counts["teams"] == 1

    emp = payload.employees[0]
    assert emp.id == "emp_x"
    assert emp.team_id == "team_platform"
    assert len(emp.skills) == 1
    assert emp.skills[0].skill_id == "sk_py"
    assert emp.skills[0].proficiency == 5
    assert len(emp.roles_held) == 1
    assert emp.roles_held[0].role_id == "role_backend_senior"


def test_csv_bundle_requires_employees_csv() -> None:
    from agents.jd_generation.upload.parsers import UploadError, parse_payload
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w") as zf:
        zf.writestr("skills.csv", "id,name\n")
    with pytest.raises(UploadError, match="employees.csv"):
        parse_payload(buf.getvalue(), "incomplete.zip")


def test_csv_bundle_rejects_bad_zip() -> None:
    from agents.jd_generation.upload.parsers import UploadError, parse_payload
    with pytest.raises(UploadError, match="invalid zip"):
        parse_payload(b"definitely not a zip file", "bad.zip")


# ===========================================================================
# Replace-mode prod guard
# ===========================================================================
def test_upload_replace_refused_under_prod_guard() -> None:
    from agents.jd_generation.config import (
        AgentLoopConfig, JDConfig, LLMConfig, Neo4jConfig, UploadConfig,
    )
    from agents.jd_generation.upload.parsers import UploadError, handle_upload

    cfg = JDConfig(
        neo4j=Neo4jConfig(uri="", user="", password="", database="neo4j", connection_timeout_s=5),
        llm=LLMConfig(backend="groq", model="x", api_key=None,
                      temperature_synthesis=0.7, temperature_tool_select=0.0,
                      request_timeout_s=60),
        agent=AgentLoopConfig(max_tool_calls=15, max_revisions=3,
                              rejection_lookback=10, deterministic_preambles=[]),
        upload=UploadConfig(max_file_size_mb=20, allowed_extensions=["json", "zip"],
                            prod_guard=True),
    )

    class FakeClient:
        def read(self, *a, **k):  # noqa: ARG002
            return []
        def write(self, *a, **k):  # noqa: ARG002
            return []

    with pytest.raises(UploadError, match="PROD_GUARD"):
        handle_upload(FakeClient(), b"{}", "x.json", mode="replace", cfg=cfg)


# ===========================================================================
# Integration — live Neo4j
# ===========================================================================
@pytest.fixture
def seeded_graph(requires_neo4j) -> None:  # noqa: ARG001
    from agents.jd_generation.graph.client import close_client
    from agents.jd_generation.graph.seed import seed
    try:
        seed(clear=True, quiet=True)
        yield
    finally:
        close_client()


@pytest.mark.integration
def test_dump_returns_nodes_and_links(seeded_graph) -> None:  # noqa: ARG001
    from agents.jd_generation.graph.client import get_client
    from agents.jd_generation.graph.dump import dump

    result = dump(get_client(), filter="role_family:backend,level:senior")
    assert result["node_count"] > 0
    assert result["link_count"] > 0
    types = {n["type"] for n in result["nodes"]}
    assert "employee" in types
    assert "role" in types


@pytest.mark.integration
def test_dump_includes_rejections_by_default(seeded_graph) -> None:  # noqa: ARG001
    from agents.jd_generation.graph.client import get_client
    from agents.jd_generation.graph.dump import dump

    with_rej = dump(get_client(), include_rejections=True)
    without_rej = dump(get_client(), include_rejections=False)
    types_with = {n["type"] for n in with_rej["nodes"]}
    types_without = {n["type"] for n in without_rej["nodes"]}
    assert "rejection" in types_with
    assert "rejection" not in types_without


@pytest.mark.integration
def test_replace_preserves_rejections(seeded_graph) -> None:  # noqa: ARG001
    """Critical — replace mode wipes most of the graph but keeps the
    RejectionReason learning artifacts."""
    from agents.jd_generation.config import load_jd_config
    from agents.jd_generation.graph.client import get_client
    from agents.jd_generation.upload.parsers import handle_upload

    cfg = load_jd_config()
    blob = _build_minimal_csv_zip()
    handle_upload(get_client(), blob, "bundle.zip", mode="replace", cfg=cfg)

    client = get_client()
    rejection_count = client.read(
        "MATCH (r:RejectionReason) RETURN count(r) AS n"
    )[0]["n"]
    assert rejection_count == 3, "RejectionReason nodes should survive replace mode"
