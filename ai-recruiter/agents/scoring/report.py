"""Report aggregation, Markdown rendering, on-disk cache, DB write-back."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .schema import InterviewReport, OverallAssessment, TurnScore


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def compute_averages(turns: list[TurnScore]) -> tuple[float, float]:
    """Means of non-None per-turn scores. (0.0, 0.0) if all turns failed."""
    tech = [t.technical_score for t in turns if t.technical_score is not None]
    coh = [t.coherence_score for t in turns if t.coherence_score is not None]
    t_avg = round(sum(tech) / len(tech), 2) if tech else 0.0
    c_avg = round(sum(coh) / len(coh), 2) if coh else 0.0
    return t_avg, c_avg


@dataclass(frozen=True)
class Fallback:
    recommendation: str
    strengths: list[str]
    concerns: list[str]


def build_overall_fallback(tech_avg: float, coh_avg: float) -> Fallback:
    """Deterministic recommendation when the overall LLM pass fails."""
    avg = (tech_avg + coh_avg) / 2.0
    if avg >= 4.25:
        rec = "strong_hire"
    elif avg >= 3.5:
        rec = "hire"
    elif avg >= 2.0:
        rec = "lean_hire"
    else:
        rec = "no_hire"
    return Fallback(
        recommendation=rec,
        strengths=[f"average technical score {tech_avg}/5",
                   f"average coherence score {coh_avg}/5"],
        concerns=["overall LLM pass failed; recommendation derived from averages"],
    )


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------
def render_markdown(report: InterviewReport) -> str:
    o = report.overall
    iso = datetime.fromtimestamp(report.generated_at, tz=timezone.utc).isoformat(timespec="seconds")
    candidate = report.candidate_name or "Unknown candidate"
    role = report.job_title or "Unknown role"

    lines: list[str] = []
    lines.append(f"# Interview Report — {candidate}, {role}")
    lines.append(
        f"**Recommendation:** {o.recommendation} · "
        f"Technical {o.technical_avg}/5 · Coherence {o.coherence_avg}/5"
    )
    lines.append(f"**Generated:** {iso} ({report.model})")
    lines.append("")
    lines.append("## Summary")
    lines.append(o.summary or "_no summary available_")
    lines.append("")
    lines.append("### Strengths")
    for s in o.strengths or ["_(none listed)_"]:
        lines.append(f"- {s}")
    lines.append("")
    lines.append("### Concerns")
    for s in o.concerns or ["_(none listed)_"]:
        lines.append(f"- {s}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Per-Answer Breakdown")
    for t in report.turns:
        tech = "n/a" if t.technical_score is None else f"{t.technical_score}/5"
        coh = "n/a" if t.coherence_score is None else f"{t.coherence_score}/5"
        suffix = " _(scoring failed)_" if t.technical_score is None else ""
        lines.append(f"### Turn {t.turn_index} — Technical {tech} · Coherence {coh}{suffix}")
        lines.append(f"> **Q:** {t.question}")
        lines.append(f"> **A:** {t.answer}")
        lines.append(f"**Technical:** {t.technical_rationale}")
        lines.append(f"**Coherence:** {t.coherence_rationale}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# On-disk cache
# ---------------------------------------------------------------------------
def _json_path(dir_: Path, interview_id: str) -> Path:
    return dir_ / f"{interview_id}_report.json"


def _md_path(dir_: Path, interview_id: str) -> Path:
    return dir_ / f"{interview_id}_report.md"


def write_cache(dir_: Path, report: InterviewReport) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    _json_path(dir_, report.interview_id).write_text(report.model_dump_json(indent=2))
    _md_path(dir_, report.interview_id).write_text(render_markdown(report))


def read_cache(dir_: Path, interview_id: str) -> Optional[InterviewReport]:
    p = _json_path(dir_, interview_id)
    if not p.exists():
        return None
    try:
        return InterviewReport.model_validate_json(p.read_text())
    except Exception:
        return None


def delete_cache(dir_: Path, interview_id: str) -> bool:
    removed = False
    for p in (_json_path(dir_, interview_id), _md_path(dir_, interview_id)):
        if p.exists():
            p.unlink()
            removed = True
    return removed


# ---------------------------------------------------------------------------
# Recruit DB write-back
# ---------------------------------------------------------------------------
def write_back_scores(conn: sqlite3.Connection, report: InterviewReport) -> None:
    """Update interviews.scores_json with the headline numbers.

    Best-effort — failures are swallowed so the on-disk JSON stays
    authoritative even when the recruit DB is unavailable.
    """
    payload = {
        "recommendation": report.overall.recommendation,
        "technical_avg": report.overall.technical_avg,
        "coherence_avg": report.overall.coherence_avg,
        "generated_at": report.generated_at,
        "model": report.model,
    }
    try:
        conn.execute(
            "UPDATE interviews SET scores_json = ? WHERE id = ?",
            (json.dumps(payload), report.interview_id),
        )
        try:
            conn.commit()
        except sqlite3.Error:
            pass
    except sqlite3.Error:
        pass
