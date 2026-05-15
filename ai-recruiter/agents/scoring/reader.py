"""Loads interview context (transcript, CV, JD) from the recruit SQLite DB.

Read-only. Normalises transcript shape variants into the canonical
[{q, a}, ...] form so the rest of the pipeline doesn't have to care.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Iterable, Optional

from .schema import TranscriptTurn


@dataclass(frozen=True)
class InterviewContext:
    interview_id: str
    candidate_name: Optional[str]
    job_title: Optional[str]
    cv_text: str
    jd_text: str
    transcript: list[TranscriptTurn]
    # Vision/voice session summary captured during the live interview by
    # the vision agent's aggregator. Empty dict when not available.
    behavior_summary: dict


def load_context(
    conn: sqlite3.Connection,
    interview_id: str,
    inline_override: Optional[list[TranscriptTurn]] = None,
) -> InterviewContext:
    """Read (transcript, CV, JD, behavior_summary) for an interview.

    Raises LookupError if the interview row is not found. If
    *inline_override* is given, it replaces whatever transcript is on disk.
    """
    row = conn.execute(
        """
        SELECT i.id              AS interview_id,
               i.transcript_json AS transcript_json,
               i.behavioral_json AS behavioral_json,
               a.candidate_name  AS candidate_name,
               a.cv_text         AS cv_text,
               j.title           AS job_title,
               j.description     AS jd_text
        FROM interviews i
        JOIN applications a ON i.application_id = a.id
        JOIN jobs         j ON a.job_id = j.id
        WHERE i.id = ?
        """,
        (interview_id,),
    ).fetchone()

    if row is None:
        raise LookupError(f"interview not found: {interview_id}")

    if inline_override:
        transcript = list(inline_override)
    else:
        transcript = _parse_transcript(row["transcript_json"])

    return InterviewContext(
        interview_id=row["interview_id"],
        candidate_name=row["candidate_name"],
        job_title=row["job_title"],
        cv_text=row["cv_text"] or "",
        jd_text=row["jd_text"] or "",
        transcript=transcript,
        behavior_summary=_parse_behavior(row["behavioral_json"]),
    )


def _parse_behavior(raw: Optional[str]) -> dict:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _parse_transcript(raw: Optional[str]) -> list[TranscriptTurn]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return list(_normalise(data))


def _normalise(items: Iterable[object]) -> Iterable[TranscriptTurn]:
    """Accept three shapes:
      1. [{"q": ..., "a": ...}]                 - canonical
      2. [{"question": ..., "answer": ...}]     - alt key naming
      3. [{"role": ..., "content": ...}, ...]   - chat-style pairs
    """
    pending_recruiter: Optional[str] = None
    for item in items:
        if not isinstance(item, dict):
            continue
        if "q" in item and "a" in item:
            yield TranscriptTurn(q=str(item["q"]), a=str(item["a"]))
            continue
        if "question" in item and "answer" in item:
            yield TranscriptTurn(q=str(item["question"]), a=str(item["answer"]))
            continue
        role = item.get("role")
        content = item.get("content")
        if role and content is not None:
            content_s = str(content)
            if role in ("recruiter", "assistant", "interviewer"):
                pending_recruiter = content_s
            elif role in ("candidate", "user") and pending_recruiter is not None:
                yield TranscriptTurn(q=pending_recruiter, a=content_s)
                pending_recruiter = None
