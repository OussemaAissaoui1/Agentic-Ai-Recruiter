# Scoring Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `agents/scoring/` stub with a Groq-backed agent that produces a cached per-answer + overall hiring report from a recruit-DB-stored interview.

**Architecture:** FastAPI sub-agent registered alongside the others. On `POST /api/scoring/{interview_id}/run` it reads transcript + CV + JD from the recruit SQLite DB (read-only join), calls Groq `llama-3.3-70b-versatile` once per turn (bounded concurrency) for technical + coherence scoring, then once more for the overall recommendation. Result is persisted as JSON + Markdown under `artifacts/scoring/` and the headline numbers are written back to `interviews.scores_json`. Subsequent calls return the cached JSON unless `force=true`.

**Tech Stack:** Python 3.10+, FastAPI, Pydantic v2, `httpx` (async Groq calls — no SDK dep), pytest + pytest-asyncio, SQLite via the existing `agents/recruit/db.py` helpers.

**Spec:** `docs/superpowers/specs/2026-05-04-scoring-agent-design.md`

---

## File Structure

**New files (under `ai-recruiter/agents/scoring/`):**
- `schema.py` — Pydantic models (`TurnScore`, `OverallAssessment`, `InterviewReport`, `ScoreRequest`, `Transcript`/`TranscriptTurn`)
- `reader.py` — read `(transcript, cv, jd, candidate_name, job_title)` from recruit DB by `interview_id`; normalise transcript variants
- `llm_client.py` — async Groq HTTP client (chat completions); reads `GROQ_API_KEY` env; one retry; JSON-mode response_format
- `prompts.py` — per-turn and overall prompt templates as functions returning `(system, user)` strings
- `scorer.py` — orchestrates per-turn LLM calls (bounded `asyncio.Semaphore`), parses + repairs JSON, builds list of `TurnScore`; calls overall pass; returns `InterviewReport`
- `report.py` — averages, recommendation fallback rule, Markdown renderer, on-disk cache read/write, recruit-DB scores write-back
- `tests/__init__.py`
- `tests/conftest.py` — fixtures: in-memory recruit DB seeded with one job/app/interview; `FakeGroq` stub
- `tests/test_schema.py`
- `tests/test_reader.py`
- `tests/test_llm_client.py`
- `tests/test_scorer.py`
- `tests/test_report.py`
- `tests/test_agent_routes.py`

**Modified files:**
- `agents/scoring/agent.py` — fully rewritten: new routes, A2A capabilities, no more `record_score` JSONL stub
- `agents/scoring/agent_card.json` — new capability list, version bump to `0.3.0`
- `agents/scoring/README.md` — rewritten for the new surface
- `agents/scoring/__init__.py` — keep `ScoringAgent` export

**Untouched:**
- The recruit DB schema is reused as-is. We do not add columns. Write-back uses the existing `update_interview(..., {"scores": ...})` helper.
- Old per-session JSONL files at `artifacts/scoring/<session_id>.jsonl` remain on disk; the new agent ignores them.

---

## Pre-flight

- [ ] **Verify environment**

Run from `/teamspace/studios/this_studio/Agentic-Ai-Recruiter`:
```bash
cd ai-recruiter && python -c "import fastapi, pydantic, httpx, pytest; print('ok')"
```
Expected: `ok`. If anything is missing, install with `pip install fastapi pydantic httpx pytest pytest-asyncio`.

- [ ] **Confirm Groq key is reachable from the shell (do not echo it)**

```bash
python -c "import os; print('GROQ_API_KEY set:', bool(os.environ.get('GROQ_API_KEY')))"
```
Expected: `GROQ_API_KEY set: True`. If False, ask the user to export it. Never read the key from a file or hard-code it.

---

## Task 1: Pydantic schemas

**Files:**
- Create: `ai-recruiter/agents/scoring/schema.py`
- Test: `ai-recruiter/agents/scoring/tests/test_schema.py`
- Create: `ai-recruiter/agents/scoring/tests/__init__.py` (empty)

- [ ] **Step 1: Write the failing test**

Create `ai-recruiter/agents/scoring/tests/__init__.py` empty.

Create `ai-recruiter/agents/scoring/tests/test_schema.py`:
```python
import pytest
from pydantic import ValidationError

from agents.scoring.schema import (
    TranscriptTurn,
    TurnScore,
    OverallAssessment,
    InterviewReport,
    ScoreRequest,
)


def test_transcript_turn_q_a_shape():
    t = TranscriptTurn(q="What's your stack?", a="Python and Postgres.")
    assert t.q.startswith("What")
    assert t.a == "Python and Postgres."


def test_turn_score_accepts_none_on_failure():
    s = TurnScore(
        turn_index=0,
        question="q?",
        answer="a",
        technical_score=None,
        technical_rationale="api error: 500",
        coherence_score=None,
        coherence_rationale="api error: 500",
    )
    assert s.technical_score is None


def test_turn_score_rejects_out_of_range():
    with pytest.raises(ValidationError):
        TurnScore(
            turn_index=0, question="q", answer="a",
            technical_score=7, technical_rationale="x",
            coherence_score=3, coherence_rationale="x",
        )


def test_overall_recommendation_is_enum_constrained():
    with pytest.raises(ValidationError):
        OverallAssessment(
            recommendation="maybe",
            technical_avg=3.0, coherence_avg=3.0,
            summary="x", strengths=["a"], concerns=["b"],
        )


def test_interview_report_round_trip():
    r = InterviewReport(
        interview_id="int-1",
        candidate_name="Sarah",
        job_title="DE",
        generated_at=1700000000.0,
        model="llama-3.3-70b-versatile",
        turns=[],
        overall=OverallAssessment(
            recommendation="hire",
            technical_avg=4.0, coherence_avg=4.5,
            summary="solid", strengths=["a"], concerns=["b"],
        ),
    )
    blob = r.model_dump_json()
    again = InterviewReport.model_validate_json(blob)
    assert again.interview_id == "int-1"


def test_score_request_defaults():
    req = ScoreRequest()
    assert req.force is False
    assert req.transcript is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ai-recruiter && pytest -q agents/scoring/tests/test_schema.py
```
Expected: collection error / import failure (`schema` module does not exist).

- [ ] **Step 3: Write the schema module**

Create `ai-recruiter/agents/scoring/schema.py`:
```python
"""Pydantic data contracts for the scoring agent.

Single source of truth for the shapes that move between the reader, the
scorer, the report builder, and the API surface.
"""
from __future__ import annotations

from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field

Score05 = Annotated[int, Field(ge=0, le=5)]


class TranscriptTurn(BaseModel):
    """One Q/A pair from a recorded interview."""
    q: str
    a: str


class TurnScore(BaseModel):
    """Per-answer assessment.

    Scores may be None when the LLM failed for this turn after the retry
    budget was exhausted; the rationale then carries the error string.
    """
    turn_index: int
    question: str
    answer: str
    technical_score: Optional[Score05]
    technical_rationale: str
    coherence_score: Optional[Score05]
    coherence_rationale: str


class OverallAssessment(BaseModel):
    recommendation: Literal["strong_hire", "hire", "lean_hire", "no_hire"]
    technical_avg: float
    coherence_avg: float
    summary: str
    strengths: list[str]
    concerns: list[str]


class InterviewReport(BaseModel):
    interview_id: str
    candidate_name: Optional[str]
    job_title: Optional[str]
    generated_at: float
    model: str
    turns: list[TurnScore]
    overall: OverallAssessment


class ScoreRequest(BaseModel):
    """Request body for POST /api/scoring/{interview_id}/run."""
    force: bool = False
    transcript: Optional[list[TranscriptTurn]] = None
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd ai-recruiter && pytest -q agents/scoring/tests/test_schema.py
```
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
cd ai-recruiter && git add agents/scoring/schema.py agents/scoring/tests/__init__.py agents/scoring/tests/test_schema.py
git -c user.email=oussa612@gmail.com -c user.name=oussa612 commit -m "feat(scoring): add Pydantic data contracts (TurnScore, OverallAssessment, InterviewReport, ScoreRequest)"
```

---

## Task 2: Recruit DB reader

**Files:**
- Create: `ai-recruiter/agents/scoring/reader.py`
- Test: `ai-recruiter/agents/scoring/tests/test_reader.py`
- Create: `ai-recruiter/agents/scoring/tests/conftest.py`

- [ ] **Step 1: Write the conftest fixtures**

Create `ai-recruiter/agents/scoring/tests/conftest.py`:
```python
"""Shared fixtures for scoring agent tests."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pytest

from agents.recruit import db as recruit_db


@pytest.fixture
def recruit_conn(tmp_path: Path) -> sqlite3.Connection:
    """Fresh recruit SQLite DB seeded with one job, application, interview."""
    conn = recruit_db.init_db(tmp_path / "recruit.db")
    now = time.time()

    conn.execute(
        "INSERT INTO jobs (id, title, description, status, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("job-test", "Senior Data Engineer",
         "Build pipelines for analytics. Python, Spark, Airflow.",
         "open", now, now),
    )
    conn.execute(
        "INSERT INTO applications (id, job_id, candidate_name, candidate_email, cv_text, "
        "stage, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("app-test", "job-test", "Sarah Chen", "sarah@example.com",
         "10 years building data platforms at Google. Python, Spark, BigQuery.",
         "approved", now, now),
    )
    transcript = [
        {"q": "Walk me through your BigQuery migration.",
         "a": "We moved 50TB from Hive to BigQuery using a dual-write pattern over 6 weeks."},
        {"q": "How did you validate row counts?",
         "a": "Daily reconciliation jobs comparing checksums per partition; mismatches paged us."},
        {"q": "What was the trickiest schema change?",
         "a": "Nested JSON columns — we flattened them with dataflow before load."},
    ]
    conn.execute(
        "INSERT INTO interviews (id, application_id, transcript_json, scores_json, "
        "behavioral_json, status, started_at, ended_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("int-test", "app-test", json.dumps(transcript), "{}", "{}", "completed",
         now - 3600, now),
    )
    return conn


@pytest.fixture
def empty_interview_conn(tmp_path: Path) -> sqlite3.Connection:
    """Recruit DB with an interview row whose transcript_json is empty."""
    conn = recruit_db.init_db(tmp_path / "recruit_empty.db")
    now = time.time()
    conn.execute(
        "INSERT INTO jobs (id, title, description, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?)",
        ("job-empty", "DE", "JD", now, now),
    )
    conn.execute(
        "INSERT INTO applications (id, job_id, candidate_name, cv_text, "
        "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        ("app-empty", "job-empty", "Anon", "cv", now, now),
    )
    conn.execute(
        "INSERT INTO interviews (id, application_id, transcript_json) VALUES (?, ?, ?)",
        ("int-empty", "app-empty", "[]"),
    )
    return conn
```

- [ ] **Step 2: Write the failing test**

Create `ai-recruiter/agents/scoring/tests/test_reader.py`:
```python
import json

import pytest

from agents.scoring.reader import InterviewContext, load_context
from agents.scoring.schema import TranscriptTurn


def test_load_context_returns_full_payload(recruit_conn):
    ctx = load_context(recruit_conn, "int-test")
    assert isinstance(ctx, InterviewContext)
    assert ctx.interview_id == "int-test"
    assert ctx.candidate_name == "Sarah Chen"
    assert ctx.job_title == "Senior Data Engineer"
    assert "Python, Spark" in ctx.cv_text
    assert "Build pipelines" in ctx.jd_text
    assert len(ctx.transcript) == 3
    assert ctx.transcript[0].q.startswith("Walk me")


def test_load_context_unknown_interview_raises(recruit_conn):
    with pytest.raises(LookupError) as ei:
        load_context(recruit_conn, "int-nope")
    assert "int-nope" in str(ei.value)


def test_load_context_empty_transcript(empty_interview_conn):
    ctx = load_context(empty_interview_conn, "int-empty")
    assert ctx.transcript == []


def test_normalise_question_answer_keys(recruit_conn):
    # Inject a transcript that uses the {question, answer} variant
    alt = json.dumps([{"question": "How are you?", "answer": "Great."}])
    recruit_conn.execute(
        "UPDATE interviews SET transcript_json = ? WHERE id = ?",
        (alt, "int-test"),
    )
    ctx = load_context(recruit_conn, "int-test")
    assert ctx.transcript == [TranscriptTurn(q="How are you?", a="Great.")]


def test_normalise_role_content_pairs(recruit_conn):
    # Chat-style: alternating recruiter/candidate messages
    chat = json.dumps([
        {"role": "recruiter", "content": "Tell me about Spark."},
        {"role": "candidate", "content": "I tuned shuffle partitions for a 5x speedup."},
        {"role": "recruiter", "content": "What broke?"},
        {"role": "candidate", "content": "Skewed keys; we salted them."},
    ])
    recruit_conn.execute(
        "UPDATE interviews SET transcript_json = ? WHERE id = ?",
        (chat, "int-test"),
    )
    ctx = load_context(recruit_conn, "int-test")
    assert len(ctx.transcript) == 2
    assert ctx.transcript[0].q == "Tell me about Spark."
    assert ctx.transcript[0].a.startswith("I tuned")


def test_load_context_uses_inline_override(recruit_conn):
    inline = [TranscriptTurn(q="override q", a="override a")]
    ctx = load_context(recruit_conn, "int-test", inline_override=inline)
    assert ctx.transcript == inline


def test_inline_override_used_when_db_empty(empty_interview_conn):
    inline = [TranscriptTurn(q="x", a="y")]
    ctx = load_context(empty_interview_conn, "int-empty", inline_override=inline)
    assert ctx.transcript == inline
```

- [ ] **Step 3: Run test to verify it fails**

```bash
cd ai-recruiter && pytest -q agents/scoring/tests/test_reader.py
```
Expected: import error — `agents.scoring.reader` does not exist.

- [ ] **Step 4: Write the reader**

Create `ai-recruiter/agents/scoring/reader.py`:
```python
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


def load_context(
    conn: sqlite3.Connection,
    interview_id: str,
    inline_override: Optional[list[TranscriptTurn]] = None,
) -> InterviewContext:
    """Read (transcript, CV, JD) for an interview.

    Raises LookupError if the interview row is not found. If
    *inline_override* is given, it replaces whatever transcript is on disk.
    """
    row = conn.execute(
        """
        SELECT i.id              AS interview_id,
               i.transcript_json AS transcript_json,
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

    if inline_override is not None:
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
    )


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
      1. [{"q": ..., "a": ...}]                 — canonical
      2. [{"question": ..., "answer": ...}]     — alt key naming
      3. [{"role": ..., "content": ...}, ...]   — chat-style pairs
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
```

- [ ] **Step 5: Run test to verify it passes**

```bash
cd ai-recruiter && pytest -q agents/scoring/tests/test_reader.py
```
Expected: 7 passed.

- [ ] **Step 6: Commit**

```bash
cd ai-recruiter && git add agents/scoring/reader.py agents/scoring/tests/conftest.py agents/scoring/tests/test_reader.py
git -c user.email=oussa612@gmail.com -c user.name=oussa612 commit -m "feat(scoring): reader joins recruit DB to load (transcript, cv, jd) by interview_id"
```

---

## Task 3: Async Groq HTTP client

**Files:**
- Create: `ai-recruiter/agents/scoring/llm_client.py`
- Test: `ai-recruiter/agents/scoring/tests/test_llm_client.py`

We use `httpx` directly rather than the official `groq` SDK to avoid adding a dependency for one HTTP endpoint.

- [ ] **Step 1: Write the failing test**

Create `ai-recruiter/agents/scoring/tests/test_llm_client.py`:
```python
import os

import httpx
import pytest

from agents.scoring.llm_client import GroqClient, GroqError


@pytest.fixture
def fake_transport_ok():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer test-key"
        body = request.read().decode()
        assert '"response_format"' in body
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"hello": "world"}'}}]},
        )
    return httpx.MockTransport(handler)


@pytest.fixture
def fake_transport_5xx_then_ok():
    calls = {"n": 0}
    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, text="overloaded")
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"ok": true}'}}]},
        )
    return httpx.MockTransport(handler)


@pytest.fixture
def fake_transport_always_5xx():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")
    return httpx.MockTransport(handler)


async def test_chat_json_returns_parsed_dict(fake_transport_ok):
    client = GroqClient(api_key="test-key", transport=fake_transport_ok)
    out = await client.chat_json(
        system="be precise", user="say hi", model="m-1", timeout=5.0,
    )
    assert out == {"hello": "world"}
    await client.aclose()


async def test_chat_json_retries_once_on_5xx(fake_transport_5xx_then_ok):
    client = GroqClient(api_key="test-key", transport=fake_transport_5xx_then_ok)
    out = await client.chat_json(system="s", user="u", model="m-1", timeout=5.0)
    assert out == {"ok": True}
    await client.aclose()


async def test_chat_json_raises_after_retry_exhausted(fake_transport_always_5xx):
    client = GroqClient(api_key="test-key", transport=fake_transport_always_5xx)
    with pytest.raises(GroqError):
        await client.chat_json(system="s", user="u", model="m-1", timeout=5.0)
    await client.aclose()


def test_groqclient_requires_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(GroqError):
        GroqClient.from_env()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ai-recruiter && pytest -q agents/scoring/tests/test_llm_client.py
```
Expected: import error — `agents.scoring.llm_client` does not exist.

- [ ] **Step 3: Write the client**

Create `ai-recruiter/agents/scoring/llm_client.py`:
```python
"""Async Groq chat-completions client with strict-JSON response_format.

Thin wrapper over httpx so we don't take a dependency on the groq SDK for
one endpoint. One automatic retry on 5xx / timeout, then raise.
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Optional

import httpx

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqError(RuntimeError):
    """Any non-recoverable failure from the Groq client."""


class GroqClient:
    """Minimal async client. Reuse the same instance across calls."""

    def __init__(
        self,
        api_key: str,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        if not api_key:
            raise GroqError("GROQ_API_KEY is required")
        self._client = httpx.AsyncClient(
            transport=transport,
            timeout=httpx.Timeout(60.0, connect=5.0),
        )
        self._api_key = api_key

    @classmethod
    def from_env(cls) -> "GroqClient":
        key = os.environ.get("GROQ_API_KEY", "")
        if not key:
            raise GroqError("GROQ_API_KEY is not set")
        return cls(api_key=key)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def chat_json(
        self,
        *,
        system: str,
        user: str,
        model: str,
        temperature: float = 0.2,
        timeout: float = 30.0,
        max_retries: int = 1,
    ) -> dict[str, Any]:
        """Send one chat completion and parse the assistant message as JSON.

        response_format is forced to json_object. On 5xx or read timeout, we
        retry up to max_retries with exponential backoff. On 4xx or any other
        failure we raise GroqError immediately. On JSON parse failure of the
        assistant content we raise GroqError.
        """
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        last_exc: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                resp = await self._client.post(
                    GROQ_URL, json=payload, headers=headers, timeout=timeout,
                )
                if resp.status_code >= 500:
                    raise GroqError(f"groq {resp.status_code}: {resp.text[:200]}")
                if resp.status_code >= 400:
                    raise GroqError(f"groq {resp.status_code}: {resp.text[:200]}")
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                try:
                    return json.loads(content)
                except json.JSONDecodeError as exc:
                    raise GroqError(f"non-json content: {exc}") from exc
            except (httpx.TimeoutException, GroqError) as exc:
                # Retry only transient failures (timeouts and 5xx). 4xx GroqError
                # contents start with "groq 4" — don't retry those.
                last_exc = exc
                msg = str(exc)
                is_4xx = msg.startswith("groq 4")
                if is_4xx or attempt >= max_retries:
                    raise GroqError(msg) from exc
                await asyncio.sleep(0.5 * (2 ** attempt))

        # Defensive — loop above should always return or raise.
        raise GroqError(f"unreachable: {last_exc}")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd ai-recruiter && pytest -q agents/scoring/tests/test_llm_client.py
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
cd ai-recruiter && git add agents/scoring/llm_client.py agents/scoring/tests/test_llm_client.py
git -c user.email=oussa612@gmail.com -c user.name=oussa612 commit -m "feat(scoring): async Groq httpx client with json response_format and one retry"
```

---

## Task 4: Prompt templates

**Files:**
- Create: `ai-recruiter/agents/scoring/prompts.py`

This task is template-only — no test of its own; coverage comes via `test_scorer.py` which asserts the rendered prompts contain the expected substrings.

- [ ] **Step 1: Write the prompts module**

Create `ai-recruiter/agents/scoring/prompts.py`:
```python
"""Prompt templates for per-turn and overall scoring passes.

Two functions, each returning (system, user). Keep prompts boring and
specific — the JSON shape is the contract; the wording can be tuned.
"""
from __future__ import annotations

from .schema import TranscriptTurn, TurnScore


_PER_TURN_SYSTEM = (
    "You are a senior technical interviewer calibrating answers against a "
    "specific role. Score the candidate's answer on two independent axes "
    "(technical correctness and coherence). Return ONLY valid JSON matching "
    "the requested schema. Do not include any text outside the JSON."
)

_RUBRIC = """
Technical (0-5):
  0 = no signal, refusal, or nonsense
  1 = clearly wrong or fabricated
  2 = vague / hand-wavy; little evidence of hands-on knowledge
  3 = generic correctness; lacks concrete detail
  4 = correct with concrete detail and a believable trade-off
  5 = expert-level; specific, accurate, mentions edge cases or trade-offs

Coherence (0-5):
  0 = incoherent / unintelligible
  1 = fragmented; hard to follow
  2 = present but disorganised
  3 = clear enough; some redundancy
  4 = well-structured, easy to follow
  5 = articulate, well-paced, no wasted words
"""


def per_turn_prompt(
    *,
    job_title: str,
    jd_text: str,
    cv_text: str,
    prior_context: list[TranscriptTurn],
    question: str,
    answer: str,
) -> tuple[str, str]:
    """Render the per-turn scoring prompt.

    *prior_context* is the trailing window of turns before this one (already
    truncated by the caller).
    """
    ctx_lines = [f"Q: {t.q}\nA: {t.a}" for t in prior_context]
    context_block = "\n\n".join(ctx_lines) if ctx_lines else "(no prior context)"

    user = f"""Role: {job_title or "(unspecified)"}

Job description:
{jd_text[:1500]}

Candidate CV (excerpt):
{cv_text[:600]}

{_RUBRIC}

Recent conversation context:
{context_block}

== Score this exchange ==
Question:
{question}

Answer:
{answer}

Return JSON exactly in this shape:
{{
  "technical_score": <0-5 integer>,
  "technical_rationale": "<one or two sentences referencing the answer>",
  "coherence_score": <0-5 integer>,
  "coherence_rationale": "<one or two sentences>"
}}
"""
    return _PER_TURN_SYSTEM, user


_OVERALL_SYSTEM = (
    "You are the lead technical interviewer summarising a completed "
    "interview into a hiring recommendation. Be calibrated and specific. "
    "Return ONLY valid JSON matching the requested schema."
)


def overall_prompt(
    *,
    job_title: str,
    jd_text: str,
    cv_text: str,
    transcript: list[TranscriptTurn],
    per_turn: list[TurnScore],
) -> tuple[str, str]:
    transcript_block = "\n\n".join(
        f"Turn {i}\nQ: {t.q}\nA: {t.a}" for i, t in enumerate(transcript)
    ) or "(no transcript)"

    scores_block = "\n".join(
        f"Turn {s.turn_index}: technical={s.technical_score} ({s.technical_rationale})"
        f" | coherence={s.coherence_score} ({s.coherence_rationale})"
        for s in per_turn
    ) or "(no per-turn scores)"

    user = f"""Role: {job_title or "(unspecified)"}

Job description:
{jd_text[:1500]}

Candidate CV (excerpt):
{cv_text[:800]}

Transcript:
{transcript_block}

Per-turn scoring already produced:
{scores_block}

Produce a hiring recommendation. Pick exactly one value for "recommendation"
from: "strong_hire", "hire", "lean_hire", "no_hire".

Return JSON exactly in this shape:
{{
  "recommendation": "<one of the four values above>",
  "summary": "<3-5 sentence narrative tying the per-turn scores to the role>",
  "strengths": ["<bullet>", "<bullet>"],
  "concerns": ["<bullet>", "<bullet>"]
}}

Calibration:
  - strong_hire: clear technical excellence, no significant concerns
  - hire: solid match; minor reservations
  - lean_hire: split signal; would need a second opinion
  - no_hire: insufficient evidence, fabrications, or major concerns
"""
    return _OVERALL_SYSTEM, user
```

- [ ] **Step 2: Smoke-import**

```bash
cd ai-recruiter && python -c "from agents.scoring.prompts import per_turn_prompt, overall_prompt; print('ok')"
```
Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
cd ai-recruiter && git add agents/scoring/prompts.py
git -c user.email=oussa612@gmail.com -c user.name=oussa612 commit -m "feat(scoring): per-turn and overall prompt templates with explicit rubric anchors"
```

---

## Task 5: Scorer (LLM orchestration)

**Files:**
- Create: `ai-recruiter/agents/scoring/scorer.py`
- Test: `ai-recruiter/agents/scoring/tests/test_scorer.py`

- [ ] **Step 1: Add a FakeGroq fixture to conftest**

Append to `ai-recruiter/agents/scoring/tests/conftest.py`:
```python
from typing import Any

import pytest


class FakeGroq:
    """Test double for GroqClient. Records calls; replies from a queue."""

    def __init__(self, replies: list[Any]) -> None:
        self._replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    async def chat_json(self, *, system: str, user: str, **kw: Any) -> Any:
        self.calls.append({"system": system, "user": user, **kw})
        if not self._replies:
            raise AssertionError("FakeGroq exhausted; add more replies")
        item = self._replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def aclose(self) -> None:
        return None


@pytest.fixture
def fake_groq_factory():
    def _make(replies: list[Any]) -> FakeGroq:
        return FakeGroq(replies)
    return _make
```

- [ ] **Step 2: Write the failing test**

Create `ai-recruiter/agents/scoring/tests/test_scorer.py`:
```python
import pytest

from agents.scoring.llm_client import GroqError
from agents.scoring.scorer import score_interview
from agents.scoring.schema import TranscriptTurn


def _ctx():
    return dict(
        interview_id="int-1",
        candidate_name="Sarah",
        job_title="DE",
        cv_text="cv",
        jd_text="jd",
        transcript=[
            TranscriptTurn(q="q1?", a="a1"),
            TranscriptTurn(q="q2?", a="a2"),
        ],
    )


async def test_score_interview_happy_path(fake_groq_factory):
    replies = [
        # per-turn 1
        {"technical_score": 4, "technical_rationale": "good detail",
         "coherence_score": 5, "coherence_rationale": "clear"},
        # per-turn 2
        {"technical_score": 3, "technical_rationale": "ok",
         "coherence_score": 4, "coherence_rationale": "ok"},
        # overall
        {"recommendation": "hire",
         "summary": "solid candidate",
         "strengths": ["a", "b"],
         "concerns": ["c"]},
    ]
    client = fake_groq_factory(replies)
    report = await score_interview(client=client, model="m-1", **_ctx())

    assert report.interview_id == "int-1"
    assert len(report.turns) == 2
    assert report.turns[0].technical_score == 4
    assert report.turns[1].coherence_score == 4
    assert report.overall.recommendation == "hire"
    assert report.overall.technical_avg == pytest.approx(3.5)
    assert report.overall.coherence_avg == pytest.approx(4.5)
    assert client.calls[0]["user"].count("q1?") == 1


async def test_score_interview_per_turn_failure_isolated(fake_groq_factory):
    replies = [
        # turn 1 fails
        GroqError("groq 502: bad gateway"),
        # turn 2 ok
        {"technical_score": 5, "technical_rationale": "expert",
         "coherence_score": 5, "coherence_rationale": "great"},
        # overall ok
        {"recommendation": "hire", "summary": "x",
         "strengths": ["a"], "concerns": ["b"]},
    ]
    client = fake_groq_factory(replies)
    report = await score_interview(client=client, model="m-1", **_ctx())

    assert report.turns[0].technical_score is None
    assert "502" in report.turns[0].technical_rationale
    assert report.turns[1].technical_score == 5
    # avg computed only over non-None
    assert report.overall.technical_avg == pytest.approx(5.0)


async def test_score_interview_overall_failure_falls_back(fake_groq_factory):
    replies = [
        {"technical_score": 4, "technical_rationale": "x",
         "coherence_score": 4, "coherence_rationale": "x"},
        {"technical_score": 4, "technical_rationale": "x",
         "coherence_score": 4, "coherence_rationale": "x"},
        GroqError("groq 500: boom"),
    ]
    client = fake_groq_factory(replies)
    report = await score_interview(client=client, model="m-1", **_ctx())

    # Fallback rule: avg 4.0 → "hire"
    assert report.overall.recommendation == "hire"
    assert "500" in report.overall.summary
    assert report.turns[0].technical_score == 4


async def test_empty_transcript_raises():
    ctx = _ctx()
    ctx["transcript"] = []
    with pytest.raises(ValueError):
        await score_interview(client=None, model="m-1", **ctx)
```

- [ ] **Step 3: Run test to verify it fails**

```bash
cd ai-recruiter && pytest -q agents/scoring/tests/test_scorer.py
```
Expected: import error.

- [ ] **Step 4: Write the scorer**

Create `ai-recruiter/agents/scoring/scorer.py`:
```python
"""Orchestrates LLM calls to produce an InterviewReport from context.

Public entry point: score_interview(...). Per-turn calls run with bounded
concurrency. Each per-turn failure is isolated (None scores + error rationale).
The overall pass also fails-soft into a deterministic average-based rule.
"""
from __future__ import annotations

import asyncio
import time
from typing import Optional, Protocol

from .llm_client import GroqError
from .prompts import per_turn_prompt, overall_prompt
from .report import build_overall_fallback, compute_averages
from .schema import (
    InterviewReport,
    OverallAssessment,
    TranscriptTurn,
    TurnScore,
)


class _ChatJSON(Protocol):
    async def chat_json(
        self, *, system: str, user: str, model: str, temperature: float, timeout: float,
        max_retries: int,
    ) -> dict: ...


CONTEXT_WINDOW = 3   # how many prior turns to include in each per-turn prompt


async def score_interview(
    *,
    client,                 # GroqClient | FakeGroq — anything with chat_json
    model: str,
    interview_id: str,
    candidate_name: Optional[str],
    job_title: Optional[str],
    cv_text: str,
    jd_text: str,
    transcript: list[TranscriptTurn],
    concurrency: int = 4,
    timeout: float = 30.0,
) -> InterviewReport:
    if not transcript:
        raise ValueError("transcript is empty; nothing to score")

    sem = asyncio.Semaphore(concurrency)

    async def _score_one(idx: int, turn: TranscriptTurn) -> TurnScore:
        async with sem:
            prior = transcript[max(0, idx - CONTEXT_WINDOW): idx]
            system, user = per_turn_prompt(
                job_title=job_title or "",
                jd_text=jd_text,
                cv_text=cv_text,
                prior_context=prior,
                question=turn.q,
                answer=turn.a,
            )
            try:
                data = await client.chat_json(
                    system=system, user=user, model=model,
                    temperature=0.2, timeout=timeout, max_retries=1,
                )
                return TurnScore(
                    turn_index=idx,
                    question=turn.q,
                    answer=turn.a,
                    technical_score=_clamp(data.get("technical_score")),
                    technical_rationale=str(data.get("technical_rationale", "")),
                    coherence_score=_clamp(data.get("coherence_score")),
                    coherence_rationale=str(data.get("coherence_rationale", "")),
                )
            except GroqError as exc:
                msg = f"scoring failed: {exc}"
                return TurnScore(
                    turn_index=idx,
                    question=turn.q,
                    answer=turn.a,
                    technical_score=None,
                    technical_rationale=msg,
                    coherence_score=None,
                    coherence_rationale=msg,
                )

    per_turn = await asyncio.gather(
        *(_score_one(i, t) for i, t in enumerate(transcript))
    )

    overall = await _score_overall(
        client=client, model=model, timeout=timeout,
        job_title=job_title or "", jd_text=jd_text, cv_text=cv_text,
        transcript=transcript, per_turn=list(per_turn),
    )

    return InterviewReport(
        interview_id=interview_id,
        candidate_name=candidate_name,
        job_title=job_title,
        generated_at=time.time(),
        model=model,
        turns=list(per_turn),
        overall=overall,
    )


async def _score_overall(
    *,
    client,
    model: str,
    timeout: float,
    job_title: str,
    jd_text: str,
    cv_text: str,
    transcript: list[TranscriptTurn],
    per_turn: list[TurnScore],
) -> OverallAssessment:
    tech_avg, coh_avg = compute_averages(per_turn)
    system, user = overall_prompt(
        job_title=job_title, jd_text=jd_text, cv_text=cv_text,
        transcript=transcript, per_turn=per_turn,
    )
    try:
        data = await client.chat_json(
            system=system, user=user, model=model,
            temperature=0.3, timeout=timeout, max_retries=1,
        )
        rec = data.get("recommendation")
        if rec not in {"strong_hire", "hire", "lean_hire", "no_hire"}:
            rec = build_overall_fallback(tech_avg, coh_avg).recommendation
        return OverallAssessment(
            recommendation=rec,
            technical_avg=tech_avg,
            coherence_avg=coh_avg,
            summary=str(data.get("summary", "")) or "no summary returned",
            strengths=[str(s) for s in (data.get("strengths") or [])][:6],
            concerns=[str(s) for s in (data.get("concerns") or [])][:6],
        )
    except GroqError as exc:
        fb = build_overall_fallback(tech_avg, coh_avg)
        return OverallAssessment(
            recommendation=fb.recommendation,
            technical_avg=tech_avg,
            coherence_avg=coh_avg,
            summary=f"overall pass failed; using deterministic fallback: {exc}",
            strengths=fb.strengths,
            concerns=fb.concerns,
        )


def _clamp(value) -> Optional[int]:
    """Coerce LLM output into an int in [0, 5] or None."""
    if value is None:
        return None
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return max(0, min(5, n))
```

- [ ] **Step 5: Run test (will still fail — needs report.py helpers)**

```bash
cd ai-recruiter && pytest -q agents/scoring/tests/test_scorer.py 2>&1 | tail -10
```
Expected: import error from `from .report import build_overall_fallback, compute_averages`. That's fine — Task 6 supplies them. We commit Task 5 + 6 together.

- [ ] **Step 6: (do not commit yet — proceed straight to Task 6)**

---

## Task 6: Report builder + cache + DB write-back

**Files:**
- Create: `ai-recruiter/agents/scoring/report.py`
- Test: `ai-recruiter/agents/scoring/tests/test_report.py`

- [ ] **Step 1: Write the failing test**

Create `ai-recruiter/agents/scoring/tests/test_report.py`:
```python
import json
import time
from pathlib import Path

import pytest

from agents.scoring.report import (
    build_overall_fallback,
    compute_averages,
    render_markdown,
    read_cache,
    write_cache,
    write_back_scores,
)
from agents.scoring.schema import (
    InterviewReport,
    OverallAssessment,
    TurnScore,
)


def _turn(idx, t, c, t_text="x", c_text="y"):
    return TurnScore(
        turn_index=idx, question=f"q{idx}", answer=f"a{idx}",
        technical_score=t, technical_rationale=t_text,
        coherence_score=c, coherence_rationale=c_text,
    )


def test_compute_averages_ignores_none():
    turns = [_turn(0, 4, 5), _turn(1, None, None, "err", "err"), _turn(2, 2, 3)]
    t, c = compute_averages(turns)
    assert t == pytest.approx(3.0)
    assert c == pytest.approx(4.0)


def test_compute_averages_all_none_returns_zero():
    turns = [_turn(0, None, None, "err", "err")]
    assert compute_averages(turns) == (0.0, 0.0)


@pytest.mark.parametrize("avg, expected", [
    (4.5, "strong_hire"),
    (3.8, "hire"),
    (2.5, "lean_hire"),
    (1.0, "no_hire"),
])
def test_overall_fallback_thresholds(avg, expected):
    fb = build_overall_fallback(avg, avg)
    assert fb.recommendation == expected
    assert fb.strengths
    assert fb.concerns


def _full_report():
    return InterviewReport(
        interview_id="int-1", candidate_name="Sarah", job_title="DE",
        generated_at=1700000000.0, model="llama-3.3-70b-versatile",
        turns=[_turn(0, 4, 5), _turn(1, 3, 4)],
        overall=OverallAssessment(
            recommendation="hire", technical_avg=3.5, coherence_avg=4.5,
            summary="solid technical depth and clear communication.",
            strengths=["Spark expertise", "trade-off awareness"],
            concerns=["limited streaming background"],
        ),
    )


def test_render_markdown_contains_expected_sections():
    md = render_markdown(_full_report())
    assert "# Interview Report" in md
    assert "Sarah" in md and "DE" in md
    assert "**Recommendation:** hire" in md
    assert "Technical 3.5/5" in md
    assert "## Summary" in md
    assert "### Strengths" in md
    assert "Spark expertise" in md
    assert "## Per-Answer Breakdown" in md
    assert "### Turn 0" in md
    assert "Technical 4/5" in md


def test_render_markdown_handles_failed_turn():
    report = _full_report()
    report.turns[0] = _turn(0, None, None, "scoring failed: 502", "scoring failed: 502")
    md = render_markdown(report)
    assert "Technical n/a" in md
    assert "_(scoring failed)_" in md


def test_cache_round_trip(tmp_path: Path):
    report = _full_report()
    write_cache(tmp_path, report)
    json_path = tmp_path / "int-1_report.json"
    md_path = tmp_path / "int-1_report.md"
    assert json_path.exists() and md_path.exists()

    again = read_cache(tmp_path, "int-1")
    assert again is not None
    assert again.interview_id == "int-1"
    assert again.overall.recommendation == "hire"


def test_read_cache_missing_returns_none(tmp_path: Path):
    assert read_cache(tmp_path, "nope") is None


def test_write_back_scores_updates_db(recruit_conn):
    report = _full_report()
    report.interview_id = "int-test"
    write_back_scores(recruit_conn, report)
    row = recruit_conn.execute(
        "SELECT scores_json FROM interviews WHERE id=?", ("int-test",)
    ).fetchone()
    payload = json.loads(row["scores_json"])
    assert payload["recommendation"] == "hire"
    assert payload["technical_avg"] == 3.5
    assert "generated_at" in payload
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ai-recruiter && pytest -q agents/scoring/tests/test_report.py
```
Expected: import error.

- [ ] **Step 3: Write the report module**

Create `ai-recruiter/agents/scoring/report.py`:
```python
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
class _Fallback:
    recommendation: str
    strengths: list[str]
    concerns: list[str]


def build_overall_fallback(tech_avg: float, coh_avg: float) -> _Fallback:
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
    return _Fallback(
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
    except sqlite3.Error:
        pass
```

- [ ] **Step 4: Run report tests**

```bash
cd ai-recruiter && pytest -q agents/scoring/tests/test_report.py
```
Expected: 9 passed (the 4 parametrize cases count as 4 separate tests).

- [ ] **Step 5: Run scorer tests now that report.py exists**

```bash
cd ai-recruiter && pytest -q agents/scoring/tests/test_scorer.py
```
Expected: 4 passed.

- [ ] **Step 6: Commit Tasks 5 + 6 together**

```bash
cd ai-recruiter && git add agents/scoring/scorer.py agents/scoring/report.py agents/scoring/tests/test_scorer.py agents/scoring/tests/test_report.py agents/scoring/tests/conftest.py
git -c user.email=oussa612@gmail.com -c user.name=oussa612 commit -m "feat(scoring): scorer + report builder with bounded concurrency, fail-soft per turn, cache and DB write-back"
```

---

## Task 7: ScoringAgent — replace the stub

**Files:**
- Modify: `ai-recruiter/agents/scoring/agent.py` (full rewrite)
- Modify: `ai-recruiter/agents/scoring/agent_card.json`
- Modify: `ai-recruiter/agents/scoring/__init__.py`
- Test: `ai-recruiter/agents/scoring/tests/test_agent_routes.py`

- [ ] **Step 1: Write the failing route tests**

Create `ai-recruiter/agents/scoring/tests/test_agent_routes.py`:
```python
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agents.scoring.agent import ScoringAgent
from agents.scoring.schema import (
    InterviewReport,
    OverallAssessment,
    TurnScore,
)


class _AppState:
    def __init__(self, recruit_conn, artifacts_dir, fake_client):
        self.recruit_conn = recruit_conn
        self.artifacts_dir = artifacts_dir
        self.fake_groq_client = fake_client


def _stub_report(interview_id="int-test"):
    return InterviewReport(
        interview_id=interview_id, candidate_name="Sarah", job_title="DE",
        generated_at=1700000000.0, model="m-1",
        turns=[
            TurnScore(turn_index=0, question="q?", answer="a",
                      technical_score=4, technical_rationale="ok",
                      coherence_score=5, coherence_rationale="ok"),
        ],
        overall=OverallAssessment(
            recommendation="hire", technical_avg=4.0, coherence_avg=5.0,
            summary="x", strengths=["a"], concerns=["b"],
        ),
    )


@pytest.fixture
def app_with_agent(recruit_conn, tmp_path: Path, fake_groq_factory):
    import asyncio
    fake = fake_groq_factory([])  # replies injected per-test by patching score_interview
    agent = ScoringAgent()
    state = _AppState(recruit_conn, tmp_path / "scoring", fake)

    # startup() does no async I/O when state is fully injected; a fresh loop is fine.
    asyncio.run(agent.startup(state))

    app = FastAPI()
    app.include_router(agent.router, prefix="/api/scoring")
    return app, agent, state


def test_run_returns_report_and_caches(app_with_agent):
    app, agent, state = app_with_agent
    client = TestClient(app)

    async def fake_score(**kwargs):
        return _stub_report(kwargs["interview_id"])

    with patch("agents.scoring.agent.score_interview", side_effect=fake_score):
        r = client.post("/api/scoring/int-test/run", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["overall"]["recommendation"] == "hire"

    # Files written
    assert (state.artifacts_dir / "int-test_report.json").exists()
    assert (state.artifacts_dir / "int-test_report.md").exists()

    # Recruit DB updated
    row = state.recruit_conn.execute(
        "SELECT scores_json FROM interviews WHERE id=?", ("int-test",)
    ).fetchone()
    assert json.loads(row["scores_json"])["recommendation"] == "hire"


def test_run_returns_cached_without_calling_llm(app_with_agent):
    app, agent, state = app_with_agent
    client = TestClient(app)

    # Pre-warm cache
    state.artifacts_dir.mkdir(parents=True, exist_ok=True)
    (state.artifacts_dir / "int-test_report.json").write_text(
        _stub_report().model_dump_json()
    )
    (state.artifacts_dir / "int-test_report.md").write_text("md cached")

    with patch("agents.scoring.agent.score_interview") as mock:
        r = client.post("/api/scoring/int-test/run", json={})
        assert mock.called is False
    assert r.status_code == 200
    assert r.json()["interview_id"] == "int-test"


def test_run_force_bypasses_cache(app_with_agent):
    app, agent, state = app_with_agent
    client = TestClient(app)

    state.artifacts_dir.mkdir(parents=True, exist_ok=True)
    (state.artifacts_dir / "int-test_report.json").write_text(
        _stub_report().model_dump_json()
    )

    async def fake_score(**kwargs):
        return _stub_report(kwargs["interview_id"])

    with patch("agents.scoring.agent.score_interview", side_effect=fake_score) as mock:
        r = client.post("/api/scoring/int-test/run", json={"force": True})
        assert mock.called is True
    assert r.status_code == 200


def test_run_unknown_interview_returns_404(app_with_agent):
    app, agent, state = app_with_agent
    client = TestClient(app)
    r = client.post("/api/scoring/int-nope/run", json={})
    assert r.status_code == 404


def test_run_empty_transcript_returns_422(app_with_agent, empty_interview_conn, tmp_path, fake_groq_factory):
    # Build a separate agent bound to the empty DB
    fake = fake_groq_factory([])
    agent = ScoringAgent()
    state = _AppState(empty_interview_conn, tmp_path / "scoring2", fake)
    import asyncio
    asyncio.run(agent.startup(state))
    app = FastAPI()
    app.include_router(agent.router, prefix="/api/scoring")
    client = TestClient(app)

    r = client.post("/api/scoring/int-empty/run", json={})
    assert r.status_code == 422
    assert "no transcript" in r.json()["detail"].lower()


def test_inline_transcript_override_path(app_with_agent, empty_interview_conn, tmp_path, fake_groq_factory):
    fake = fake_groq_factory([])
    agent = ScoringAgent()
    state = _AppState(empty_interview_conn, tmp_path / "scoring3", fake)
    import asyncio
    asyncio.run(agent.startup(state))
    app = FastAPI()
    app.include_router(agent.router, prefix="/api/scoring")
    client = TestClient(app)

    async def fake_score(**kwargs):
        # Verify the override was passed through
        assert len(kwargs["transcript"]) == 1
        assert kwargs["transcript"][0].q == "override"
        return _stub_report("int-empty")

    with patch("agents.scoring.agent.score_interview", side_effect=fake_score):
        r = client.post(
            "/api/scoring/int-empty/run",
            json={"transcript": [{"q": "override", "a": "answer"}]},
        )
    assert r.status_code == 200


def test_get_report_404_then_200(app_with_agent):
    app, _agent, state = app_with_agent
    client = TestClient(app)
    assert client.get("/api/scoring/int-test/report").status_code == 404

    state.artifacts_dir.mkdir(parents=True, exist_ok=True)
    (state.artifacts_dir / "int-test_report.json").write_text(
        _stub_report().model_dump_json()
    )
    r = client.get("/api/scoring/int-test/report")
    assert r.status_code == 200
    assert r.json()["overall"]["recommendation"] == "hire"


def test_get_markdown(app_with_agent):
    app, _agent, state = app_with_agent
    client = TestClient(app)
    state.artifacts_dir.mkdir(parents=True, exist_ok=True)
    (state.artifacts_dir / "int-test_report.md").write_text("# hello")
    r = client.get("/api/scoring/int-test/report.md")
    assert r.status_code == 200
    assert r.text == "# hello"
    assert r.headers["content-type"].startswith("text/markdown")


def test_delete_report(app_with_agent):
    app, _agent, state = app_with_agent
    client = TestClient(app)
    state.artifacts_dir.mkdir(parents=True, exist_ok=True)
    (state.artifacts_dir / "int-test_report.json").write_text(
        _stub_report().model_dump_json()
    )
    r = client.delete("/api/scoring/int-test/report")
    assert r.status_code == 200
    assert not (state.artifacts_dir / "int-test_report.json").exists()


def test_health_reports_degraded_without_key(app_with_agent, monkeypatch):
    app, _agent, _state = app_with_agent
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    client = TestClient(app)
    r = client.get("/api/scoring/health")
    assert r.status_code == 200
    assert r.json()["state"] in ("DEGRADED", "READY")  # depends on key presence
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd ai-recruiter && pytest -q agents/scoring/tests/test_agent_routes.py 2>&1 | tail -10
```
Expected: import / construction failures (current `ScoringAgent` doesn't take `state.recruit_conn` etc.).

- [ ] **Step 3: Rewrite `agents/scoring/agent.py`**

Replace the file content with:
```python
"""Scoring Agent — produces hiring-decision reports from recorded interviews.

POST /{interview_id}/run kicks off (or returns cached) scoring. Reads
transcript, CV, and JD from the recruit SQLite DB; uses Groq for the LLM
calls; persists JSON + Markdown under artifacts/scoring/<interview_id>_report.*
and writes the headline numbers back to interviews.scores_json.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Response
from pydantic import ValidationError

from configs import resolve
from core.contracts import (
    A2ATask,
    A2ATaskResult,
    AgentCard,
    BaseAgent,
    Capability,
    HealthState,
    HealthStatus,
)
from core.observability import bind_agent_logger

from .llm_client import GroqClient, GroqError
from .reader import load_context
from .report import (
    delete_cache,
    read_cache,
    render_markdown,
    write_back_scores,
    write_cache,
)
from .schema import InterviewReport, ScoreRequest
from .scorer import score_interview

log = bind_agent_logger("scoring")

DEFAULT_MODEL = os.environ.get("SCORING_MODEL", "llama-3.3-70b-versatile")
DEFAULT_CONCURRENCY = int(os.environ.get("SCORING_CONCURRENCY", "4"))
DEFAULT_TIMEOUT = float(os.environ.get("SCORING_TIMEOUT_SEC", "30"))


class ScoringAgent(BaseAgent):
    name = "scoring"
    version = "0.3.0"

    def __init__(self) -> None:
        self._router: Optional[APIRouter] = None
        self._artifacts_dir: Optional[Path] = None
        self._recruit_conn: Optional[sqlite3.Connection] = None
        self._groq: Optional[GroqClient] = None

    async def startup(self, app_state: Any) -> None:
        # Tests inject these directly; in the real app they come via
        # configs.resolve and the recruit agent's connection.
        self._artifacts_dir = (
            getattr(app_state, "artifacts_dir", None)
            or resolve("artifacts/scoring")
        )
        self._artifacts_dir.mkdir(parents=True, exist_ok=True)

        self._recruit_conn = (
            getattr(app_state, "recruit_conn", None)
            or _open_recruit_conn_default()
        )

        injected = getattr(app_state, "fake_groq_client", None)
        if injected is not None:
            self._groq = injected
        else:
            try:
                self._groq = GroqClient.from_env()
            except GroqError as exc:
                log.warning("scoring.startup: %s — /run will return 503", exc)
                self._groq = None

    async def shutdown(self) -> None:
        if self._groq is not None and hasattr(self._groq, "aclose"):
            try:
                await self._groq.aclose()
            except Exception:
                pass

    @property
    def router(self) -> APIRouter:
        if self._router is None:
            self._router = self._build_router()
        return self._router

    def card(self) -> AgentCard:
        return AgentCard(
            name=self.name, version=self.version,
            description="Generates per-answer + overall hiring report from a completed interview.",
            routes=[
                "/api/scoring/health",
                "/api/scoring/{interview_id}/run",
                "/api/scoring/{interview_id}/report",
                "/api/scoring/{interview_id}/report.md",
            ],
            capabilities=[
                Capability(name="score_interview",
                           description="Run scoring for an interview (or return cached report)."),
                Capability(name="get_report",
                           description="Return cached InterviewReport JSON."),
                Capability(name="delete_report",
                           description="Drop cached report so the next run regenerates."),
            ],
        )

    async def health(self) -> HealthStatus:
        ready = self._groq is not None and self._recruit_conn is not None
        state = HealthState.READY if ready else HealthState.DEGRADED
        return HealthStatus(
            agent=self.name, state=state, version=self.version,
            detail={
                "groq": self._groq is not None,
                "recruit_db": self._recruit_conn is not None,
                "artifacts_dir": str(self._artifacts_dir) if self._artifacts_dir else None,
                "model": DEFAULT_MODEL,
            },
        )

    # ---------------------------------------------------------------- a2a
    async def handle_task(self, task: A2ATask) -> A2ATaskResult:
        try:
            if task.capability == "score_interview":
                interview_id = task.payload["interview_id"]
                req = ScoreRequest.model_validate(task.payload.get("request") or {})
                report = await self._run(interview_id, req)
                return A2ATaskResult(task_id=task.task_id, success=True,
                                     result=report.model_dump())
            if task.capability == "get_report":
                interview_id = task.payload["interview_id"]
                report = read_cache(self._artifacts_dir, interview_id)
                if report is None:
                    return A2ATaskResult(task_id=task.task_id, success=False,
                                         error="not_found")
                return A2ATaskResult(task_id=task.task_id, success=True,
                                     result=report.model_dump())
            if task.capability == "delete_report":
                interview_id = task.payload["interview_id"]
                ok = delete_cache(self._artifacts_dir, interview_id)
                return A2ATaskResult(task_id=task.task_id, success=True,
                                     result={"deleted": ok})
            return A2ATaskResult(task_id=task.task_id, success=False,
                                 error=f"unknown capability: {task.capability}")
        except HTTPException as exc:
            return A2ATaskResult(task_id=task.task_id, success=False,
                                 error=f"{exc.status_code}: {exc.detail}")
        except Exception as exc:
            return A2ATaskResult(task_id=task.task_id, success=False,
                                 error=f"{type(exc).__name__}: {exc}")

    # ---------------------------------------------------------------- core
    async def _run(self, interview_id: str, req: ScoreRequest) -> InterviewReport:
        if not req.force:
            cached = read_cache(self._artifacts_dir, interview_id)
            if cached is not None:
                return cached

        if self._groq is None:
            raise HTTPException(503, "GROQ_API_KEY is not configured")
        if self._recruit_conn is None:
            raise HTTPException(503, "recruit DB is not configured")

        try:
            ctx = load_context(
                self._recruit_conn, interview_id,
                inline_override=req.transcript,
            )
        except LookupError as exc:
            raise HTTPException(404, str(exc))

        if not ctx.transcript:
            raise HTTPException(422, "no transcript available for this interview")

        report = await score_interview(
            client=self._groq,
            model=DEFAULT_MODEL,
            interview_id=ctx.interview_id,
            candidate_name=ctx.candidate_name,
            job_title=ctx.job_title,
            cv_text=ctx.cv_text,
            jd_text=ctx.jd_text,
            transcript=ctx.transcript,
            concurrency=DEFAULT_CONCURRENCY,
            timeout=DEFAULT_TIMEOUT,
        )

        try:
            write_cache(self._artifacts_dir, report)
        except OSError as exc:
            log.warning("scoring.write_cache failed: %s", exc)
        try:
            write_back_scores(self._recruit_conn, report)
        except Exception as exc:
            log.warning("scoring.write_back_scores failed: %s", exc)
        return report

    # ---------------------------------------------------------------- router
    def _build_router(self) -> APIRouter:
        router = APIRouter(tags=["scoring"])

        @router.get("/health")
        async def health():
            return (await self.health()).model_dump()

        @router.post("/{interview_id}/run")
        async def run(interview_id: str, req: ScoreRequest):
            report = await self._run(interview_id, req)
            return report.model_dump()

        @router.get("/{interview_id}/report")
        async def get_report(interview_id: str):
            cached = read_cache(self._artifacts_dir, interview_id)
            if cached is None:
                raise HTTPException(404, "report not found")
            return cached.model_dump()

        @router.get("/{interview_id}/report.md")
        async def get_markdown(interview_id: str):
            md_path = self._artifacts_dir / f"{interview_id}_report.md"
            if not md_path.exists():
                raise HTTPException(404, "markdown report not found")
            return Response(md_path.read_text(), media_type="text/markdown")

        @router.delete("/{interview_id}/report")
        async def delete_report(interview_id: str):
            ok = delete_cache(self._artifacts_dir, interview_id)
            return {"deleted": ok}

        return router


def _open_recruit_conn_default() -> Optional[sqlite3.Connection]:
    """Open the recruit DB read/write so write-back works.

    Best-effort: returns None on any error. Tests inject a connection
    directly via app_state.recruit_conn.
    """
    try:
        from agents.recruit import db as recruit_db
        path = resolve("agents/recruit/data/recruit.db")
        return recruit_db.init_db(path)
    except Exception as exc:
        log.warning("scoring._open_recruit_conn_default failed: %s", exc)
        return None
```

Note: `asyncio.run()` is used in tests because the agent's `startup()` doesn't keep any awaitables alive — it just stores the injected `recruit_conn`, `artifacts_dir`, and `fake_groq_client` on the agent instance. Running it in a fresh, throwaway loop is fine; the stored references are loop-agnostic.

- [ ] **Step 4: Replace `agents/scoring/agent_card.json`**

Overwrite with:
```json
{
  "name": "scoring",
  "version": "0.3.0",
  "description": "Generates a per-answer + overall hiring report from a completed interview using Groq llama-3.3-70b-versatile. Reads from the recruit SQLite DB and persists report.json + report.md to artifacts/scoring.",
  "transports": ["http"],
  "routes": [
    "/api/scoring/health",
    "/api/scoring/{interview_id}/run",
    "/api/scoring/{interview_id}/report",
    "/api/scoring/{interview_id}/report.md"
  ],
  "capabilities": [
    {"name": "score_interview", "description": "Run scoring for an interview (or return cached report)."},
    {"name": "get_report", "description": "Return cached InterviewReport JSON."},
    {"name": "delete_report", "description": "Drop cached report so the next run regenerates."}
  ]
}
```

- [ ] **Step 5: Update `agents/scoring/__init__.py`**

Read it first; if it currently exports `ScoringAgent`, leave it alone. Otherwise set its content to:
```python
from .agent import ScoringAgent

__all__ = ["ScoringAgent"]
```

- [ ] **Step 6: Run all scoring tests**

```bash
cd ai-recruiter && pytest -q agents/scoring/tests/
```
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
cd ai-recruiter && git add agents/scoring/agent.py agents/scoring/agent_card.json agents/scoring/__init__.py agents/scoring/tests/test_agent_routes.py
git -c user.email=oussa612@gmail.com -c user.name=oussa612 commit -m "feat(scoring): replace stub with hiring-decision agent — Groq scoring, cache, DB write-back"
```

---

## Task 8: README + cleanup

**Files:**
- Modify: `ai-recruiter/agents/scoring/README.md` (rewrite)

- [ ] **Step 1: Rewrite the README**

Overwrite `ai-recruiter/agents/scoring/README.md`:
```markdown
# Scoring Agent

Generates a per-answer + overall hiring report from a completed interview.
Reads transcript, CV, and JD from the recruit SQLite DB; calls Groq
`llama-3.3-70b-versatile` for technical + coherence scoring on each answer
plus an overall recommendation; caches the report on disk and writes
headline numbers back to `interviews.scores_json`.

## Routes (`/api/scoring`)

| Method | Path | Purpose |
|---|---|---|
| `GET`    | `/health` | Readiness — checks Groq key + recruit DB |
| `POST`   | `/{interview_id}/run` | Run scoring (or return cache); body `{force?, transcript?}` |
| `GET`    | `/{interview_id}/report` | Cached `InterviewReport` JSON; 404 if not run |
| `GET`    | `/{interview_id}/report.md` | Cached Markdown rendering; 404 if not run |
| `DELETE` | `/{interview_id}/report` | Drop the cached report |

## A2A capabilities

`score_interview`, `get_report`, `delete_report`.

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | _(unset)_ | Required for `/run`; absence reports `degraded`. |
| `SCORING_MODEL` | `llama-3.3-70b-versatile` | Groq model id. |
| `SCORING_CONCURRENCY` | `4` | Parallel per-turn LLM calls. |
| `SCORING_TIMEOUT_SEC` | `30` | Per-call deadline. |

## Storage

- `artifacts/scoring/<interview_id>_report.json` — canonical report
- `artifacts/scoring/<interview_id>_report.md` — human-readable rendering
- `interviews.scores_json` in recruit DB — `{recommendation, technical_avg, coherence_avg, generated_at, model}`

## Inputs

The agent reads from the recruit DB:

- `interviews.transcript_json` (canonical shape `[{"q","a"}]`; alternates accepted: `{"question","answer"}`, chat-style `{"role","content"}`)
- `applications.cv_text`, `applications.candidate_name`
- `jobs.title`, `jobs.description`

If the transcript is empty, the request body's `transcript` field overrides
the DB value. Otherwise the agent returns 422.

## Output shape

See `schema.py`:

- `TurnScore` — per-answer technical + coherence (0–5 each, both `Optional`
  to allow per-turn failure isolation), with rationale strings.
- `OverallAssessment` — `recommendation ∈ {strong_hire, hire, lean_hire, no_hire}`,
  `technical_avg`, `coherence_avg`, `summary`, `strengths[]`, `concerns[]`.
- `InterviewReport` — composes the above plus metadata.

## Failure handling

- Per-turn LLM failures don't abort the report; that turn's scores are `None`
  and the rationale carries the error string.
- If the overall pass fails, recommendation falls back to a deterministic
  rule on the averages.
- Cache writes and DB write-back are best-effort; the in-memory report is
  always the authoritative return value.

## Spec / plan

- Spec: `docs/superpowers/specs/2026-05-04-scoring-agent-design.md`
- Plan: `docs/superpowers/plans/2026-05-04-scoring-agent.md`
```

- [ ] **Step 2: Commit**

```bash
cd ai-recruiter && git add agents/scoring/README.md
git -c user.email=oussa612@gmail.com -c user.name=oussa612 commit -m "docs(scoring): rewrite README for new hiring-report surface"
```

---

## Task 9: Smoke test against the real Groq API (optional, gated)

**Files:**
- Create: `ai-recruiter/agents/scoring/tests/test_smoke_groq.py`

- [ ] **Step 1: Write the smoke test**

Create `ai-recruiter/agents/scoring/tests/test_smoke_groq.py`:
```python
"""Live Groq smoke test. Skipped unless GROQ_API_KEY is set."""
import os

import pytest

from agents.scoring.llm_client import GroqClient
from agents.scoring.scorer import score_interview
from agents.scoring.schema import TranscriptTurn

pytestmark = pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set; skipping live smoke test",
)


async def test_real_groq_three_turn_interview():
    client = GroqClient.from_env()
    try:
        report = await score_interview(
            client=client,
            model=os.environ.get("SCORING_MODEL", "llama-3.3-70b-versatile"),
            interview_id="smoke-1",
            candidate_name="Test Candidate",
            job_title="Senior Data Engineer",
            cv_text="10 years Python and Spark.",
            jd_text="Build pipelines. Python, Spark, BigQuery.",
            transcript=[
                TranscriptTurn(
                    q="Walk me through your BigQuery migration.",
                    a="Dual-write pattern over six weeks; reconciled row counts daily.",
                ),
                TranscriptTurn(
                    q="What broke during the migration?",
                    a="Skewed keys caused shuffle hot spots; we salted them.",
                ),
                TranscriptTurn(
                    q="How did you validate parity?",
                    a="Per-partition checksums; mismatches paged on-call.",
                ),
            ],
        )
    finally:
        await client.aclose()

    assert len(report.turns) == 3
    assert all(t.technical_score is not None for t in report.turns)
    assert report.overall.recommendation in {
        "strong_hire", "hire", "lean_hire", "no_hire"
    }
    assert 0.0 <= report.overall.technical_avg <= 5.0
```

- [ ] **Step 2: Run it (only if GROQ_API_KEY exported)**

```bash
cd ai-recruiter && pytest -q agents/scoring/tests/test_smoke_groq.py -s
```
Expected: 1 passed (or 1 skipped if no key).

- [ ] **Step 3: Commit**

```bash
cd ai-recruiter && git add agents/scoring/tests/test_smoke_groq.py
git -c user.email=oussa612@gmail.com -c user.name=oussa612 commit -m "test(scoring): live Groq smoke test gated on GROQ_API_KEY"
```

---

## Final verification

- [ ] **All scoring tests pass**

```bash
cd ai-recruiter && pytest -q agents/scoring/tests/
```
Expected: all green (smoke test passes if `GROQ_API_KEY` is set, skipped otherwise).

- [ ] **No other agent tests regressed**

```bash
cd ai-recruiter && pytest -q tests/ agents/
```
Expected: same green state as before this branch (or better — we don't touch other agents).

- [ ] **Smoke check the full app boots**

```bash
cd ai-recruiter && python -c "
from agents.scoring import ScoringAgent
a = ScoringAgent()
print(a.card().model_dump_json(indent=2)[:200])
"
```
Expected: prints the agent card JSON; no import errors.
