"""SQLite schema + CRUD helpers for the recruit agent.

The agent owns four tables:
  - jobs
  - applications
  - notifications
  - interviews

All access goes through small synchronous helpers; callers wrap them in
``asyncio.to_thread`` for use inside async routes.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("agents.recruit")


# ---------------------------------------------------------------------------
# Connection / schema
# ---------------------------------------------------------------------------
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
    id              TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    team            TEXT,
    location        TEXT,
    work_mode       TEXT,
    level           TEXT,
    employment_type TEXT,
    salary_min      INTEGER,
    salary_max      INTEGER,
    currency        TEXT DEFAULT 'USD',
    description     TEXT,
    must_have_json  TEXT DEFAULT '[]',
    nice_to_have_json TEXT DEFAULT '[]',
    status          TEXT DEFAULT 'open',
    created_at      REAL,
    updated_at      REAL
);

CREATE TABLE IF NOT EXISTS applications (
    id                  TEXT PRIMARY KEY,
    job_id              TEXT NOT NULL,
    candidate_name      TEXT,
    candidate_email     TEXT,
    cv_text             TEXT,
    cv_filename         TEXT,
    cv_path             TEXT,
    fit_score           REAL,
    matched_skills_json TEXT DEFAULT '[]',
    missing_skills_json TEXT DEFAULT '[]',
    stage               TEXT DEFAULT 'applied',
    notes               TEXT,
    created_at          REAL,
    updated_at          REAL,
    FOREIGN KEY(job_id) REFERENCES jobs(id)
);

CREATE TABLE IF NOT EXISTS notifications (
    id          TEXT PRIMARY KEY,
    user_role   TEXT,
    user_id     TEXT,
    kind        TEXT,
    title       TEXT,
    body        TEXT,
    link        TEXT,
    read        INTEGER DEFAULT 0,
    created_at  REAL
);

CREATE TABLE IF NOT EXISTS interviews (
    id              TEXT PRIMARY KEY,
    application_id  TEXT NOT NULL,
    transcript_json TEXT DEFAULT '[]',
    scores_json     TEXT DEFAULT '{}',
    behavioral_json TEXT DEFAULT '{}',
    status          TEXT DEFAULT 'scheduled',
    started_at      REAL,
    ended_at        REAL,
    FOREIGN KEY(application_id) REFERENCES applications(id)
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_apps_job_id ON applications(job_id);
CREATE INDEX IF NOT EXISTS idx_apps_email  ON applications(candidate_email);
CREATE INDEX IF NOT EXISTS idx_apps_stage  ON applications(stage);
CREATE INDEX IF NOT EXISTS idx_notif_user  ON notifications(user_role, user_id);
CREATE INDEX IF NOT EXISTS idx_int_app     ON interviews(application_id);

-- Recruiter-taste (POC): per-recruiter preference learning. Free-form
-- recruiter_id (TEXT) — matches the existing notifications.user_id pattern;
-- there is no users table in this codebase. See agents/recruit/taste/.
CREATE TABLE IF NOT EXISTS candidate_features (
    application_id    TEXT NOT NULL,
    features_version  INTEGER NOT NULL,
    features_json     TEXT NOT NULL,
    computed_at       REAL NOT NULL,
    PRIMARY KEY (application_id, features_version),
    FOREIGN KEY (application_id) REFERENCES applications(id)
);

CREATE TABLE IF NOT EXISTS recruiter_decisions (
    id                      TEXT PRIMARY KEY,
    recruiter_id            TEXT NOT NULL,
    application_id          TEXT NOT NULL,
    features_snapshot_json  TEXT NOT NULL,
    features_version        INTEGER NOT NULL,
    decision                TEXT NOT NULL CHECK (decision IN ('approved', 'rejected')),
    dwell_seconds           REAL,
    notes_text              TEXT,
    created_at              REAL NOT NULL,
    FOREIGN KEY (application_id) REFERENCES applications(id)
);

CREATE INDEX IF NOT EXISTS idx_recdec_recruiter ON recruiter_decisions(recruiter_id);
CREATE INDEX IF NOT EXISTS idx_recdec_app       ON recruiter_decisions(application_id);

CREATE TABLE IF NOT EXISTS recruiter_profiles (
    recruiter_id      TEXT PRIMARY KEY,
    weights_json      TEXT NOT NULL,
    n_decisions       INTEGER NOT NULL DEFAULT 0,
    confidence_level  TEXT NOT NULL CHECK (confidence_level IN ('cold', 'warming', 'warm')),
    version           INTEGER NOT NULL DEFAULT 1,
    updated_at        REAL NOT NULL
);
"""


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db(db_path: Path) -> sqlite3.Connection:
    """Open a connection, ensure tables exist, run idempotent data migrations,
    and return it."""
    conn = _connect(db_path)
    conn.executescript(SCHEMA_SQL)
    # Idempotent column adds for older DBs that pre-date a migration.
    # SQLite has no `ADD COLUMN IF NOT EXISTS`, so we swallow the duplicate-
    # column error specifically.
    for stmt in (
        "ALTER TABLE applications ADD COLUMN cv_path TEXT",
    ):
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError as e:
            if "duplicate column name" not in str(e).lower():
                raise
    # One-shot data migration: legacy stages 'screening' and 'interview' both
    # meant "recruiter green-lit this candidate for interview". Map both to
    # the canonical 'approved' value. Idempotent: safe to run on every boot.
    conn.execute(
        "UPDATE applications SET stage = 'approved' "
        "WHERE stage IN ('screening', 'interview')"
    )
    return conn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _now() -> float:
    return time.time()


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _row_to_job(r: sqlite3.Row) -> Dict[str, Any]:
    d = dict(r)
    d["must_have"] = json.loads(d.pop("must_have_json") or "[]")
    d["nice_to_have"] = json.loads(d.pop("nice_to_have_json") or "[]")
    return d


def _row_to_application(r: sqlite3.Row) -> Dict[str, Any]:
    d = dict(r)
    d["matched_skills"] = json.loads(d.pop("matched_skills_json") or "[]")
    d["missing_skills"] = json.loads(d.pop("missing_skills_json") or "[]")
    # cv_path is an absolute filesystem path used by the CV-serving route.
    # Drop it from public dicts — consumers should fetch the file via the
    # GET /applications/{id}/cv endpoint instead. The CV route itself uses
    # get_application_cv_path() below.
    d.pop("cv_path", None)
    return d


def get_application_cv_path(
    conn: sqlite3.Connection, app_id: str
) -> Optional[str]:
    row = conn.execute(
        "SELECT cv_path FROM applications WHERE id = ?", (app_id,),
    ).fetchone()
    if row is None:
        return None
    val = row["cv_path"]
    return str(val) if val else None


def _row_to_notification(r: sqlite3.Row) -> Dict[str, Any]:
    d = dict(r)
    d["read"] = bool(d.get("read"))
    return d


def _row_to_interview(r: sqlite3.Row) -> Dict[str, Any]:
    d = dict(r)
    d["transcript"] = json.loads(d.pop("transcript_json") or "[]")
    d["scores"] = json.loads(d.pop("scores_json") or "{}")
    d["behavioral"] = json.loads(d.pop("behavioral_json") or "{}")
    return d


# ---------------------------------------------------------------------------
# Jobs CRUD
# ---------------------------------------------------------------------------
def list_jobs(
    conn: sqlite3.Connection,
    status: Optional[str] = None,
    q: Optional[str] = None,
) -> List[Dict[str, Any]]:
    sql = "SELECT * FROM jobs WHERE 1=1"
    args: List[Any] = []
    if status:
        sql += " AND status = ?"
        args.append(status)
    if q:
        like = f"%{q}%"
        sql += " AND (title LIKE ? OR team LIKE ? OR description LIKE ?)"
        args.extend([like, like, like])
    sql += " ORDER BY created_at DESC"
    rows = conn.execute(sql, args).fetchall()
    return [_row_to_job(r) for r in rows]


def get_job(conn: sqlite3.Connection, job_id: str) -> Optional[Dict[str, Any]]:
    r = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return _row_to_job(r) if r else None


def create_job(conn: sqlite3.Connection, payload: Dict[str, Any]) -> Dict[str, Any]:
    jid = payload.get("id") or _new_id("job")
    now = _now()
    conn.execute(
        """
        INSERT INTO jobs (id, title, team, location, work_mode, level,
                          employment_type, salary_min, salary_max, currency,
                          description, must_have_json, nice_to_have_json,
                          status, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            jid,
            payload.get("title", "Untitled"),
            payload.get("team"),
            payload.get("location"),
            payload.get("work_mode"),
            payload.get("level"),
            payload.get("employment_type"),
            payload.get("salary_min"),
            payload.get("salary_max"),
            payload.get("currency", "USD"),
            payload.get("description"),
            json.dumps(payload.get("must_have", []) or []),
            json.dumps(payload.get("nice_to_have", []) or []),
            payload.get("status", "open"),
            now,
            now,
        ),
    )
    return get_job(conn, jid)  # type: ignore[return-value]


def update_job(
    conn: sqlite3.Connection, job_id: str, patch: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    if not get_job(conn, job_id):
        return None
    fields = []
    args: List[Any] = []
    mapping = {
        "title": "title",
        "team": "team",
        "location": "location",
        "work_mode": "work_mode",
        "level": "level",
        "employment_type": "employment_type",
        "salary_min": "salary_min",
        "salary_max": "salary_max",
        "currency": "currency",
        "description": "description",
        "status": "status",
    }
    for k, col in mapping.items():
        if k in patch and patch[k] is not None:
            fields.append(f"{col} = ?")
            args.append(patch[k])
    if "must_have" in patch and patch["must_have"] is not None:
        fields.append("must_have_json = ?")
        args.append(json.dumps(patch["must_have"]))
    if "nice_to_have" in patch and patch["nice_to_have"] is not None:
        fields.append("nice_to_have_json = ?")
        args.append(json.dumps(patch["nice_to_have"]))
    if not fields:
        return get_job(conn, job_id)
    fields.append("updated_at = ?")
    args.append(_now())
    args.append(job_id)
    conn.execute(f"UPDATE jobs SET {', '.join(fields)} WHERE id = ?", args)
    return get_job(conn, job_id)


def delete_job(conn: sqlite3.Connection, job_id: str) -> bool:
    cur = conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
    return cur.rowcount > 0


def count_jobs(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0])


# ---------------------------------------------------------------------------
# Applications CRUD
# ---------------------------------------------------------------------------
def list_applications(
    conn: sqlite3.Connection,
    job_id: Optional[str] = None,
    candidate_email: Optional[str] = None,
    stage: Optional[str] = None,
) -> List[Dict[str, Any]]:
    sql = "SELECT * FROM applications WHERE 1=1"
    args: List[Any] = []
    if job_id:
        sql += " AND job_id = ?"
        args.append(job_id)
    if candidate_email:
        sql += " AND candidate_email = ?"
        args.append(candidate_email)
    if stage:
        sql += " AND stage = ?"
        args.append(stage)
    sql += " ORDER BY created_at DESC"
    return [_row_to_application(r) for r in conn.execute(sql, args).fetchall()]


def get_application(conn: sqlite3.Connection, app_id: str) -> Optional[Dict[str, Any]]:
    r = conn.execute("SELECT * FROM applications WHERE id = ?", (app_id,)).fetchone()
    return _row_to_application(r) if r else None


def create_application(
    conn: sqlite3.Connection, payload: Dict[str, Any]
) -> Dict[str, Any]:
    aid = payload.get("id") or _new_id("app")
    now = _now()
    conn.execute(
        """
        INSERT INTO applications (id, job_id, candidate_name, candidate_email,
                                  cv_text, cv_filename, cv_path, fit_score,
                                  matched_skills_json, missing_skills_json,
                                  stage, notes, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            aid,
            payload["job_id"],
            payload.get("candidate_name"),
            payload.get("candidate_email"),
            payload.get("cv_text"),
            payload.get("cv_filename"),
            payload.get("cv_path"),
            payload.get("fit_score"),
            json.dumps(payload.get("matched_skills", []) or []),
            json.dumps(payload.get("missing_skills", []) or []),
            payload.get("stage", "applied"),
            payload.get("notes"),
            now,
            now,
        ),
    )
    return get_application(conn, aid)  # type: ignore[return-value]


def update_application(
    conn: sqlite3.Connection, app_id: str, patch: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    if not get_application(conn, app_id):
        return None
    fields = []
    args: List[Any] = []
    for k in ("stage", "notes", "candidate_name", "candidate_email", "fit_score"):
        if k in patch and patch[k] is not None:
            fields.append(f"{k} = ?")
            args.append(patch[k])
    if "matched_skills" in patch and patch["matched_skills"] is not None:
        fields.append("matched_skills_json = ?")
        args.append(json.dumps(patch["matched_skills"]))
    if "missing_skills" in patch and patch["missing_skills"] is not None:
        fields.append("missing_skills_json = ?")
        args.append(json.dumps(patch["missing_skills"]))
    if not fields:
        return get_application(conn, app_id)
    fields.append("updated_at = ?")
    args.append(_now())
    args.append(app_id)
    conn.execute(
        f"UPDATE applications SET {', '.join(fields)} WHERE id = ?", args
    )
    return get_application(conn, app_id)


# ---------------------------------------------------------------------------
# Notifications
# ---------------------------------------------------------------------------
def list_notifications(
    conn: sqlite3.Connection,
    user_role: Optional[str] = None,
    user_id: Optional[str] = None,
    unread_only: bool = False,
) -> List[Dict[str, Any]]:
    sql = "SELECT * FROM notifications WHERE 1=1"
    args: List[Any] = []
    if user_role:
        sql += " AND user_role = ?"
        args.append(user_role)
    if user_id:
        sql += " AND user_id = ?"
        args.append(user_id)
    if unread_only:
        sql += " AND read = 0"
    sql += " ORDER BY created_at DESC"
    return [_row_to_notification(r) for r in conn.execute(sql, args).fetchall()]


def create_notification(
    conn: sqlite3.Connection, payload: Dict[str, Any]
) -> Dict[str, Any]:
    nid = payload.get("id") or _new_id("ntf")
    conn.execute(
        """
        INSERT INTO notifications (id, user_role, user_id, kind, title, body,
                                   link, read, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            nid,
            payload.get("user_role"),
            payload.get("user_id"),
            payload.get("kind"),
            payload.get("title"),
            payload.get("body"),
            payload.get("link"),
            1 if payload.get("read") else 0,
            _now(),
        ),
    )
    r = conn.execute(
        "SELECT * FROM notifications WHERE id = ?", (nid,)
    ).fetchone()
    return _row_to_notification(r)


def mark_notification_read(
    conn: sqlite3.Connection, notif_id: str, read: bool = True
) -> Optional[Dict[str, Any]]:
    cur = conn.execute(
        "UPDATE notifications SET read = ? WHERE id = ?",
        (1 if read else 0, notif_id),
    )
    if cur.rowcount == 0:
        return None
    r = conn.execute(
        "SELECT * FROM notifications WHERE id = ?", (notif_id,)
    ).fetchone()
    return _row_to_notification(r) if r else None


# ---------------------------------------------------------------------------
# Interviews
# ---------------------------------------------------------------------------
def list_interviews(
    conn: sqlite3.Connection, application_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    sql = "SELECT * FROM interviews"
    args: List[Any] = []
    if application_id:
        sql += " WHERE application_id = ?"
        args.append(application_id)
    sql += " ORDER BY started_at DESC NULLS LAST"
    try:
        rows = conn.execute(sql, args).fetchall()
    except sqlite3.OperationalError:
        # SQLite versions without NULLS LAST — fall back
        sql2 = sql.replace(" NULLS LAST", "")
        rows = conn.execute(sql2, args).fetchall()
    return [_row_to_interview(r) for r in rows]


def get_interview(conn: sqlite3.Connection, iid: str) -> Optional[Dict[str, Any]]:
    r = conn.execute("SELECT * FROM interviews WHERE id = ?", (iid,)).fetchone()
    return _row_to_interview(r) if r else None


def create_interview(
    conn: sqlite3.Connection, payload: Dict[str, Any]
) -> Dict[str, Any]:
    iid = payload.get("id") or _new_id("int")
    conn.execute(
        """
        INSERT INTO interviews (id, application_id, transcript_json,
                                scores_json, behavioral_json, status,
                                started_at, ended_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            iid,
            payload["application_id"],
            json.dumps(payload.get("transcript", []) or []),
            json.dumps(payload.get("scores", {}) or {}),
            json.dumps(payload.get("behavioral", {}) or {}),
            payload.get("status", "scheduled"),
            payload.get("started_at"),
            payload.get("ended_at"),
        ),
    )
    return get_interview(conn, iid)  # type: ignore[return-value]


def update_interview(
    conn: sqlite3.Connection, iid: str, patch: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    if not get_interview(conn, iid):
        return None
    fields = []
    args: List[Any] = []
    for k in ("status", "started_at", "ended_at"):
        if k in patch and patch[k] is not None:
            fields.append(f"{k} = ?")
            args.append(patch[k])
    for k_in, k_col in (
        ("transcript", "transcript_json"),
        ("scores", "scores_json"),
        ("behavioral", "behavioral_json"),
    ):
        if k_in in patch and patch[k_in] is not None:
            fields.append(f"{k_col} = ?")
            args.append(json.dumps(patch[k_in]))
    if not fields:
        return get_interview(conn, iid)
    args.append(iid)
    conn.execute(
        f"UPDATE interviews SET {', '.join(fields)} WHERE id = ?", args
    )
    return get_interview(conn, iid)


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------
SEED_JOBS: List[Dict[str, Any]] = [
    {
        "id": "job-seed-1",
        "title": "Senior Frontend Engineer",
        "team": "Web Platform",
        "location": "Remote (EU)",
        "work_mode": "remote",
        "level": "senior",
        "employment_type": "full_time",
        "salary_min": 110000,
        "salary_max": 145000,
        "currency": "EUR",
        "description": (
            "We're looking for a Senior Frontend Engineer to lead the next "
            "generation of our hiring product. You'll own complex React + "
            "TypeScript surfaces, drive design-system improvements, and "
            "partner with product to ship rapidly and accessibly.\n\n"
            "You'll work alongside a small, senior team where every engineer "
            "ships from idea to production. Expect to influence architecture, "
            "review PRs, and mentor mid-level engineers as we scale.\n\n"
            "Our stack today is React 18, Next.js, Tailwind, tRPC, and "
            "Vercel — but we care more about taste and judgement than the "
            "specific tools you've used."
        ),
        "must_have": [
            "5+ years building production React applications",
            "Deep TypeScript fluency",
            "Strong CSS / accessibility instincts",
            "Comfort with build tooling (Vite, Webpack, esbuild)",
            "Ownership mindset across the full feature lifecycle",
        ],
        "nice_to_have": [
            "Experience with design systems",
            "Background in B2B SaaS",
            "Some Node.js / API work",
            "Open-source contributions",
        ],
        "status": "open",
    },
    {
        "id": "job-seed-2",
        "title": "Staff ML Engineer",
        "team": "AI Platform",
        "location": "San Francisco, CA",
        "work_mode": "hybrid",
        "level": "staff",
        "employment_type": "full_time",
        "salary_min": 230000,
        "salary_max": 310000,
        "currency": "USD",
        "description": (
            "Join our AI Platform team to build the production ML infrastructure "
            "that powers candidate matching, screening, and interview analysis "
            "at scale. You'll architect inference systems serving millions of "
            "scoring requests per day.\n\n"
            "This is a hands-on staff role: you'll write systems code, set "
            "technical direction across multiple squads, and partner with "
            "research to ship novel models from prototype to production.\n\n"
            "Our stack: PyTorch, vLLM, Triton, Kubernetes on GCP, with a "
            "strong bias toward open-source models and low-latency serving."
        ),
        "must_have": [
            "7+ years ML / systems experience",
            "Production experience serving LLMs or embeddings",
            "Strong Python + at least one of CUDA / Rust / Go",
            "Distributed systems fundamentals",
            "Track record of cross-team technical leadership",
        ],
        "nice_to_have": [
            "Experience with vLLM, Triton, or TGI",
            "Prior staff/principal scope",
            "Open-source ML infra contributions",
            "Background in NLP or recommender systems",
        ],
        "status": "open",
    },
    {
        "id": "job-seed-3",
        "title": "Product Designer",
        "team": "Design",
        "location": "Berlin, Germany",
        "work_mode": "hybrid",
        "level": "mid",
        "employment_type": "full_time",
        "salary_min": 70000,
        "salary_max": 95000,
        "currency": "EUR",
        "description": (
            "We're hiring a Product Designer to shape how recruiters and "
            "candidates experience hiring. You'll own end-to-end design work "
            "across discovery, prototyping, and high-fidelity craft.\n\n"
            "Designers here are full collaborators in product strategy. You'll "
            "lead workshops, partner with research, and pair daily with "
            "engineers to ship pixel-precise interfaces.\n\n"
            "We use Figma, FigJam, and Lottie. We care about systems thinking, "
            "motion, and a clear point of view about what good looks like."
        ),
        "must_have": [
            "3+ years product design experience",
            "Strong portfolio with shipped work",
            "Fluency in Figma + design systems",
            "Comfort with interaction and motion design",
            "Strong written + verbal communication",
        ],
        "nice_to_have": [
            "Background in B2B / SaaS",
            "Some prototyping in code",
            "User-research chops",
            "Brand or marketing design experience",
        ],
        "status": "open",
    },
    {
        "id": "job-seed-4",
        "title": "AI Recruiter",
        "team": "People",
        "location": "Remote (Global)",
        "work_mode": "remote",
        "level": "mid",
        "employment_type": "full_time",
        "salary_min": 90000,
        "salary_max": 130000,
        "currency": "USD",
        "description": (
            "We're looking for a recruiter who's excited about AI-augmented "
            "hiring. You'll partner with our product and engineering leaders "
            "to source, screen, and close exceptional candidates worldwide.\n\n"
            "You'll be one of the first power users of the very tool we're "
            "building — feeding insights back to product to make it sharper. "
            "Expect to run experiments on outreach, screening, and candidate "
            "experience.\n\n"
            "We hire globally, async-first, and care a lot about candidate "
            "respect at every step of the process."
        ),
        "must_have": [
            "3+ years technical recruiting experience",
            "Track record closing senior eng / ML talent",
            "Strong written communication",
            "Comfort working async across timezones",
            "Curiosity about AI tooling for hiring",
        ],
        "nice_to_have": [
            "In-house startup experience",
            "Prior agency background",
            "Sourcing automation chops",
            "Some SQL / spreadsheet fluency",
        ],
        "status": "open",
    },
    {
        "id": "job-seed-5",
        "title": "Data Engineer",
        "team": "Data Platform",
        "location": "London, UK",
        "work_mode": "hybrid",
        "level": "mid",
        "employment_type": "full_time",
        "salary_min": 75000,
        "salary_max": 105000,
        "currency": "GBP",
        "description": (
            "Join our Data Platform team to build the pipelines and warehouse "
            "models that power analytics, ML features, and product surfaces. "
            "You'll own ingestion, transformation, and observability end-to-end.\n\n"
            "We process tens of millions of events per day across resumes, "
            "interviews, and product telemetry. You'll partner with analysts "
            "and ML engineers to ship reliable, well-documented data products.\n\n"
            "Our stack: dbt, BigQuery, Airflow, Python, and a healthy amount "
            "of SQL. We invest in tooling so engineers can move fast safely."
        ),
        "must_have": [
            "3+ years data engineering experience",
            "Strong SQL + Python",
            "Experience with dbt or similar",
            "Comfort with cloud warehouses (BQ, Snowflake, Redshift)",
            "Bias toward reliability and data quality",
        ],
        "nice_to_have": [
            "Streaming experience (Kafka, Pub/Sub)",
            "Some infrastructure / Terraform background",
            "Analytics engineering chops",
            "Domain expertise in HR-tech",
        ],
        "status": "open",
    },
]


def seed_if_empty(conn: sqlite3.Connection) -> int:
    """Insert seed jobs if the jobs table is empty. Returns count inserted."""
    if count_jobs(conn) > 0:
        return 0
    for job in SEED_JOBS:
        create_job(conn, job)
    log.info("recruit.db seeded %d jobs", len(SEED_JOBS))
    return len(SEED_JOBS)
