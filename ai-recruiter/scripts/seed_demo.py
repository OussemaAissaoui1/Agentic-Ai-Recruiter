"""Seed the recruit DB with realistic demo data so the dashboard + analytics
have shape.

Writes directly to SQLite (agents/recruit/data/recruit.db) so we can backdate
`created_at` across the last 14 days — the API enforces "now" on POST.

Run:
    python scripts/seed_demo.py
    python scripts/seed_demo.py --reset   # wipe before seeding
    python scripts/seed_demo.py --jobs 6 --apps-per-job 18

Idempotent-ish: by default it skips if the DB already has >= 20 applications.
Use --reset to clear and reseed.
"""

from __future__ import annotations
import argparse
import json
import random
import sqlite3
import sys
import time
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "agents" / "recruit" / "data" / "recruit.db"

DAY = 24 * 60 * 60

# ---------------------------------------------------------------------------
# Synthetic content
# ---------------------------------------------------------------------------
JOBS_BLUEPRINT = [
    {
        "title": "Senior Frontend Engineer",
        "team": "Platform",
        "location": "Remote — EU",
        "level": "Senior",
        "must_have": ["React", "TypeScript", "Testing", "Accessibility"],
        "nice_to_have": ["Next.js", "Tailwind", "Design Systems"],
        "salary_min": 110000,
        "salary_max": 160000,
    },
    {
        "title": "Staff Product Designer",
        "team": "Design",
        "location": "Berlin, DE",
        "level": "Staff",
        "must_have": ["Figma", "Design Systems", "Prototyping", "Mentoring"],
        "nice_to_have": ["Motion", "Brand", "Workshop facilitation"],
        "salary_min": 120000,
        "salary_max": 175000,
    },
    {
        "title": "ML Research Engineer",
        "team": "Research",
        "location": "Remote — Global",
        "level": "Senior",
        "must_have": ["PyTorch", "Python", "Distributed training", "Linear algebra"],
        "nice_to_have": ["CUDA", "Triton", "Paper writing"],
        "salary_min": 140000,
        "salary_max": 210000,
    },
    {
        "title": "Growth Product Manager",
        "team": "Growth",
        "location": "London, UK",
        "level": "Senior",
        "must_have": ["A/B testing", "SQL", "User research", "Roadmapping"],
        "nice_to_have": ["Mixpanel", "Amplitude", "B2B SaaS"],
        "salary_min": 100000,
        "salary_max": 150000,
    },
    {
        "title": "Backend Engineer — Infra",
        "team": "Infra",
        "location": "Remote — EU",
        "level": "Mid",
        "must_have": ["Go", "Postgres", "Kubernetes", "Observability"],
        "nice_to_have": ["Rust", "AWS", "Terraform"],
        "salary_min": 95000,
        "salary_max": 140000,
    },
    {
        "title": "Developer Relations Lead",
        "team": "Brand",
        "location": "NYC, US",
        "level": "Lead",
        "must_have": ["Public speaking", "Writing", "OSS contribution", "Community"],
        "nice_to_have": ["Video", "Workshop facilitation"],
        "salary_min": 105000,
        "salary_max": 150000,
    },
]

FIRST_NAMES = [
    "Aria", "Ben", "Cleo", "Diego", "Elena", "Felix", "Gia", "Hugo",
    "Iris", "Jamal", "Kira", "Leo", "Maya", "Nico", "Otto", "Priya",
    "Quinn", "Rhea", "Sam", "Tara", "Uma", "Vik", "Wren", "Yuki", "Zane",
    "Noor", "Idris", "Saoirse", "Marek", "Anya", "Yousef", "Linnea",
]

LAST_NAMES = [
    "Park", "Okafor", "Müller", "Costa", "Reyes", "Kim", "Patel",
    "Nguyen", "Silva", "Levi", "Tan", "Mori", "Cohen", "Singh",
    "Larsen", "Adeyemi", "Rossi", "Suzuki", "Khan", "Dubois",
    "Schmidt", "Andersson", "Volkov", "Hassan", "Bianchi", "Romero",
]

EMAIL_DOMAINS = [
    # weighted: more gmail, some corporate, some outlook
    ("gmail.com", 5),
    ("outlook.com", 2),
    ("hotmail.com", 1),
    ("proton.me", 1),
    ("acme.dev", 2),
    ("northwind.tech", 1),
    ("lumen.io", 1),
    ("arcade.studio", 1),
]

# Realistic-ish stage distribution. Earlier days lean toward later stages
# (those candidates had time to move forward); recent days lean "applied".
STAGE_DISTRIBUTION = [
    ("applied", 0.45),
    ("approved", 0.20),
    ("interviewed", 0.15),
    ("offer", 0.05),
    ("hired", 0.05),
    ("rejected", 0.10),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _weighted_pick(pairs):
    """pairs = [(value, weight), ...] -> single value sampled by weight."""
    total = sum(w for _, w in pairs)
    r = random.uniform(0, total)
    acc = 0.0
    for value, weight in pairs:
        acc += weight
        if r <= acc:
            return value
    return pairs[-1][0]


def _stage_for_age(age_days: float) -> str:
    """Older applications are more likely to have advanced past 'applied'."""
    if age_days < 1:
        # less than a day old: almost always still 'applied'
        return random.choices(
            ["applied", "approved", "rejected"], weights=[85, 10, 5], k=1
        )[0]
    if age_days < 3:
        return random.choices(
            ["applied", "approved", "rejected", "interviewed"],
            weights=[55, 25, 10, 10], k=1,
        )[0]
    if age_days < 7:
        return random.choices(
            ["applied", "approved", "interviewed", "offer", "rejected"],
            weights=[25, 30, 25, 5, 15], k=1,
        )[0]
    # 7-14 days old: most should be terminal
    return random.choices(
        ["approved", "interviewed", "offer", "hired", "rejected"],
        weights=[15, 30, 15, 15, 25], k=1,
    )[0]


def _fit_score_for_skills(must: list, matched_count: int) -> float:
    """Map matched-skill ratio + noise into a 0..1 fit score with a realistic
    long-tail distribution."""
    if not must:
        base = 0.5
    else:
        base = matched_count / len(must)
    # Sigmoid-ish push toward middling fits + some noise
    noise = random.uniform(-0.15, 0.15)
    fit = 0.35 + 0.55 * base + noise
    return max(0.05, min(0.98, fit))


def _make_application_row(job: dict, when_ts: float) -> tuple:
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    name = f"{first} {last}"
    domain = _weighted_pick(EMAIL_DOMAINS)
    email = f"{first.lower()}.{last.lower().replace('ü', 'u')}@{domain}"

    must = job["must_have"]
    nice = job["nice_to_have"]
    # How many must-haves does this candidate hit? Skewed: most hit ~half.
    matched_count = max(
        0, min(len(must), int(random.gauss(len(must) * 0.6, 1.2)))
    )
    random.shuffle(must)
    matched = list(must[:matched_count])
    missing = [s for s in must if s not in matched]
    # Some extras (from nice-to-have) so matched_skills isn't only must-haves
    extras = random.sample(nice, k=min(len(nice), random.randint(0, 2)))
    matched_full = matched + extras

    fit_score = _fit_score_for_skills(must, matched_count)

    age_days = (time.time() - when_ts) / DAY
    stage = _stage_for_age(age_days)

    cv_text = (
        f"{name}\n{email}\n\nExperience:\n"
        f"- {random.randint(2, 12)} years in {job['team'].lower()} roles\n"
        f"- Strong in {', '.join(matched_full[:3]) or 'core stack'}\n"
        f"- Shipped {random.choice(['onboarding', 'checkout', 'search', 'analytics', 'realtime'])} "
        f"improvements lifting {random.choice(['activation', 'retention', 'conversion'])} "
        f"by {random.randint(5, 28)}%\n"
        f"- Led teams of {random.randint(2, 8)} across {random.randint(2, 4)} timezones\n\n"
        f"Skills: {', '.join(matched_full + missing[:1])}\n"
    )

    notes = None
    if stage in ("offer", "hired"):
        notes = "Strong systems thinker. Reference checks clean."
    elif stage == "rejected":
        notes = "Skill gap on " + (missing[0] if missing else "core stack")

    return (
        str(uuid.uuid4()),
        job["id"],
        name,
        email,
        cv_text,
        f"{first.lower()}_{last.lower()}_cv.txt",
        None,  # cv_path — no real file, just demo data
        fit_score,
        json.dumps(matched_full),
        json.dumps(missing),
        stage,
        notes,
        when_ts,
        when_ts,
    )


# ---------------------------------------------------------------------------
# DB ops
# ---------------------------------------------------------------------------
def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), check_same_thread=False, isolation_level=None)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_db(conn: sqlite3.Connection) -> None:
    """Sanity check — the agent should have initialized the schema already.
    If we run before the API has ever been hit we still want to work."""
    tables = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    required = {"jobs", "applications"}
    if not required.issubset(tables):
        raise SystemExit(
            "Required tables not present in recruit.db. Start the FastAPI "
            "server once so the agent initializes the schema, then re-run."
        )


def _count(conn, table: str) -> int:
    return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def _reset(conn: sqlite3.Connection) -> None:
    # Wipe in FK-safe order: children of applications first, then applications,
    # then jobs. Notifications + recruiter_profiles have no FKs but we leave
    # recruiter_profiles alone so a recruiter's taste model survives a reseed.
    for table in (
        "interviews",
        "candidate_features",
        "recruiter_decisions",
        "applications",
        "jobs",
        "notifications",
    ):
        try:
            conn.execute(f"DELETE FROM {table}")
        except sqlite3.OperationalError:
            pass


def _insert_job(conn, job: dict) -> None:
    now = time.time()
    conn.execute(
        """
        INSERT INTO jobs (id, title, team, location, work_mode, level,
                          employment_type, salary_min, salary_max, currency,
                          description, must_have_json, nice_to_have_json,
                          status, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            job["id"],
            job["title"],
            job["team"],
            job["location"],
            "Remote" if "Remote" in job["location"] else "Hybrid",
            job["level"],
            "Full-time",
            job["salary_min"],
            job["salary_max"],
            "USD",
            f"We're hiring a {job['title']} on the {job['team']} team. "
            f"You'll partner with cross-functional leaders to ship work "
            f"that shapes the product. Demo seed data.",
            json.dumps(job["must_have"]),
            json.dumps(job["nice_to_have"]),
            "open",
            now - random.uniform(7, 30) * DAY,  # job posted some days ago
            now,
        ),
    )


def _insert_application(conn, row: tuple) -> None:
    conn.execute(
        """
        INSERT INTO applications (id, job_id, candidate_name, candidate_email,
                                  cv_text, cv_filename, cv_path, fit_score,
                                  matched_skills_json, missing_skills_json,
                                  stage, notes, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        row,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Seed demo recruit data.")
    ap.add_argument(
        "--reset", action="store_true",
        help="Wipe jobs/applications/interviews/notifications before seeding.",
    )
    ap.add_argument(
        "--jobs", type=int, default=len(JOBS_BLUEPRINT),
        help=f"How many jobs (max {len(JOBS_BLUEPRINT)}).",
    )
    ap.add_argument(
        "--apps-per-job", type=int, default=14,
        help="Approximate applications to create per job.",
    )
    ap.add_argument(
        "--days", type=int, default=14,
        help="Spread applications across the last N days (default 14).",
    )
    ap.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility.",
    )
    args = ap.parse_args()

    random.seed(args.seed)

    if not DB_PATH.exists():
        print(f"recruit.db not found at {DB_PATH}", file=sys.stderr)
        print(
            "Start the FastAPI server once so the schema is initialized, "
            "then re-run.",
            file=sys.stderr,
        )
        return 2

    conn = _connect(DB_PATH)
    try:
        _ensure_db(conn)

        if args.reset:
            print("Wiping existing demo data…")
            _reset(conn)

        existing_apps = _count(conn, "applications")
        existing_jobs = _count(conn, "jobs")
        if existing_apps >= 20 and not args.reset:
            print(
                f"Skipping seed: {existing_jobs} jobs / {existing_apps} apps "
                "already present. Use --reset to wipe and reseed."
            )
            return 0

        jobs_to_use = JOBS_BLUEPRINT[: args.jobs]
        for j in jobs_to_use:
            j["id"] = f"job-seed-{uuid.uuid4().hex[:8]}"
            _insert_job(conn, j)
        print(f"Inserted {len(jobs_to_use)} jobs.")

        now = time.time()
        total_apps = 0
        for job in jobs_to_use:
            # Vary count per job +/-30%
            count = max(
                3,
                int(args.apps_per_job * random.uniform(0.7, 1.3)),
            )
            for _ in range(count):
                # Skew application timestamps: more recent days get more apps
                # (looks like growing pipeline). We sample with a triangular
                # distribution biased toward "now".
                age_days = random.triangular(0, args.days, args.days * 0.35)
                when_ts = now - age_days * DAY
                row = _make_application_row(job, when_ts)
                _insert_application(conn, row)
                total_apps += 1

        print(f"Inserted {total_apps} applications across {args.days} days.")
        print()
        print("Stage breakdown:")
        for stage, count in conn.execute(
            "SELECT stage, COUNT(*) FROM applications GROUP BY stage ORDER BY COUNT(*) DESC"
        ).fetchall():
            print(f"  {stage:<12} {count}")
        print()
        print("Done. Refresh /app or /app/analytics in HireFlow.")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
