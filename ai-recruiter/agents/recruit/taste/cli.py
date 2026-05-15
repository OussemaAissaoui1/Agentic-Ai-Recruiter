"""CLI for the recruiter-taste POC.

Usage:
    python -m agents.recruit.taste.cli seed_synthetic_recruiter \\
        --persona pragmatist --n 100

Creates synthetic application + decision rows for a fake recruiter and
runs the fit. Used to demo / regression-test the learning loop without
involving real users. The fake recruiter id is derived from the persona
name (e.g. "synth-pragmatist") so re-runs accumulate decisions on the
same recruiter unless --reset is passed.

Other commands:
    list                            — print all recruiter profiles
    show --recruiter <id>           — pretty-print one profile
    reset --recruiter <id>          — drop a profile (keeps decisions)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid

from configs import resolve

from .. import db as recruit_db
from .features import FEATURES_VERSION
from .learner import fit_recruiter_weights, load_profile, reset_profile
from .synthetic import PERSONAS, generate_synthetic_decisions


def _open_conn():
    return recruit_db.init_db(resolve("agents/recruit/data/recruit.db"))


def _seed_job_if_missing(conn, job_id: str = "synth-job") -> None:
    if conn.execute("SELECT 1 FROM jobs WHERE id = ?", (job_id,)).fetchone():
        return
    now = time.time()
    conn.execute(
        "INSERT INTO jobs (id, title, description, status, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (job_id, "Synthetic Test Job",
         "Auto-generated job for recruiter-taste seeding.", "open", now, now),
    )


def cmd_seed(args: argparse.Namespace) -> int:
    if args.persona not in PERSONAS:
        print(
            f"unknown persona: {args.persona!r}; pick from {list(PERSONAS)}",
            file=sys.stderr,
        )
        return 2
    if args.n <= 0:
        print("--n must be positive", file=sys.stderr)
        return 2

    conn = _open_conn()
    recruiter_id = args.recruiter or f"synth-{args.persona}"
    if args.reset:
        # Wipe both decisions and profile for a clean seeding run.
        conn.execute(
            "DELETE FROM recruiter_decisions WHERE recruiter_id = ?",
            (recruiter_id,),
        )
        reset_profile(conn, recruiter_id)
        print(f"reset {recruiter_id}: cleared decisions and profile.")

    _seed_job_if_missing(conn)

    decisions = generate_synthetic_decisions(args.persona, args.n, seed=args.seed)
    now = time.time()
    for i, d in enumerate(decisions):
        app_id = f"synth-{recruiter_id}-{uuid.uuid4().hex[:8]}"
        conn.execute(
            "INSERT INTO applications (id, job_id, candidate_name, "
            "candidate_email, stage, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (app_id, "synth-job", f"Synthetic Candidate {i}",
             f"synth-{i}@example.invalid",
             d.decision, now, now),
        )
        conn.execute(
            "INSERT INTO recruiter_decisions (id, recruiter_id, application_id, "
            "features_snapshot_json, features_version, decision, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (f"dec-{uuid.uuid4().hex[:12]}", recruiter_id, app_id,
             json.dumps(d.features), FEATURES_VERSION, d.decision, now),
        )

    profile = fit_recruiter_weights(conn, recruiter_id)
    print(f"\nseeded {args.n} decisions for {recruiter_id} (persona={args.persona})")
    _print_profile(profile)
    return 0


def cmd_list(_args: argparse.Namespace) -> int:
    conn = _open_conn()
    rows = conn.execute(
        "SELECT recruiter_id, n_decisions, confidence_level, version, updated_at "
        "FROM recruiter_profiles ORDER BY updated_at DESC"
    ).fetchall()
    if not rows:
        print("(no profiles)")
        return 0
    for r in rows:
        print(
            f"  {r['recruiter_id']:30s}  n={r['n_decisions']:4d}  "
            f"{r['confidence_level']:8s}  v{r['version']}"
        )
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    conn = _open_conn()
    profile = load_profile(conn, args.recruiter)
    if profile is None:
        print(f"no profile for {args.recruiter!r}", file=sys.stderr)
        return 1
    _print_profile(profile)
    return 0


def cmd_reset(args: argparse.Namespace) -> int:
    conn = _open_conn()
    existed = reset_profile(conn, args.recruiter)
    print(f"reset {args.recruiter}: {'profile cleared' if existed else '(no profile)'}")
    return 0


def _print_profile(profile) -> None:
    print(f"  recruiter_id     : {profile.recruiter_id}")
    print(f"  n_decisions      : {profile.n_decisions}")
    print(f"  confidence       : {profile.confidence.value}")
    print(f"  version          : {profile.version}")
    print(f"  intercept        : {profile.intercept:+.3f}")
    print(f"  weights (z-scored, sorted by magnitude):")
    ranked = sorted(profile.weights.items(), key=lambda kv: -abs(kv[1]))
    for name, w in ranked:
        bar = "+" * int(abs(w) * 10) if w > 0 else "-" * int(abs(w) * 10)
        print(f"    {name:25s} {w:+.3f}  {bar}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="agents.recruit.taste.cli")
    sub = p.add_subparsers(dest="cmd", required=True)

    seed_p = sub.add_parser(
        "seed_synthetic_recruiter",
        help="Generate synthetic decisions for a persona and run a fit.",
    )
    seed_p.add_argument("--persona", required=True, choices=list(PERSONAS))
    seed_p.add_argument("--n", type=int, default=100)
    seed_p.add_argument("--seed", type=int, default=42)
    seed_p.add_argument(
        "--recruiter", default=None,
        help="Recruiter id (default: synth-<persona>).",
    )
    seed_p.add_argument(
        "--reset", action="store_true",
        help="Clear existing decisions + profile for this recruiter first.",
    )
    seed_p.set_defaults(func=cmd_seed)

    list_p = sub.add_parser("list", help="List all recruiter profiles.")
    list_p.set_defaults(func=cmd_list)

    show_p = sub.add_parser("show", help="Pretty-print a recruiter profile.")
    show_p.add_argument("--recruiter", required=True)
    show_p.set_defaults(func=cmd_show)

    reset_p = sub.add_parser("reset", help="Drop a profile (keeps decisions).")
    reset_p.add_argument("--recruiter", required=True)
    reset_p.set_defaults(func=cmd_reset)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
