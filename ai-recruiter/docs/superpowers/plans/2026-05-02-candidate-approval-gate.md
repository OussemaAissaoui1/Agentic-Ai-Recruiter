# Candidate Approval Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the AI interview reachable only after a recruiter has explicitly approved a candidate's application; surface approval and rejection as in-app candidate notifications.

**Architecture:** Stage is the single source of truth — `applied → approved → interviewed`. The gate is enforced server-side at `POST /api/recruit/interviews` (defense in depth) and surfaced in the UI as a stage-aware `WaitingScreen` on `/c/interview/{id}`. Reuse existing endpoints; no new HTTP surface, no schema migration.

**Tech Stack:** Python (FastAPI, sqlite3, pytest), TypeScript (TanStack Router, TanStack Query, Tailwind, framer-motion), Lucide React icons.

**Spec:** `docs/superpowers/specs/2026-05-02-candidate-approval-gate-design.md`

---

## File map

**Backend** (`agents/recruit/`)
- Modify `api.py` — `VALID_STAGES`, `invite_interview`, `update_application`, `create_interview`.
- Modify `db.py` — add data-migration UPDATE inside `init_db()`.

**Backend tests**
- Create `tests/integration/test_application_gate.py` — full integration tests against an isolated `:memory:`/tmp_path SQLite DB.

**Frontend** (`apps/hireflow/src/`)
- Modify `lib/api.ts` — `ApplicationStage` union.
- Modify `lib/queries.ts` — add conditional polling to `useApplication`.
- Create `components/WaitingScreen.tsx` — stage-aware copy.
- Modify `routes/c.interview.$id.tsx` — gate via `WaitingScreen` before mounting interview engine.
- Modify `routes/c.apply.$id.tsx` — drop "Continue to interview" CTA, replace with "Track my application".
- Modify `routes/c.applications.tsx` — `stageLabel` updates, gate Start CTA on `"approved"` only.
- Modify `routes/app.applicants.tsx` — collapse two interview buttons into single "Approve for interview".

---

## Task 1: Backend integration test fixture

**Files:**
- Create: `tests/integration/test_application_gate.py`

**Goal:** Stand up a minimal FastAPI app that mounts only the recruit router, backed by a fresh per-test SQLite DB seeded with the standard jobs. No other agents loaded.

- [ ] **Step 1: Create the test file with fixture and one trivial test**

```python
"""Integration tests for the candidate approval gate.

These tests mount only the recruit agent's router against an isolated SQLite
database (one fresh DB per test) so they don't touch the dev DB and don't
need the full unified app's heavy ML startup.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agents.recruit import db as recruit_db
from agents.recruit.agent import RecruitAgent


@pytest.fixture
def client(tmp_path):
    app = FastAPI()
    agent = RecruitAgent()
    agent._db_path = tmp_path / "recruit.db"
    agent._conn = recruit_db.init_db(agent._db_path)
    recruit_db.seed_if_empty(agent._conn)
    app.include_router(agent.router, prefix="/api/recruit")
    with TestClient(app) as c:
        yield c
    agent._conn.close()


@pytest.fixture
def seeded_job_id(client) -> str:
    jobs = client.get("/api/recruit/jobs").json()
    assert jobs, "seed jobs missing"
    return jobs[0]["id"]


def test_fixture_boots_and_seeds_jobs(client, seeded_job_id):
    assert seeded_job_id.startswith("job-")
```

- [ ] **Step 2: Run the test**

Run: `cd /teamspace/studios/this_studio/Agentic-Ai-Recruiter/ai-recruiter && pytest tests/integration/test_application_gate.py -v`

Expected: PASS — fixture wires up a recruit agent against an isolated DB and returns a seeded job id.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_application_gate.py
git commit -m "test(recruit): add isolated integration fixture for approval gate tests"
```

---

## Task 2: Backend `VALID_STAGES` cleanup

**Files:**
- Modify: `agents/recruit/api.py:114` (`VALID_STAGES`)
- Modify: `tests/integration/test_application_gate.py`

**Goal:** Replace the stage set so `"screening"` is rejected at the boundary and `"approved"` is allowed.

- [ ] **Step 1: Add the failing test for invalid stage rejection**

Append to `tests/integration/test_application_gate.py`:

```python
@pytest.fixture
def applied_app(client, seeded_job_id) -> dict:
    r = client.post(
        "/api/recruit/applications",
        data={"job_id": seeded_job_id, "candidate_email": "ada@example.com",
              "candidate_name": "Ada Lovelace"},
    )
    assert r.status_code == 200, r.text
    return r.json()


def test_apply_starts_in_applied_stage(applied_app):
    assert applied_app["stage"] == "applied"


def test_invalid_stage_rejected(client, applied_app):
    r = client.patch(
        f"/api/recruit/applications/{applied_app['id']}",
        json={"stage": "screening"},
    )
    assert r.status_code == 400
    assert "screening" in r.text.lower() or "stage" in r.text.lower()
```

- [ ] **Step 2: Run tests, verify the new ones fail**

Run: `pytest tests/integration/test_application_gate.py -v`

Expected: `test_apply_starts_in_applied_stage` PASSES (no code change needed). `test_invalid_stage_rejected` FAILS — current `VALID_STAGES` includes `"screening"`, so the patch returns 200.

- [ ] **Step 3: Update `VALID_STAGES` in api.py**

Edit `agents/recruit/api.py` line 114:

```python
VALID_STAGES = {"applied", "approved", "interviewed", "offer", "hired", "rejected"}
```

(Was: `{"applied", "screening", "interviewed", "offer", "hired", "rejected"}`.)

- [ ] **Step 4: Run tests, verify both pass**

Run: `pytest tests/integration/test_application_gate.py -v`

Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add agents/recruit/api.py tests/integration/test_application_gate.py
git commit -m "feat(recruit): replace 'screening' with 'approved' in VALID_STAGES"
```

---

## Task 3: Backend startup data migration

**Files:**
- Modify: `agents/recruit/db.py:108-112` (`init_db`)
- Modify: `tests/integration/test_application_gate.py`

**Goal:** On every boot, rewrite legacy stage values (`"screening"`, `"interview"`) to `"approved"` so existing rows survive the cleanup.

- [ ] **Step 1: Add the failing migration test**

Append to `tests/integration/test_application_gate.py`:

```python
def test_legacy_stage_migration_runs_on_init(tmp_path):
    """Rows with legacy stage values get rewritten to 'approved' at init."""
    db_path = tmp_path / "legacy.db"
    conn = recruit_db.init_db(db_path)
    recruit_db.seed_if_empty(conn)
    job_id = recruit_db.list_jobs(conn)[0]["id"]
    # Simulate two legacy rows (one of each old stage).
    a = recruit_db.create_application(conn, {"job_id": job_id, "candidate_email": "x@y"})
    b = recruit_db.create_application(conn, {"job_id": job_id, "candidate_email": "y@z"})
    conn.execute("UPDATE applications SET stage = 'screening' WHERE id = ?", (a["id"],))
    conn.execute("UPDATE applications SET stage = 'interview' WHERE id = ?", (b["id"],))
    conn.close()

    # Re-init the DB on the same file — migration should run.
    conn2 = recruit_db.init_db(db_path)
    a2 = recruit_db.get_application(conn2, a["id"])
    b2 = recruit_db.get_application(conn2, b["id"])
    conn2.close()
    assert a2["stage"] == "approved"
    assert b2["stage"] == "approved"
```

- [ ] **Step 2: Run, verify it fails**

Run: `pytest tests/integration/test_application_gate.py::test_legacy_stage_migration_runs_on_init -v`

Expected: FAIL — `init_db` does not rewrite legacy stages today; values stay as `"screening"` / `"interview"`.

- [ ] **Step 3: Add the migration to `init_db()` in db.py**

Edit `agents/recruit/db.py` lines 108-112. Replace the `init_db` body so it runs the schema then the rewrite:

```python
def init_db(db_path: Path) -> sqlite3.Connection:
    """Open a connection, ensure tables exist, run idempotent data migrations,
    and return it."""
    conn = _connect(db_path)
    conn.executescript(SCHEMA_SQL)
    # One-shot data migration: legacy stages 'screening' and 'interview' both
    # meant "recruiter green-lit this candidate for interview". Map both to
    # the canonical 'approved' value. Idempotent: safe to run on every boot.
    conn.execute(
        "UPDATE applications SET stage = 'approved' "
        "WHERE stage IN ('screening', 'interview')"
    )
    return conn
```

- [ ] **Step 4: Run, verify it passes**

Run: `pytest tests/integration/test_application_gate.py -v`

Expected: all tests PASS, including the new migration test.

- [ ] **Step 5: Commit**

```bash
git add agents/recruit/db.py tests/integration/test_application_gate.py
git commit -m "feat(recruit): migrate legacy 'screening'/'interview' stages to 'approved' on init"
```

---

## Task 4: Approve sets `stage="approved"` and is idempotent

**Files:**
- Modify: `agents/recruit/api.py:348-392` (`invite_interview`)
- Modify: `tests/integration/test_application_gate.py`

**Goal:** The recruiter-side approval endpoint moves the application to `"approved"` and creates exactly one `interview_invite` notification, even if clicked twice.

- [ ] **Step 1: Add the failing tests**

Append to `tests/integration/test_application_gate.py`:

```python
def test_approve_sets_stage_and_notifies(client, applied_app):
    r = client.post(f"/api/recruit/applications/{applied_app['id']}/invite-interview")
    assert r.status_code == 200, r.text

    app_after = client.get(f"/api/recruit/applications/{applied_app['id']}").json()
    assert app_after["stage"] == "approved"

    notifs = client.get(
        "/api/recruit/notifications",
        params={"user_role": "candidate", "user_id": "ada@example.com"},
    ).json()
    invite = [n for n in notifs if n["kind"] == "interview_invite"]
    assert len(invite) == 1
    assert invite[0]["link"] == f"/c/interview/{applied_app['id']}"


def test_approve_is_idempotent(client, applied_app):
    client.post(f"/api/recruit/applications/{applied_app['id']}/invite-interview")
    client.post(f"/api/recruit/applications/{applied_app['id']}/invite-interview")

    notifs = client.get(
        "/api/recruit/notifications",
        params={"user_role": "candidate", "user_id": "ada@example.com"},
    ).json()
    invite = [n for n in notifs if n["kind"] == "interview_invite"]
    assert len(invite) == 1, f"expected 1 invite notification, got {len(invite)}"
```

- [ ] **Step 2: Run, verify both fail**

Run: `pytest tests/integration/test_application_gate.py -v -k approve`

Expected: `test_approve_sets_stage_and_notifies` FAILS (stage becomes `"screening"`, not `"approved"`). `test_approve_is_idempotent` FAILS (a duplicate notification is created).

- [ ] **Step 3: Update `invite_interview` to land `approved` and dedupe**

Edit `agents/recruit/api.py` (the `invite_interview` route, lines 348-392). Replace its body:

```python
    @router.post("/applications/{app_id}/invite-interview")
    async def invite_interview(app_id: str):
        app_row = await asyncio.to_thread(
            recruit_db.get_application, agent.conn, app_id
        )
        if app_row is None:
            raise HTTPException(404, f"application not found: {app_id}")
        job = await asyncio.to_thread(
            recruit_db.get_job, agent.conn, app_row["job_id"]
        )
        job_title = (job or {}).get("title", "the role")

        title = f"You've been approved for an interview: {job_title}"
        body = (
            f"Hi {app_row.get('candidate_name') or 'there'} — a recruiter has "
            f"approved your application for {job_title}. Click the link to "
            f"start the AI interview."
        )
        link = f"/c/interview/{app_id}"

        # Idempotency: if the application is already approved, just return the
        # existing row + most recent invite notification without dispatching a
        # second one. This guards against double-clicks and accidental retries.
        if app_row.get("stage") == "approved":
            existing = await asyncio.to_thread(
                recruit_db.list_notifications,
                agent.conn,
                "candidate",
                app_row.get("candidate_email"),
                False,
            )
            existing_invite = next(
                (n for n in existing
                 if n.get("kind") == "interview_invite" and n.get("link") == link),
                None,
            )
            return {
                "ok": True,
                "application_id": app_id,
                "notification": existing_invite,
                "invite": {"title": title, "body": body, "link": link},
            }

        notif = await asyncio.to_thread(
            recruit_db.create_notification,
            agent.conn,
            {
                "user_role": "candidate",
                "user_id": app_row.get("candidate_email"),
                "kind": "interview_invite",
                "title": title,
                "body": body,
                "link": link,
            },
        )
        await asyncio.to_thread(
            recruit_db.update_application,
            agent.conn,
            app_id,
            {"stage": "approved"},
        )
        return {
            "ok": True,
            "application_id": app_id,
            "notification": notif,
            "invite": {"title": title, "body": body, "link": link},
        }
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `pytest tests/integration/test_application_gate.py -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add agents/recruit/api.py tests/integration/test_application_gate.py
git commit -m "feat(recruit): approve sets stage='approved' and is idempotent"
```

---

## Task 5: Reject creates a candidate notification

**Files:**
- Modify: `agents/recruit/api.py:331-346` (`update_application`)
- Modify: `tests/integration/test_application_gate.py`

**Goal:** A `PATCH /applications/{id} {stage:"rejected"}` triggers exactly one `application_rejected` notification on the first transition into `rejected`; subsequent reject patches are no-ops for notification.

- [ ] **Step 1: Add the failing tests**

Append to `tests/integration/test_application_gate.py`:

```python
def test_reject_creates_notification(client, applied_app):
    r = client.patch(
        f"/api/recruit/applications/{applied_app['id']}",
        json={"stage": "rejected"},
    )
    assert r.status_code == 200, r.text

    notifs = client.get(
        "/api/recruit/notifications",
        params={"user_role": "candidate", "user_id": "ada@example.com"},
    ).json()
    rejected = [n for n in notifs if n["kind"] == "application_rejected"]
    assert len(rejected) == 1
    assert rejected[0]["link"] == "/c/applications"


def test_reject_idempotent(client, applied_app):
    client.patch(f"/api/recruit/applications/{applied_app['id']}",
                 json={"stage": "rejected"})
    client.patch(f"/api/recruit/applications/{applied_app['id']}",
                 json={"stage": "rejected"})

    notifs = client.get(
        "/api/recruit/notifications",
        params={"user_role": "candidate", "user_id": "ada@example.com"},
    ).json()
    rejected = [n for n in notifs if n["kind"] == "application_rejected"]
    assert len(rejected) == 1
```

- [ ] **Step 2: Run, verify both fail**

Run: `pytest tests/integration/test_application_gate.py -v -k reject`

Expected: both fail — `update_application` does not currently dispatch a notification.

- [ ] **Step 3: Update the `update_application` route to dispatch on transition**

Edit `agents/recruit/api.py` lines 331-346. Replace the route handler body:

```python
    @router.patch("/applications/{app_id}")
    async def update_application(app_id: str, body: ApplicationPatch):
        if body.stage is not None and body.stage not in VALID_STAGES:
            raise HTTPException(
                400,
                f"invalid stage '{body.stage}'. Allowed: {sorted(VALID_STAGES)}",
            )

        # Capture the previous stage *before* the update so we can detect
        # transitions into 'rejected' and dispatch the candidate notification
        # exactly once (idempotent).
        prev = await asyncio.to_thread(
            recruit_db.get_application, agent.conn, app_id
        )
        if prev is None:
            raise HTTPException(404, f"application not found: {app_id}")

        row = await asyncio.to_thread(
            recruit_db.update_application,
            agent.conn,
            app_id,
            body.model_dump(exclude_unset=True),
        )
        if row is None:
            raise HTTPException(404, f"application not found: {app_id}")

        if (
            body.stage == "rejected"
            and prev.get("stage") != "rejected"
        ):
            job = await asyncio.to_thread(
                recruit_db.get_job, agent.conn, row["job_id"]
            )
            job_title = (job or {}).get("title", "the role")
            try:
                await asyncio.to_thread(
                    recruit_db.create_notification,
                    agent.conn,
                    {
                        "user_role": "candidate",
                        "user_id": row.get("candidate_email"),
                        "kind": "application_rejected",
                        "title": f"Update on your application for {job_title}",
                        "body": (
                            f"Hi {row.get('candidate_name') or 'there'} — your "
                            f"application for {job_title} has been closed. "
                            f"Thanks for your interest."
                        ),
                        "link": "/c/applications",
                    },
                )
            except Exception as e:
                log.warning("rejection notification failed: %s", e)

        return row
```

- [ ] **Step 4: Run tests, verify all pass**

Run: `pytest tests/integration/test_application_gate.py -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add agents/recruit/api.py tests/integration/test_application_gate.py
git commit -m "feat(recruit): notify candidate on first transition into 'rejected'"
```

---

## Task 6: Server-side interview-creation gate

**Files:**
- Modify: `agents/recruit/api.py:441-453` (`create_interview`)
- Modify: `tests/integration/test_application_gate.py`

**Goal:** `POST /api/recruit/interviews` returns 403 unless the application is in `"approved"` or `"interviewed"` stage. This is the security boundary; the frontend WaitingScreen is just UX.

- [ ] **Step 1: Add the failing tests**

Append to `tests/integration/test_application_gate.py`:

```python
def test_create_interview_refuses_when_applied(client, applied_app):
    r = client.post(
        "/api/recruit/interviews",
        json={"application_id": applied_app["id"]},
    )
    assert r.status_code == 403
    assert "approved" in r.text.lower() or "stage" in r.text.lower()


def test_create_interview_allowed_when_approved(client, applied_app):
    client.post(f"/api/recruit/applications/{applied_app['id']}/invite-interview")
    r = client.post(
        "/api/recruit/interviews",
        json={"application_id": applied_app["id"]},
    )
    assert r.status_code == 200, r.text


def test_create_interview_allowed_when_interviewed(client, applied_app):
    # Move directly to 'interviewed' (re-entry case).
    client.patch(
        f"/api/recruit/applications/{applied_app['id']}",
        json={"stage": "interviewed"},
    )
    r = client.post(
        "/api/recruit/interviews",
        json={"application_id": applied_app["id"]},
    )
    assert r.status_code == 200, r.text
```

- [ ] **Step 2: Run, verify they fail**

Run: `pytest tests/integration/test_application_gate.py -v -k create_interview`

Expected: `test_create_interview_refuses_when_applied` FAILS (currently 200). The other two PASS by accident (the endpoint is unconditional today).

- [ ] **Step 3: Add the gate to `create_interview`**

Edit `agents/recruit/api.py` lines 441-453. Replace the route body:

```python
    @router.post("/interviews")
    async def create_interview(body: InterviewIn):
        # Confirm application exists
        app_row = await asyncio.to_thread(
            recruit_db.get_application, agent.conn, body.application_id
        )
        if app_row is None:
            raise HTTPException(
                404, f"application not found: {body.application_id}"
            )
        # Gate: an interview can only be created once the recruiter has
        # approved the application (stage='approved') or after a prior
        # interview has already been run (stage='interviewed', re-entry).
        if app_row.get("stage") not in {"approved", "interviewed"}:
            raise HTTPException(
                403,
                f"application stage '{app_row.get('stage')}' is not eligible "
                f"for an interview — recruiter approval required.",
            )
        return await asyncio.to_thread(
            recruit_db.create_interview, agent.conn, body.model_dump()
        )
```

- [ ] **Step 4: Run tests, verify all pass**

Run: `pytest tests/integration/test_application_gate.py -v`

Expected: all 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add agents/recruit/api.py tests/integration/test_application_gate.py
git commit -m "feat(recruit): gate POST /interviews on stage in {approved, interviewed}"
```

---

## Task 7: Frontend stage union narrowing + expose cv_text

**Files:**
- Modify: `apps/hireflow/src/lib/api.ts:32-53` (`ApplicationStage` and `Application`)

**Goal:** Drop the removed stage values from the TypeScript union so `tsc` will surface every stale call site, and add `cv_text` to the `Application` type so the recruiter UI can read the parsed resume.

- [ ] **Step 1: Replace the `ApplicationStage` type**

Edit `apps/hireflow/src/lib/api.ts` lines 32-39:

```ts
export type ApplicationStage =
  | "applied"
  | "approved"
  | "interviewed"
  | "offer"
  | "hired"
  | "rejected";
```

(Removed: `"screened"`, `"interview"`. Added: `"approved"`.)

- [ ] **Step 2: Add `cv_text` to the `Application` type**

In the same file, edit the `Application` type (currently lines 41-53) to add `cv_text`:

```ts
export type Application = {
  id: string;
  job_id: string;
  candidate_name: string;
  candidate_email: string;
  cv_filename: string;
  cv_text?: string | null;
  fit_score: number; // 0-1
  matched_skills: string[];
  missing_skills: string[];
  stage: ApplicationStage;
  notes?: string | null;
  created_at: string;
};
```

(`cv_text` is already populated by `GET /api/recruit/applications/{id}` — see `_row_to_application` in `agents/recruit/db.py`. No backend change needed.)

- [ ] **Step 2: Try a TypeScript build to enumerate broken sites**

Run: `cd apps/hireflow && npx tsc --noEmit 2>&1 | head -40`

Expected: errors in `c.applications.tsx` and `app.applicants.tsx` — they reference the removed values. These will be fixed in Tasks 11 and 13. **Do not commit yet** — leaving TS broken between commits is fine because the next commit fixes the call sites.

---

## Task 8: `useApplication` polls while stage is `applied`

**Files:**
- Modify: `apps/hireflow/src/lib/queries.ts:115-121` (`useApplication`)

**Goal:** When the candidate is sitting on the WaitingScreen, the hook auto-refetches so the UI flips to the interview as soon as the recruiter approves — no manual reload.

- [ ] **Step 1: Replace `useApplication` with a polling-aware version**

Edit `apps/hireflow/src/lib/queries.ts` lines 115-121:

```ts
export function useApplication(id: string | undefined) {
  return useQuery<Application>({
    queryKey: qk.application(id ?? ""),
    queryFn: ({ signal }) => applications.get(id as string, { signal }),
    enabled: !!id,
    refetchOnWindowFocus: true,
    // Poll every 15s only while the candidate is waiting for review.
    // Once approved/rejected/interviewed/etc the result is terminal for the
    // gate and we stop polling to keep load near zero.
    refetchInterval: (query) => {
      const data = query.state.data;
      return data?.stage === "applied" ? 15_000 : false;
    },
  });
}
```

- [ ] **Step 2: Verify the file still type-checks**

Run: `cd apps/hireflow && npx tsc --noEmit 2>&1 | grep -E "queries\\.ts" || echo "no errors in queries.ts"`

Expected: `no errors in queries.ts`. (Other files may still have errors from Task 7 — that's fine.)

---

## Task 9: `WaitingScreen` component

**Files:**
- Create: `apps/hireflow/src/components/WaitingScreen.tsx`

**Goal:** A self-contained, stage-aware "you can't enter the interview yet" screen used by `c.interview.$id.tsx`.

- [ ] **Step 1: Create the component**

```tsx
import { Link } from "@tanstack/react-router";
import { motion } from "framer-motion";
import { Sparkles, Hourglass, XCircle } from "lucide-react";
import type { Application } from "@/lib/api";

type Props = {
  application: Application;
};

export function WaitingScreen({ application }: Props) {
  const { stage } = application;

  let Icon = Hourglass;
  let heading = "Application not available yet";
  let body =
    "We couldn't open the interview for this application. Head back to the dashboard to see what's next.";

  if (stage === "applied") {
    Icon = Hourglass;
    heading = "Your application is under review";
    body =
      "We've sent your application to the team. You'll get a notification as soon as a recruiter approves it — then this page will unlock automatically.";
  } else if (stage === "rejected") {
    Icon = XCircle;
    heading = "This application is closed";
    body =
      "Thanks for your interest. The team isn't moving forward with this application. You can browse other open roles whenever you're ready.";
  }

  return (
    <div className="fixed inset-0 z-50 flex flex-col bg-ink text-white">
      <div className="absolute inset-0 bg-violet-grad opacity-20" />
      <header className="relative flex items-center justify-between border-b border-white/10 px-6 py-4">
        <div className="flex items-center gap-2">
          <div className="grid h-7 w-7 place-items-center rounded-lg bg-white/10">
            <Sparkles className="h-3.5 w-3.5" />
          </div>
          <div>
            <div className="text-xs text-white/60">AI Interview Room</div>
            <div className="text-sm font-semibold">Awaiting recruiter</div>
          </div>
        </div>
        <Link to="/c" className="text-xs text-white/60 hover:text-white">
          Exit
        </Link>
      </header>
      <div className="relative flex flex-1 items-center justify-center px-6">
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className="max-w-md rounded-2xl border border-white/10 bg-white/5 p-8 text-center backdrop-blur"
        >
          <div className="mx-auto grid h-12 w-12 place-items-center rounded-2xl bg-white/10">
            <Icon className="h-6 w-6" />
          </div>
          <h2 className="mt-4 font-display text-2xl">{heading}</h2>
          <p className="mt-2 text-sm text-white/70">{body}</p>
          <Link
            to="/c/applications"
            className="mt-6 inline-flex items-center justify-center rounded-full bg-white px-4 py-2 text-sm font-semibold text-ink"
          >
            Track my application
          </Link>
        </motion.div>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Verify it type-checks**

Run: `cd apps/hireflow && npx tsc --noEmit 2>&1 | grep -E "WaitingScreen" || echo "no WaitingScreen errors"`

Expected: `no WaitingScreen errors`.

---

## Task 10: Wire `WaitingScreen` into the interview route

**Files:**
- Modify: `apps/hireflow/src/routes/c.interview.$id.tsx:43-54` (top of `Interview()`)

**Goal:** Before the interview engine boots, check application stage and render `<WaitingScreen/>` if the candidate isn't cleared.

- [ ] **Step 1: Add the gate at the top of `Interview()`**

Edit `apps/hireflow/src/routes/c.interview.$id.tsx`. Right after the imports, add:

```tsx
import { WaitingScreen } from "@/components/WaitingScreen";
```

Then in `Interview()` (currently around line 43), after the existing data hooks, add the gate. Replace lines 43-54 (the start of the function up to and including the cleanup `useEffect`) with:

```tsx
function Interview() {
  const { id } = Route.useParams();
  const { data: application, isLoading: appLoading } = useApplication(id);
  const { data: job } = useJob(application?.job_id);

  const [phase, setPhase] = useState<Phase>("preflight");
  const streamRef = useRef<MediaStream | null>(null);

  useEffect(() => () => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
  }, []);

  if (appLoading || !application) {
    return (
      <div className="fixed inset-0 z-50 grid place-items-center bg-ink text-white">
        <Loader2 className="h-6 w-6 animate-spin opacity-60" />
      </div>
    );
  }
  if (application.stage !== "approved" && application.stage !== "interviewed") {
    return <WaitingScreen application={application} />;
  }
```

(Note: `Loader2` is already imported on line 4 of the file.)

- [ ] **Step 2: Verify the file type-checks**

Run: `cd apps/hireflow && npx tsc --noEmit 2>&1 | grep -E "c\\.interview" || echo "no c.interview errors"`

Expected: `no c.interview errors`.

---

## Task 11: Drop the candidate's "Continue to interview" CTA

**Files:**
- Modify: `apps/hireflow/src/routes/c.apply.$id.tsx:182-251` (the `result` stage panel)

**Goal:** Replace the immediate-interview button with a "Track my application" link and refresh the next-step copy.

- [ ] **Step 1: Replace the result panel**

Edit `apps/hireflow/src/routes/c.apply.$id.tsx`. Replace the entire `{stage === "result" && result && ( ... )}` block (currently lines 182-251) with:

```tsx
        {stage === "result" && result && (
          <motion.div
            key="res"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-4 rounded-2xl border border-border bg-card p-6"
          >
            <div className="grid h-12 w-12 place-items-center rounded-2xl bg-success/20 text-success-foreground">
              <Check className="h-6 w-6" />
            </div>
            <h3 className="font-display text-2xl">
              You're a strong match — {Math.round(result.fit_score * 100)}%
            </h3>
            {result.matched_skills?.length ? (
              <p className="text-pretty text-muted-foreground">
                You match on{" "}
                {result.matched_skills.slice(0, 3).map((s, i, arr) => (
                  <span key={s}>
                    <strong>{s}</strong>
                    {i < arr.length - 1 ? (i === arr.length - 2 ? ", and " : ", ") : ""}
                  </span>
                ))}
                .
                {result.missing_skills?.length
                  ? ` If you're invited to interview, the team may ask about ${result.missing_skills.slice(0, 2).join(" and ")}.`
                  : ""}
              </p>
            ) : (
              <p className="text-muted-foreground">Application submitted. The hiring team will review shortly.</p>
            )}
            <div className="grid gap-3 sm:grid-cols-2">
              <div className="rounded-xl border border-border p-3">
                <div className="text-xs uppercase tracking-widest text-muted-foreground">Matched</div>
                <div className="mt-2 flex flex-wrap gap-1.5">
                  {(result.matched_skills ?? []).map((s) => (
                    <span key={s} className="rounded-full bg-success/20 px-2 py-0.5 text-xs">
                      {s}
                    </span>
                  ))}
                </div>
              </div>
              <div className="rounded-xl border border-border p-3">
                <div className="text-xs uppercase tracking-widest text-muted-foreground">To strengthen</div>
                <div className="mt-2 flex flex-wrap gap-1.5">
                  {(result.missing_skills ?? []).length ? (
                    (result.missing_skills ?? []).map((s) => (
                      <span key={s} className="rounded-full bg-destructive/15 px-2 py-0.5 text-xs">
                        {s}
                      </span>
                    ))
                  ) : (
                    <span className="text-xs text-muted-foreground">Nothing critical</span>
                  )}
                </div>
              </div>
            </div>
            <div className="rounded-xl border border-border bg-background p-4">
              <div className="text-xs uppercase tracking-widest text-muted-foreground">What happens next</div>
              <div className="mt-1 font-medium">A recruiter will review your application</div>
              <div className="text-sm text-muted-foreground">
                You'll get a notification here as soon as they approve you for the AI interview.
              </div>
            </div>
            <button
              onClick={() => nav({ to: "/c/applications" })}
              className="inline-flex w-full items-center justify-center gap-1.5 rounded-full bg-violet-grad py-2.5 text-sm font-semibold text-accent-foreground shadow-glow"
            >
              Track my application <ArrowRight className="h-4 w-4" />
            </button>
          </motion.div>
        )}
```

- [ ] **Step 2: Verify the file type-checks**

Run: `cd apps/hireflow && npx tsc --noEmit 2>&1 | grep -E "c\\.apply" || echo "no c.apply errors"`

Expected: `no c.apply errors`.

---

## Task 12: Update the candidate applications list

**Files:**
- Modify: `apps/hireflow/src/routes/c.applications.tsx:14-33` (`stageLabel`)
- Modify: `apps/hireflow/src/routes/c.applications.tsx:104-114` (the inline Start CTA condition)

**Goal:** Map `"approved"` to "Interview ready" (drop the removed `"screened"` and `"interview"` cases) and only render the Start CTA when stage is exactly `"approved"`.

- [ ] **Step 1: Replace `stageLabel`**

Edit `apps/hireflow/src/routes/c.applications.tsx` lines 14-33:

```tsx
function stageLabel(s: Application["stage"]): string {
  switch (s) {
    case "applied":
      return "Submitted";
    case "approved":
      return "Interview ready";
    case "interviewed":
      return "Interview done";
    case "offer":
      return "Offer";
    case "hired":
      return "Hired";
    case "rejected":
      return "Closed";
    default:
      return s;
  }
}
```

- [ ] **Step 2: Tighten the inline Start CTA**

In the same file, replace the existing block:

```tsx
                {(a.stage === "interview" || a.stage === "applied") && (
                  <Link
                    to="/c/interview/$id"
                    params={{ id: a.id }}
                    className="inline-flex items-center gap-1 rounded-full bg-foreground px-3 py-1.5 text-xs font-semibold text-background"
                  >
                    Start <ArrowRight className="h-3.5 w-3.5" />
                  </Link>
                )}
```

with:

```tsx
                {a.stage === "approved" && (
                  <Link
                    to="/c/interview/$id"
                    params={{ id: a.id }}
                    className="inline-flex items-center gap-1 rounded-full bg-foreground px-3 py-1.5 text-xs font-semibold text-background"
                  >
                    Start <ArrowRight className="h-3.5 w-3.5" />
                  </Link>
                )}
```

- [ ] **Step 3: Verify the file type-checks**

Run: `cd apps/hireflow && npx tsc --noEmit 2>&1 | grep -E "c\\.applications" || echo "no c.applications errors"`

Expected: `no c.applications errors`.

---

## Task 13: Single "Approve for interview" button on the applicants drawer

**Files:**
- Modify: `apps/hireflow/src/routes/app.applicants.tsx:309-438` (the `CandidatePanel` component)

**Goal:** Replace the two interview buttons with one "Approve for interview" CTA backed by `useInviteInterview`. Drop the now-unused `moveTo("interview")` path.

- [ ] **Step 1: Trim imports**

Edit `apps/hireflow/src/routes/app.applicants.tsx` lines 16-22:

```tsx
import {
  useApplications,
  useInviteInterview,
  useJobs,
  useUpdateApplication,
} from "@/lib/queries";
```

Stays the same — both hooks are still used (`useUpdateApplication` powers the Reject button).

- [ ] **Step 2: Replace the action footer in `CandidatePanel`**

Replace the trailing action `div` of `CandidatePanel` (currently lines 413-436):

```tsx
      <div className="flex flex-wrap gap-2 border-t border-border pt-4">
        <button
          disabled={update.isPending}
          onClick={() => moveTo("interview")}
          className="inline-flex items-center gap-2 rounded-full bg-violet-grad px-4 py-2 text-sm font-semibold text-accent-foreground shadow-glow disabled:opacity-60"
        >
          {update.isPending && <Loader2 className="h-4 w-4 animate-spin" />}
          Move to interview
        </button>
        <button
          disabled={invite.isPending}
          onClick={sendInvite}
          className="inline-flex items-center gap-1.5 rounded-full border border-border px-4 py-2 text-sm font-semibold disabled:opacity-60"
        >
          <Mail className="h-4 w-4" /> Invite to interview
        </button>
        <button
          disabled={update.isPending}
          onClick={() => moveTo("rejected")}
          className="rounded-full px-4 py-2 text-sm text-muted-foreground hover:text-destructive disabled:opacity-60"
        >
          Reject
        </button>
      </div>
```

with:

```tsx
      <div className="flex flex-wrap gap-2 border-t border-border pt-4">
        <button
          disabled={invite.isPending || c.stage === "approved" || c.stage === "interviewed"}
          onClick={sendInvite}
          className="inline-flex items-center gap-2 rounded-full bg-violet-grad px-4 py-2 text-sm font-semibold text-accent-foreground shadow-glow disabled:opacity-60"
        >
          {invite.isPending && <Loader2 className="h-4 w-4 animate-spin" />}
          {c.stage === "approved" || c.stage === "interviewed"
            ? "Approved for interview"
            : "Approve for interview"}
        </button>
        <button
          disabled={update.isPending || c.stage === "rejected"}
          onClick={() => moveTo("rejected")}
          className="rounded-full px-4 py-2 text-sm text-muted-foreground hover:text-destructive disabled:opacity-60"
        >
          Reject
        </button>
      </div>
```

- [ ] **Step 3: Drop the unused `Mail` import**

In the same file's import block (lines 5-15), remove `Mail` from the lucide-react imports:

```tsx
import {
  Filter,
  GitCompare,
  X,
  ChevronRight,
  MapPin,
  Briefcase,
  Calendar,
  Loader2,
} from "lucide-react";
```

- [ ] **Step 4: Update `stageToChip` to map `approved`**

In the same file, replace `stageToChip` (lines 36-53):

```tsx
function stageToChip(s: Application["stage"]): ChipStage {
  switch (s) {
    case "applied":
      return "New";
    case "approved":
    case "interviewed":
      return "Interview";
    case "offer":
    case "hired":
      return "Offer";
    case "rejected":
      return "Rejected";
    default:
      return "New";
  }
}
```

(Removed: `"screened"` → `"Screen"` case, `"interview"` case. Replaced both interview-pipeline cases with `"approved" | "interviewed" → "Interview"`.)

- [ ] **Step 5: Update the stage filter list**

Same file, replace the dropdown options (currently around line 143):

```tsx
          {["all", "New", "Interview", "Offer", "Rejected"].map((s) => (
            <option key={s} value={s}>
              {s === "all" ? "All stages" : s}
            </option>
          ))}
```

(Removed `"Screen"` since the chip no longer exists.)

- [ ] **Step 6: Verify the whole frontend type-checks**

Run: `cd apps/hireflow && npx tsc --noEmit`

Expected: zero errors.

- [ ] **Step 7: Commit all frontend changes together**

```bash
git add apps/hireflow/src/lib/api.ts apps/hireflow/src/lib/queries.ts \
        apps/hireflow/src/components/WaitingScreen.tsx \
        apps/hireflow/src/routes/c.interview.\$id.tsx \
        apps/hireflow/src/routes/c.apply.\$id.tsx \
        apps/hireflow/src/routes/c.applications.tsx \
        apps/hireflow/src/routes/app.applicants.tsx
git commit -m "feat(hireflow): require recruiter approval before AI interview"
```

---

## Task 14: Recruiter can read the parsed CV before approving

**Files:**
- Modify: `apps/hireflow/src/routes/app.applicants.tsx:370-373` (the CV panel inside `CandidatePanel`)

**Goal:** Replace the filename-only CV panel with a disclosure that lets the recruiter read the parsed resume text inline before clicking Approve. Long CVs stay collapsed by default to keep the drawer scannable.

- [ ] **Step 1: Replace the CV panel block**

Edit `apps/hireflow/src/routes/app.applicants.tsx`. Find this block (currently lines 370-373):

```tsx
      <div className="rounded-2xl border border-border bg-background p-4">
        <div className="text-xs uppercase tracking-widest text-muted-foreground">CV</div>
        <p className="mt-2 text-sm">{c.cv_filename || "—"}</p>
      </div>
```

Replace it with:

```tsx
      <div className="rounded-2xl border border-border bg-background p-4">
        <div className="flex items-center justify-between gap-2">
          <div className="text-xs uppercase tracking-widest text-muted-foreground">CV</div>
          <span className="truncate text-xs text-muted-foreground">
            {c.cv_filename || "—"}
          </span>
        </div>
        {c.cv_text ? (
          <details className="mt-3 group">
            <summary className="cursor-pointer text-sm text-accent hover:underline">
              <span className="group-open:hidden">Read full CV</span>
              <span className="hidden group-open:inline">Hide CV</span>
            </summary>
            <pre className="mt-3 max-h-96 overflow-auto whitespace-pre-wrap rounded-lg border border-border bg-muted/30 p-3 font-mono text-xs leading-relaxed">
              {c.cv_text}
            </pre>
          </details>
        ) : (
          <p className="mt-2 text-xs italic text-muted-foreground">
            No parsed text available for this CV.
          </p>
        )}
      </div>
```

- [ ] **Step 2: Verify the file type-checks**

Run: `cd apps/hireflow && npx tsc --noEmit 2>&1 | grep -E "app\\.applicants" || echo "no app.applicants errors"`

Expected: `no app.applicants errors`.

- [ ] **Step 3: Commit**

```bash
git add apps/hireflow/src/routes/app.applicants.tsx
git commit -m "feat(hireflow): let recruiter read parsed CV before approving"
```

---

## Task 15: Manual smoke test

**Files:** none (verification only).

**Goal:** Walk through each user journey end-to-end with the live unified server and confirm the expected behavior.

- [ ] **Step 1: Restart the unified server**

```bash
pkill -9 -f "EngineCore" ; pkill -f "uvicorn.*main:app" ; sleep 2
cd /teamspace/studios/this_studio/Agentic-Ai-Recruiter/ai-recruiter
uvicorn main:app --host 0.0.0.0 --port 8000 &
```

Confirm the boot log shows `recruit.startup ok` (and any legacy stage rows in the dev DB are silently migrated to `approved`).

- [ ] **Step 2: Build (or run) the frontend**

If serving prod bundle:

```bash
cd apps/hireflow && npm run build
```

Otherwise run `npm run dev` and use that URL for the next steps.

- [ ] **Step 3: Run the smoke checklist (each must succeed)**

1. Apply with a fresh email → result panel shows score + skills + "Track my application" button (no interview link).
2. Click "Track my application" → `/c/applications` shows row at stage "Submitted", **no** "Start" button.
3. Open `/c/interview/{appId}` directly → WaitingScreen with "under review" copy.
4. As recruiter on `/app/applicants` → drawer shows **one** button labeled "Approve for interview" + "Reject". The CV panel exposes a "Read full CV" disclosure that expands to a scrollable monospace block containing the parsed resume text. After reading, click "Approve for interview".
5. Back as candidate: bell shows unread badge; `/c/applications` flips to "Interview ready" with "Start" button; the WaitingScreen page auto-refreshes to interview UI within ~15s without manual reload.
6. Reject path: from a fresh app, recruiter clicks "Reject" → candidate's bell shows the rejection notification; `/c/interview/{appId}` shows "closed" copy.
7. Direct attempt to start an interview while applied (e.g. via curl): `POST /api/recruit/interviews {"application_id":"..."}` returns 403 with the gate error.

- [ ] **Step 4: If any step fails, debug and re-run; do not declare success without all 7 ticks.**
