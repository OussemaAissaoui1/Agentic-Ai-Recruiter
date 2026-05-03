---
date: 2026-05-02
status: approved
authors: [oussa612]
---

# Candidate Approval Gate

## Problem

When a candidate applies through `/c/apply/{jobId}`, today's flow shows their fit score and immediately presents a "Continue to interview" button that navigates to `/c/interview/{appId}`. There is no recruiter in the loop and no server-side check before the candidate starts a live AI interview. The recruiter UI also exposes two redundant buttons ("Move to interview", "Invite to interview") that both route candidates toward the interview pipeline with subtly different effects, and the backend's `VALID_STAGES` set is out of sync with the stage values the frontend uses.

We want the candidate to see their match score after applying, but to be blocked from interviewing until the recruiter explicitly approves. Approval triggers an in-app notification that links the candidate back to the interview. Rejection also triggers a notification.

## Decisions (locked in)

1. **Approval is always manual.** No fit-score auto-approve threshold in this iteration; the hook for one is left for later.
2. **Recruiter side: a single "Approve for interview" CTA** replaces the two existing interview buttons. "Reject" stays.
3. **Candidate side: keep the score and skill breakdown**, drop the interview CTA, replace it with a "Track my application" button linking to `/c/applications`.
4. **Friendly waiting screen** when an unapproved candidate hits `/c/interview/{id}`. Stage-aware copy (under-review vs. closed). Server-side gate enforces the same rule defense-in-depth.
5. **Notify on rejection.** Symmetric with approval; both sides of the recruiter decision produce a candidate notification.
6. **Stage is the single source of truth.** New stage value `"approved"` between `applied` and `interviewed`. No new tables, no new columns, no new HTTP endpoints.

## Architecture

```
┌─ candidate apply ──────────────────────────────┐
│ POST /api/recruit/applications                  │ stage = "applied"
│ → recruiter "new_application" notification      │ (no change)
└─────────────────────────────────────────────────┘
                       │
                       ▼  shows score + skills + "Track my application"
┌─ recruiter approve ────────────────────────────┐
│ POST /api/recruit/applications/{id}/invite-     │ stage = "approved"
│ interview (existing endpoint, tweaked)          │ + candidate "interview_invite" notif
└─────────────────────────────────────────────────┘
                       │
                       ▼  notification bell → /c/interview/{id}
┌─ interview gate ───────────────────────────────┐
│ Frontend: GET /api/recruit/applications/{id}    │ stage ∈ {approved, interviewed}?
│   → render interview UI                         │   yes → live interview
│   → otherwise render <WaitingScreen/>           │   no  → "under review" / "closed" copy
│ Backend: POST /api/recruit/interviews enforces  │ stage check (defense in depth)
│ the same gate before creating an interview row  │
└─────────────────────────────────────────────────┘
                       │
                       ▼ (parallel: reject path)
┌─ recruiter reject ─────────────────────────────┐
│ PATCH /api/recruit/applications/{id}            │ stage = "rejected"
│ {stage: "rejected"} (existing)                  │ + candidate "application_rejected" notif
└─────────────────────────────────────────────────┘
```

**Principles**

- Stage transitions: `applied → approved → interviewed`. Terminals: `offer / hired / rejected`. Re-approval after reject is out of scope.
- The gate is enforced server-side at `POST /api/recruit/interviews`. The frontend WaitingScreen is UX, never the security boundary.
- All change is in `agents/recruit/` (Python) and `apps/hireflow/src/` (TypeScript). No changes to NLP, voice, vision, matching, or scoring agents. No changes to vLLM, Kokoro, or model configs.
- Reuse existing endpoints (`invite-interview`, `PATCH /applications/{id}`, `POST /interviews`). One new responsibility on each.

## Components & contracts

### Backend (`agents/recruit/`)

| File | Change |
|---|---|
| `api.py` `VALID_STAGES` | New set: `{"applied", "approved", "interviewed", "offer", "hired", "rejected"}`. `"screening"` removed. |
| `api.py` `invite_interview()` | Set `stage="approved"` (was `"screening"`). Notification body updated to mention approval. Idempotent: if the row is already `"approved"`, return existing row without dispatching a duplicate notification. |
| `api.py` `update_application()` | When `body.stage == "rejected"` and the previous stage was not `"rejected"`, create a candidate notification (`kind="application_rejected"`, `link="/c/applications"`). Idempotent: re-patching `rejected → rejected` does not duplicate. |
| `api.py` `create_interview()` | New gate: load the application; refuse with 403 if its stage is not in `{"approved", "interviewed"}`. Existing 200 path is preserved otherwise. |
| `db.py` `init_db()` | One-shot data migration on every boot: `UPDATE applications SET stage='approved' WHERE stage IN ('screening', 'interview')`. Idempotent. Lives next to `seed_if_empty`. |

No SQL schema migration. The `stage` column already accepts arbitrary strings; the constraint lives in `VALID_STAGES`.

### Frontend (`apps/hireflow/src/`)

| File | Change |
|---|---|
| `routes/c.apply.$id.tsx` | Drop the "Continue to interview" button. Replace the "Next step" card with a "What happens next" card: "We've sent your application to the team. You'll get a notification as soon as a recruiter reviews it." Bottom CTA: "Track my application" → `/c/applications`. |
| `routes/c.applications.tsx` | `stageLabel`: replace `"interview"` case with `"approved"` rendering as "Interview ready". The inline "Start" CTA on each card renders only when `stage === "approved"` (drop the current `\|\| "applied"` fallback). |
| `routes/c.interview.$id.tsx` | At the top of the route component, fetch the application via `useApplication(id)`; if `stage ∉ {"approved", "interviewed"}`, render `<WaitingScreen application={app} />` and skip the interview engine entirely. |
| `components/WaitingScreen.tsx` (new) | Stage-aware copy. `applied` → "Your application is under review. We'll notify you as soon as a recruiter approves it." + link to `/c/applications`. `rejected` → "This application has been closed." + link. Anything else → fallback "Not available yet." |
| `routes/app.applicants.tsx` `CandidatePanel` | Collapse "Move to interview" + "Invite to interview" into a single "Approve for interview" button that calls `useInviteInterview` (already wired). "Reject" stays. Drop the now-orphan `useUpdateApplication` call for the interview-stage transition. **CV panel** also gets a "Read full CV" disclosure that renders the parsed `cv_text` in a scrollable monospace block, so the recruiter can read the resume before deciding to approve. |
| `lib/api.ts` `Application["stage"]` | New union: `"applied" \| "approved" \| "interviewed" \| "offer" \| "hired" \| "rejected"`. `"screened"` and `"interview"` removed. TypeScript will surface every site that referenced the removed values. |
| `lib/api.ts` `Application` | Add `cv_text?: string \| null` so the recruiter UI can render the parsed CV body (already returned by `GET /api/recruit/applications/{id}`). |
| `lib/queries.ts` `useApplication(id)` | Refetch on window focus and a 15s polling interval **only** when stage is `"applied"`. Stops polling once stage flips to `"approved"` or `"rejected"`. Lets a candidate sitting on the WaitingScreen see the gate flip without manual reload. |

**HTTP contract surface:** zero new endpoints, zero new request shapes. Only response semantics change: `invite-interview` now lands `stage="approved"`, `PATCH /applications` may now create a notification side-effect, `POST /interviews` may now 403.

## Data flow

### Happy path

1. Candidate `POST /api/recruit/applications` (multipart) → row created with `stage="applied"`; recruiter notification dispatched. *(No change.)*
2. Apply page renders score + skills + "Track my application" → `/c/applications` shows the application at stage "Submitted".
3. Recruiter opens `/app/applicants` → drawer → "Approve for interview" → `POST /api/recruit/applications/{id}/invite-interview` → `stage="approved"` + candidate notification (`kind="interview_invite"`, `link="/c/interview/{id}"`).
4. Candidate's bell badge increments (Shell.tsx polls `useNotifications`). Click → navigate to `/c/interview/{id}`.
5. `c.interview.$id.tsx` fetches the application; stage is `"approved"` → renders interview UI; agents start as today.
6. Interview ends → `stage="interviewed"`. The route still renders the post-interview "Done" view.

### Reject path

1. Recruiter clicks "Reject" → `PATCH /api/recruit/applications/{id} {stage:"rejected"}`.
2. Backend detects the transition `<not rejected> → "rejected"` and creates a `kind="application_rejected"` candidate notification.
3. Candidate sees it in the bell + on `/c/applications` (stage label "Closed"). Direct nav to `/c/interview/{id}` shows WaitingScreen with rejected copy.

## Edge cases

| Edge case | Handling |
|---|---|
| Candidate sits on WaitingScreen while recruiter approves | `useApplication(id)` runs with `refetchOnWindowFocus: true` and a 15s polling interval **only** while stage is `"applied"`. Gate flips to interview UI without manual reload. Polling stops once approved or rejected. |
| Recruiter clicks Approve twice in quick succession | `invite_interview` is idempotent: if stage is already `"approved"`, skip the notification dispatch and return the existing row. |
| Reject after approve | Allowed. Stage moves to `"rejected"`; rejection notification fires. If the candidate is mid-interview, the existing interview row is not yanked mid-session — the gate is only on `POST /interviews`, not on every interview chunk. |
| Re-approve after reject | Out of scope. No UI affordance. Reachable later by a backend allowance + a recruiter button. |
| Direct URL on already-approved app | Renders interview UI normally. |
| Direct URL on `interviewed` (already done) stage | Renders the existing post-interview "Done" view, same as today. |
| Application doesn't exist (bad id) | Backend 404 → frontend shows "Application not found". *(Existing behavior.)* |
| Backend 403 on `POST /interviews` because frontend has stale state | Frontend catches, shows toast "Application is no longer eligible for interview", redirects to `/c/applications`. |
| Matching agent fails / `fit_score` is null | Apply page still completes — score shows 0, flow proceeds. Recruiter still gets notified. *(Existing behavior preserved.)* |
| TypeScript narrowing breaks code that referenced removed stages | Intentional. `tsc` is the forcing function for the cleanup. Fewer than five sites currently reference `"screened"` or `"interview"`. |

## Testing

### Backend (new file `tests/integration/test_application_gate.py`)

| Test | Asserts |
|---|---|
| `test_apply_starts_in_applied_stage` | `POST /api/recruit/applications` returns row with `stage="applied"`. |
| `test_approve_sets_stage_and_notifies` | After `POST /applications/{id}/invite-interview`: stage is `"approved"`; one candidate notification of kind `"interview_invite"` with link `/c/interview/{id}` exists. |
| `test_approve_is_idempotent` | Calling `/invite-interview` twice does not create a second notification. |
| `test_reject_creates_notification` | `PATCH /applications/{id} {stage:"rejected"}` creates one candidate notification of kind `"application_rejected"`. |
| `test_reject_idempotent` | Patching to `rejected` again does not duplicate the notification. |
| `test_create_interview_refuses_when_applied` | `POST /interviews {application_id}` returns 403 while stage is `"applied"`. |
| `test_create_interview_allowed_when_approved` | Same call succeeds with stage `"approved"`. |
| `test_create_interview_allowed_when_interviewed` | Re-entry case (already done) returns 200. |
| `test_invalid_stage_rejected` | `PATCH /applications/{id} {stage:"screening"}` returns 400. |

Run against the FastAPI app via `TestClient` with a per-test `:memory:` SQLite database (or a `tmp_path` fixture) so the dev DB is untouched.

### Frontend manual smoke checklist

1. Apply with a fresh email → result panel shows score + skills + "Track my application" button (no interview link).
2. Click "Track my application" → `/c/applications` shows row at stage "Submitted", no "Start" button.
3. Open `/c/interview/{appId}` directly in same tab → WaitingScreen with "under review" copy.
4. As recruiter: open `/app/applicants` drawer → see one button labeled "Approve for interview" (not two), click it.
5. Back as candidate: bell shows unread badge; `/c/applications` flips to "Interview ready" with "Start" button; the WaitingScreen page auto-refreshes to interview UI within 15s without manual reload.
6. Reject path: from a fresh app, recruiter clicks Reject → candidate's bell shows the rejection notification; `/c/interview/{appId}` shows "closed" copy.
7. TypeScript: `npm run build` (or `tsc --noEmit`) compiles cleanly — confirms the stage-type cleanup did not leave stragglers.

E2E / Playwright tests are out of scope for this iteration.

## Rollout & migration

**Schema:** none.

**Data migration:** the dev DB likely has rows with stages we removed (`"screening"`, `"interview"`) from earlier recruiter actions. One-shot rewrite at startup in `db.py:init_db()`:

```sql
UPDATE applications SET stage = 'approved' WHERE stage IN ('screening', 'interview');
```

Both old values were effectively "recruiter green-lit this candidate". Mapping both to `"approved"` preserves the semantic for in-flight candidates. The migration is idempotent — safe to run on every boot.

**Deployment steps:**

1. Land the backend changes + restart uvicorn → migration runs, stage values normalize, gate is live.
2. `cd apps/hireflow && npm run build` (or run `npm run dev`) → frontend picks up the new types/routes/components.
3. No vLLM, Kokoro, or matching agent restart needed — none of those code paths change.

**Rollback:** revert the commits + restart. To fully revert data, run `UPDATE applications SET stage='interview' WHERE stage='approved';` once. No data is lost.

**Risk surface**

- The interview-creation gate is the new auth boundary. A bug there could let an unapproved candidate start an interview. Mitigated by `test_create_interview_refuses_when_applied`.
- WaitingScreen polling at 15s while stage is `"applied"` is a small load increase per open browser tab. Negligible at the current scale.

## Out of scope

- Email notifications (in-app only, matching the existing notification system).
- Auto-approve threshold (Q1 option B). Hook left for later: a single check in `create_application` could auto-set `stage="approved"` and dispatch the notification when `fit_score ≥ X`.
- Re-approve after reject. Would be a one-line backend allowance + a recruiter button.
- E2E browser tests (Playwright). Manual smoke checklist for now.
