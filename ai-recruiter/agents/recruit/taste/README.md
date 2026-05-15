# Recruiter Taste — per-recruiter preference learning

A POC subsystem that learns what each recruiter weights from their
approve / reject decisions and surfaces a "recruiter fit" pill alongside
the AI's existing recommendation.

It lives inside the **recruit** agent (not as a standalone agent) because
it owns the same data — applications, decisions, the recruit DB.

## What it does, in one paragraph

When a recruiter clicks **approve** (invite-to-interview) or **reject**
on an application, a snapshot of that candidate's feature vector is
stored alongside the decision. After every 5 decisions, an L2-regularised
logistic regression is refit from scratch on the recruiter's full
history. The fitted weights become a "taste profile." When the recruiter
later views any candidate, the same feature vector is run through that
recruiter's profile and a `score ∈ [0, 1]` is shown as a separate pill —
visually distinct from the AI's recommendation chip — with the top
positive and negative contributing factors in plain language.

The pill is a **tiebreaker signal**, not a re-ranker. The AI's
recommendation is never modified, never overridden, never recomputed.

## Why it exists

Two recruiters scoring the same candidate often disagree, and that
disagreement is itself signal. One values raw technical depth; another
values communication clarity; a third tolerates missing skills if the
candidate seems engaged. A single global recommendation can't reflect
this. The POC lets the system learn the disagreement and surface it back
to each recruiter as their own personalised view, so the AI feels less
like a black-box gatekeeper and more like a colleague calibrated to
them.

## Architecture

```
                       ┌──────────────────────────────────────┐
                       │   HireFlow UI (apps/hireflow)        │
                       │                                      │
                       │   • Applicants list                  │
                       │   • Candidate panel                  │
                       │   • /app/applicants_/$id/report      │
                       │   • /app/settings  ← Your taste page │
                       └──────────────────────────────────────┘
                                 │ X-Recruiter-Id header
                                 ▼
   POST /api/recruit/apps/{id}/invite-interview ───┐
   PATCH /api/recruit/apps/{id}  (stage=rejected) ─┤   capture hook
                                                   │   (fail-soft)
                                                   ▼
                       ┌──────────────────────────────────────┐
                       │   agents/recruit/taste/capture.py    │
                       │     • fetch features (cache)         │
                       │     • write recruiter_decisions row  │
                       │     • if N % 5 == 0 → schedule refit │
                       └──────────────────────────────────────┘
                                 │
                ┌────────────────┼─────────────────────────┐
                ▼                ▼                         ▼
    ┌──────────────────┐  ┌──────────────────┐  ┌────────────────────┐
    │  features.py     │  │  learner.py      │  │  recruit.db        │
    │  cached 10-dim   │  │  sklearn LR(L2)  │  │  3 new tables:     │
    │  per (app, ver)  │  │  refit-from-     │  │  • candidate_      │
    │                  │  │  scratch         │  │    features        │
    │  pulls from:     │  │                  │  │  • recruiter_      │
    │  • applications  │  │  produces:       │  │    decisions       │
    │  • interviews.   │  │  • weights       │  │  • recruiter_      │
    │    behavioral    │  │  • intercept     │  │    profiles        │
    │  • scoring/      │  │  • means/stds    │  └────────────────────┘
    │    {iid}_report  │  │  • confidence    │
    └──────────────────┘  └──────────────────┘
                                 ▲
                                 │
   GET /api/recruit/taste/me/profile  ────────────┐
   GET /api/recruit/taste/score/{app_id} ─────────┤   read paths
   POST /api/recruit/taste/me/refit  ─────────────┤
   POST /api/recruit/taste/me/reset  ─────────────┘
```

## Data flow at a glance

**Decision capture**

1. Recruiter clicks approve or reject in the HireFlow UI.
2. The existing recruit-API transition runs first; the `recruiter_taste`
   capture hook is wrapped in `try/except` and runs after — a failure
   here is logged but never blocks the transition.
3. Hook reads (or computes-and-caches) the candidate's feature vector,
   writes a row to `recruiter_decisions` with both the
   `features_snapshot_json` and `features_version`.
4. If this is the recruiter's `5th, 10th, 15th, …` decision, schedule
   a background refit via `asyncio.to_thread` — fire-and-forget.

**Refit**

1. Pull all `recruiter_decisions` for `recruiter_id`.
2. Z-score features using the training-set mean/std.
3. Fit `sklearn.linear_model.LogisticRegression(penalty="l2", C=1.0)`.
4. Persist `weights`, `intercept`, `means`, `stds`, `n_decisions`,
   `confidence`, `version` into `recruiter_profiles`.

**Score-on-read**

1. UI requests `GET /api/recruit/taste/score/{application_id}`.
2. Server fetches features (cache), loads recruiter's profile, computes
   `sigmoid(W · z(features) + intercept)`, and returns the score plus
   the top 3 positive and top 2 negative per-feature contributions in
   plain-text labels.

## Database schema

Three tables appended to `agents/recruit/db.py`'s `SCHEMA_SQL`. All
idempotent — `init_db()` keeps existing rows; first call after deploy
just creates the tables.

```sql
candidate_features (
  application_id    TEXT NOT NULL,
  features_version  INTEGER NOT NULL,
  features_json     TEXT NOT NULL,
  computed_at       REAL NOT NULL,
  PRIMARY KEY (application_id, features_version)
);

recruiter_decisions (
  id                      TEXT PRIMARY KEY,
  recruiter_id            TEXT NOT NULL,        -- free-text; no users table yet
  application_id          TEXT NOT NULL,
  features_snapshot_json  TEXT NOT NULL,
  features_version        INTEGER NOT NULL,
  decision                TEXT NOT NULL CHECK (decision IN ('approved','rejected')),
  dwell_seconds           REAL,
  notes_text              TEXT,
  created_at              REAL NOT NULL
);

recruiter_profiles (
  recruiter_id      TEXT PRIMARY KEY,
  weights_json      TEXT NOT NULL,              -- {weights, intercept, means, stds}
  n_decisions       INTEGER NOT NULL DEFAULT 0,
  confidence_level  TEXT NOT NULL CHECK (confidence_level IN ('cold','warming','warm')),
  version           INTEGER NOT NULL DEFAULT 1,
  updated_at        REAL NOT NULL
);
```

`recruiter_id` is `TEXT` (no FK) because this codebase has no users
table or auth; the API reads it from the `X-Recruiter-Id` request
header and falls back to `"hr-default"`. Adding real auth later is a
no-schema-change upgrade — point the header at the authenticated user
and existing rows continue to work.

## Feature vector

10 features, `FEATURES_VERSION = 1`. Each decision row carries its own
snapshot at the version it was captured under, so future feature-set
changes don't invalidate history.

| name | source | range | sign |
|---|---|---|---|
| `fit_score` | `applications.fit_score` (matching agent) | 0–1 | + |
| `hard_skill_overlap` | `len(applications.matched_skills)` | 0–~15 | + |
| `missing_skills_count` | `len(applications.missing_skills)` | 0–~10 | − |
| `technical_avg` | `scoring report.overall.technical_avg` | 0–5 | + |
| `coherence_avg` | `scoring report.overall.coherence_avg` | 0–5 | + |
| `recommendation_tier` | scoring's `recommendation` (no_hire→0 … strong_hire→3) | 0–3 | + |
| `engagement_mean` | `interviews.behavioral_json` (vision) | 0–1 | + |
| `confidence_mean` | `interviews.behavioral_json` (vision) | 0–1 | + |
| `cognitive_load_mean` | `interviews.behavioral_json` (vision) | 0–1 | − |
| `emotional_arousal_mean` | `interviews.behavioral_json` (vision) | 0–1 | mixed |

Sign column is the *expected* direction of preference for a "typical"
recruiter; the learner determines actual direction per recruiter.

## API surface

All under `/api/recruit/taste/*`, all read `X-Recruiter-Id` (default
`hr-default`).

| Method | Path | Purpose |
|---|---|---|
| GET | `/me/profile` | Current recruiter's profile: top 5 positive + top 3 negative factors, `n_decisions`, `confidence`, `version`. |
| GET | `/score/{application_id}` | `{score, confidence, top_positive[3], top_negative[2]}`. Computes features on miss. |
| POST | `/me/refit` | Force a refit from the full decision history. Returns the updated profile. |
| POST | `/me/reset` | Drop the profile (decisions are kept). |

## Backend modules

```
agents/recruit/
├── db.py                              ← schema + init_db (added 3 tables)
├── api.py                             ← capture wired into transitions; +4 routes
└── taste/
    ├── __init__.py                    ← public re-exports
    ├── features.py                    ← 10-dim feature cache
    ├── learner.py                     ← sklearn LR fit + score
    ├── synthetic.py                   ← 4 ground-truth personas
    ├── capture.py                     ← decision-capture hook (fail-soft)
    └── cli.py                         ← seed / list / show / reset CLI
```

`Confidence` tiers from `learner.py`:
- `cold`     — `n_decisions < 10` (UI mutes the pill)
- `warming`  — `10 ≤ n < 50`
- `warm`     — `n ≥ 50`

## Frontend integration

```
apps/hireflow/src/
├── lib/
│   ├── api.ts                         ← `taste` namespace + types
│   └── queries.ts                     ← useRecruiterProfile, useRecruiterFitScore,
│                                        useRefitProfile, useResetProfile
├── components/
│   └── recruiter-taste/
│       └── RecruiterFitPill.tsx       ← the pill (tooltip with top factors)
└── routes/
    ├── app.applicants.tsx             ← pill in row + candidate panel
    ├── app.applicants_.$id.report.tsx ← pill in report header
    ├── app.settings.tsx               ← /app/settings — "Your taste" page
    └── (Shell.tsx)                    ← Settings entry added to HR_NAV
```

The pill is **deliberately visually distinct** from the existing
`RecommendationChip`: dashed outline, neutral palette, sparkle icon. It
must never be confused with the AI's hire/no-hire signal.

## Synthetic personas

Four ground-truth weight vectors live in `synthetic.py` and serve as
both demo seed data and the learner's acceptance test (Pearson > 0.7
between recovered and true weights at n=500 decisions).

| persona | core preference | top features in truth weights |
|---|---|---|
| **Pragmatist** | delivered output, technical correctness, low struggle | `technical_avg (+1.8)`, `fit_score (+1.5)`, `cognitive_load (−0.9)` |
| **Potential-spotter** | engagement / drive over polish; tolerant of gaps | `engagement_mean (+1.7)`, `confidence_mean (+1.5)` |
| **Communicator** | coherence-first; low struggle, calm | `coherence_avg (+1.9)`, `confidence_mean (+1.1)`, `cognitive_load (−1.0)` |
| **Generalist** | balanced — control persona | all features ~0.5–0.7, `missing_skills (−0.5)` |

## CLI

```bash
# Seed 100 synthetic decisions for the Pragmatist persona, fit weights,
# print the resulting profile.
python -m agents.recruit.taste.cli seed_synthetic_recruiter \
    --persona pragmatist --n 100

# All four personas, fresh each time.
for p in pragmatist potential_spotter communicator generalist; do
    python -m agents.recruit.taste.cli seed_synthetic_recruiter \
        --persona "$p" --n 100 --reset
done

# Inspect what was learned.
python -m agents.recruit.taste.cli list
python -m agents.recruit.taste.cli show --recruiter synth-pragmatist

# Wipe one profile (decisions are kept).
python -m agents.recruit.taste.cli reset --recruiter synth-pragmatist
```

`seed_synthetic_recruiter` creates real `applications` rows under a
synthetic job (`synth-job`) so the FK constraint on
`recruiter_decisions.application_id` holds and the decisions are
realistic end-to-end.

## Tests

```bash
pytest agents/recruit/tests/ -v
```

12 tests, two suites:

**`test_taste_learner.py` — does the model recover what the persona
actually weights?**
- For each of the 4 personas, generate 500 decisions, fit, compare
  recovered raw-scale weights against the persona's ground truth.
  Acceptance: Pearson > 0.7. Currently passes for all four.
- Same candidate, four personas → at least one pair of scores differs by
  more than 0.1. Proves the personas learned different things.
- `score_candidate` returns ≤3 positive and ≤2 negative factors with
  plain-text labels.
- Confidence tiers (cold / warming / warm) match `n_decisions` ranges.

**`test_taste_capture.py` — is the hook safe and correct?**
- Approving / rejecting an application writes a `recruiter_decisions`
  row with the cached feature snapshot.
- Unknown decision values are ignored.
- Missing application id is fail-soft (returns None, no raise).
- Refit is triggered exactly on the 5th, 10th, 15th… decision — patched
  fit fn count check.
- Even if every internal call raises, `capture_decision` returns None
  and never propagates — the surrounding HTTP transition stays clean.

## Confidence semantics in the UI

| `confidence` | UI behavior |
|---|---|
| `cold` (<10 decisions) | Pill renders muted, label says "Learning your taste". No score number. |
| `warming` (10–49) | Score visible but tooltip notes it's based on a small history. |
| `warm` (50+) | Score visible, tooltip says it reflects historical patterns. |

Bias-amplification mitigations are NOT implemented in the POC; see
"Out of scope" below.

## Constraints honoured

The spec for this module locked in five architectural constraints —
all upheld:

1. **Feature vectors are cached, not recomputed at decision time.** The
   `candidate_features` table is keyed by `(application_id, version)`;
   the capture hook reads from there.
2. **Refit-from-scratch, not online SGD.** Every refit pulls the full
   decision history.
3. **Simple L2-regularised logistic regression.** No GBDT, no
   hierarchical / mixed-effects models.
4. **Tiebreaker, not re-ranker.** The pill renders next to — never
   replaces — the existing recommendation chip. The scoring agent's
   output is unchanged.
5. **Removable.** Drop `agents/recruit/taste/`, the four `/taste/*`
   routes in `recruit/api.py`, the capture-hook calls in
   `update_application` / `invite_interview`, the three frontend
   touches, and the `Settings` entry in `Shell.tsx` — and the rest of
   the platform works unchanged.

## Out of scope (deliberate)

- **Bias auditing.** The feature set excludes demographic-correlated
  signals by construction, but no formal audit / fairness check exists.
  Real deployment needs (a) per-feature exclusion list, (b) periodic
  drift / divergence audit, (c) recruiter-facing transparency on what's
  being learned. None of this is in the POC.
- **Onboarding quiz.** Cold-start uses zero prior; the UI just mutes
  the pill until ~10 decisions accumulate.
- **LLM-generated style memos.** Top factors are template-rendered
  from `FEATURE_LABELS` — no LLM in this feature.
- **Hierarchical / cross-recruiter learning.** Each recruiter is fit
  independently. No shared base, no transfer.
- **Online SGD / incremental updates.** Refit only.
- **Modifying the scoring agent's output.** The scoring agent is
  untouched aside from the pre-existing feature-cache write path; no
  fork, no shadow mode.
- **Real auth.** `recruiter_id` is a header value with a default.

## How to demo (no real users needed)

```bash
# 1. Seed all four personas.
for p in pragmatist potential_spotter communicator generalist; do
    python -m agents.recruit.taste.cli seed_synthetic_recruiter \
        --persona "$p" --n 100 --reset
done

# 2. Inspect the learned profiles.
python -m agents.recruit.taste.cli list
python -m agents.recruit.taste.cli show --recruiter synth-communicator

# 3. From the running FastAPI server, switch recruiter identities by
#    just changing the header — no auth needed.
curl -H "X-Recruiter-Id: synth-pragmatist" \
     http://localhost:8000/api/recruit/taste/me/profile
curl -H "X-Recruiter-Id: synth-communicator" \
     http://localhost:8000/api/recruit/taste/score/<some-application-id>
```

The two `score` calls should produce **visibly different** numbers and
factor lists for the same candidate — that's the demo gate.
```

## Status as shipped

Backend: 12 tests passing, CLI demo verified for all 4 personas.
Frontend: code complete, **not yet built** — needs `pnpm install` /
`vite build` on a node-equipped VM. Confidence levels render correctly
in the existing test fixtures; the live HTTP path was verified via
FastAPI test client only.
