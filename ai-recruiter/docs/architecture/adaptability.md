# Adaptability — the Recruiter-Taste subsystem

> Why HireFlow shows a *second*, dashed pill next to the AI's
> recommendation, what's behind it, and why it stays muted at first.

This document is the end-to-end spec for the **adaptability** feature
(internally called *recruiter taste*). It lives inside the **recruit**
agent and learns one logistic-regression model **per recruiter** from
their own approve / reject history, then surfaces a "your fit" pill that
sits *next to* — never replaces — the existing AI recommendation chip.

---

## 1. The feature in one paragraph

When a recruiter clicks **Approve** (invite-to-interview) or **Reject**
on a candidate, HireFlow snapshots a 10-dimensional feature vector for
that candidate (CV/JD fit, hard-skill counts, scoring averages, vision
behavioral means…) alongside the decision. After every 5 decisions, an
L2-regularised logistic regression is **refit from scratch** on the
recruiter's full history. The fitted weights become a *taste profile*.
Whenever the recruiter later opens any candidate, the same feature
vector is run through their personal profile and a `score ∈ [0, 1]` is
shown in a separate pill, with the **top 3 positive** and **top 2
negative** contributing factors in plain language.

The pill is a **tiebreaker signal**, not a re-ranker. The AI's
recommendation chip is never modified, never overridden, never
recomputed. Two recruiters disagreeing on the same candidate is what
drives the model — and what it surfaces back to each of them.

---

## 2. High-level spec

| Property | Value |
|---|---|
| Where it lives | `agents/recruit/taste/` (a sub-package of the recruit agent — *not* a separate agent) |
| Persistence | SQLite (same DB the recruit agent owns), 3 new tables |
| Model | `sklearn.linear_model.LogisticRegression(penalty="l2", C=1.0, solver="lbfgs")` |
| Training mode | **Refit from scratch** at every 5th decision (no online SGD, no incremental update) |
| Feature vector | 10 dims, `FEATURES_VERSION = 1`, snapshot stored on every decision |
| Cold-start handling | Score muted in UI until the recruiter has accumulated **≥ 10 decisions** |
| Identity model | `X-Recruiter-Id` header, default `"hr-default"` (no auth required for the POC) |
| Removability | Drop the package + 4 routes + 3 frontend touchpoints → rest of platform unchanged |
| Out of scope (POC) | bias auditing, onboarding quiz, LLM rationales, hierarchical models, real auth |

---

## 3. Detailed architecture

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
   PATCH /api/recruit/apps/{id} (stage=rejected) ──┤   capture hook
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
   GET  /api/recruit/taste/me/profile        ─────┐
   GET  /api/recruit/taste/score/{app_id}    ─────┤   read paths
   POST /api/recruit/taste/me/refit          ─────┤
   POST /api/recruit/taste/me/reset          ─────┘
```

### 3.1 Data flow — write side (decision capture)

1. Recruiter clicks **Approve** or **Reject** in the HireFlow UI.
2. The existing recruit-API transition runs first (state machine,
   notifications, etc.) — that path is unchanged.
3. The `recruiter_taste.capture_decision(...)` hook runs **after** the
   transition, wrapped in `try/except`. A failure here is logged but
   never blocks the transition.
4. Hook reads (or computes-and-caches) the candidate's feature vector
   via `get_or_compute_features(...)`, then writes a row to
   `recruiter_decisions` with both the **features snapshot** (frozen
   in time) and the **features version**.
5. If this is the recruiter's `5th, 10th, 15th, …` decision, schedule
   a background refit via `asyncio.to_thread` — fire-and-forget.

### 3.2 Data flow — refit

1. Pull all `recruiter_decisions` for `recruiter_id`.
2. Build `(X, y)` with `X[i]` = the persisted snapshot at decision *i*
   in canonical `FEATURE_NAMES` order, `y[i] = 1 if decision == "approved"`.
3. Compute `means`, `stds` from the training-set, z-score `X`.
4. Fit `LogisticRegression(penalty="l2", C=1.0, max_iter=500)`.
5. Persist `{weights, intercept, means, stds, n_decisions, confidence,
   version}` into `recruiter_profiles` (INSERT OR REPLACE; version
   increments).

Edge cases:
- **0 decisions** → zero-weight profile, COLD.
- **All same label** (only approvals so far) → zero-weight profile,
  intercept set to the empirical log-odds, COLD/WARMING by count.
- **Constant feature** (std ≈ 0) → safe-divide treats `std=0 as 1`, the
  weight collapses naturally toward zero.

### 3.3 Data flow — read side (score-on-read)

1. UI calls `GET /api/recruit/taste/score/{application_id}`.
2. Server resolves recruiter id from `X-Recruiter-Id` (or default).
3. `get_or_compute_features(...)` returns the cached vector.
4. `score_candidate(...)`:
   - If no profile exists → return **`{score: 0.5, confidence: cold}`**
     with no factors. This is what the UI shows as "Learning your taste".
   - Else: `logit = intercept + Σ wᵢ · zᵢ`, `score = sigmoid(logit)`,
     and the per-feature `wᵢ · zᵢ` contributions are sorted into top 3
     positive / top 2 negative for the tooltip.

### 3.4 Database schema

Three tables appended to `agents/recruit/db.py`'s `SCHEMA_SQL`. All
idempotent — `init_db()` keeps existing rows.

```sql
-- candidate feature cache, keyed by (application, version)
candidate_features (
  application_id    TEXT NOT NULL,
  features_version  INTEGER NOT NULL,
  features_json     TEXT NOT NULL,
  computed_at       REAL NOT NULL,
  PRIMARY KEY (application_id, features_version)
);

-- one row per approve/reject; carries its own snapshot so future
-- feature-set bumps don't invalidate history.
recruiter_decisions (
  id                      TEXT PRIMARY KEY,
  recruiter_id            TEXT NOT NULL,        -- free-text, no FK to a users table
  application_id          TEXT NOT NULL,
  features_snapshot_json  TEXT NOT NULL,
  features_version        INTEGER NOT NULL,
  decision                TEXT NOT NULL CHECK (decision IN ('approved','rejected')),
  dwell_seconds           REAL,
  notes_text              TEXT,
  created_at              REAL NOT NULL
);

-- one row per recruiter; INSERT OR REPLACE on every refit.
recruiter_profiles (
  recruiter_id      TEXT PRIMARY KEY,
  weights_json      TEXT NOT NULL,              -- {weights, intercept, means, stds}
  n_decisions       INTEGER NOT NULL DEFAULT 0,
  confidence_level  TEXT NOT NULL CHECK (confidence_level IN ('cold','warming','warm')),
  version           INTEGER NOT NULL DEFAULT 1,
  updated_at        REAL NOT NULL
);
```

### 3.5 The 10 features (`FEATURES_VERSION = 1`)

| name | source | range | typical sign |
|---|---|---|---|
| `fit_score` | `applications.fit_score` (matching agent) | 0–1 | + |
| `hard_skill_overlap` | `len(applications.matched_skills)` | 0–~15 | + |
| `missing_skills_count` | `len(applications.missing_skills)` | 0–~10 | − |
| `technical_avg` | scoring report `overall.technical_avg` | 0–5 | + |
| `coherence_avg` | scoring report `overall.coherence_avg` | 0–5 | + |
| `recommendation_tier` | scoring `recommendation` (no_hire 0 → strong_hire 3) | 0–3 | + |
| `engagement_mean` | vision behavioral `engagement_level.mean` | 0–1 | + |
| `confidence_mean` | vision behavioral `confidence_level.mean` | 0–1 | + |
| `cognitive_load_mean` | vision behavioral `cognitive_load.mean` | 0–1 | − |
| `emotional_arousal_mean` | vision behavioral `emotional_arousal.mean` | 0–1 | mixed |

Sign is the *expected* direction for a "typical" recruiter; the learner
discovers each recruiter's actual direction. The plain-language labels
shown in the tooltip live in `FEATURE_LABELS` (e.g. `cognitive_load_mean`
→ *"cognitive load (stress proxy)"*).

### 3.6 Confidence tiers

```
n_decisions  <  10   → "cold"      ← UI mutes the pill
10 ≤ n  <  50        → "warming"   ← score visible, tooltip notes small history
n ≥ 50               → "warm"      ← score visible, "warm" footer in tooltip
```

`Confidence.for_n(n)` is the single source of truth — both backend
serialisation and frontend tooltip text branch on this enum.

### 3.7 Code map

```
agents/recruit/
├── db.py                              ← SCHEMA_SQL (added 3 tables)
├── api.py                             ← capture wired into transitions; +4 routes
└── taste/
    ├── __init__.py                    ← public re-exports
    ├── features.py                    ← 10-dim feature cache
    ├── learner.py                     ← sklearn LR fit + score
    ├── synthetic.py                   ← 4 ground-truth personas
    ├── capture.py                     ← decision-capture hook (fail-soft)
    └── cli.py                         ← seed / list / show / reset CLI

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

---

## 4. End-to-end lifecycle (what *actually* happens, decision-by-decision)

This is the part that explains why your pill stays muted at first.
Suppose a brand-new recruiter (or a fresh `hr-default` after a DB
wipe) starts approving / rejecting:

| Decision N | What the capture hook does | Profile state after | What `GET /score/{id}` returns | What the pill shows |
|:--:|---|---|---|---|
| 1 | Writes `recruiter_decisions` row. No refit (1 % 5 ≠ 0). | **none** (no profile row) | `score = 0.5`, `confidence = cold`, `top_positive = []`, `top_negative = []` | "Learning your taste" (muted) |
| 2 | Writes row. No refit. | none | same as above | "Learning your taste" |
| 3 | Writes row. No refit. | none | same | "Learning your taste" |
| 4 | Writes row. No refit. | none | same | "Learning your taste" |
| **5** | Writes row. **Schedules refit.** | profile created, `n=5`, **`confidence = cold`** | real score, **`cold`** | "Learning your taste" (still muted because cold) |
| 6–9 | Writes rows. No refit. | profile from N=5 (stale), still `n=5`, `cold` | real score, `cold` | "Learning your taste" |
| **10** | Writes row. **Schedules refit.** | profile updated, `n=10`, **`confidence = warming`** | real score, **`warming`** | **"Your fit XX%"** ← finally activates |
| 11–14 | Writes rows. No refit. | profile from N=10, `n=10`, `warming` | real score, `warming` | "Your fit XX%" |
| **15** | Writes row. **Schedules refit.** | `n=15`, `warming` | real score, `warming` | "Your fit XX%" |
| … | … | … | … | … |
| **50** | Writes row. **Schedules refit.** | `n=50`, **`warm`** | real score, `warm` | "Your fit XX%" |

Two thresholds, both intentional:

1. **Refit cadence (every 5 decisions).** The first refit only happens
   at N=5 — before that, no profile row exists at all and the score
   endpoint returns the neutral `0.5 / cold` placeholder.
2. **Confidence gate (n ≥ 10).** Even after the first refit, the UI
   keeps the pill muted until the model has seen at least 10 decisions.
   Logistic regression on 5 points is too noisy to display as a
   percentage with a straight face.

These are independent. If you wanted the pill to activate as soon as a
profile exists, you'd have to drop the confidence-gate (set the COLD
boundary to ≤ 5 in `Confidence.for_n` in `agents/recruit/taste/learner.py`).

---

## 5. UI interaction (where the pill shows up and what each state looks like)

### 5.1 Where the pill is rendered

The component is `apps/hireflow/src/components/recruiter-taste/RecruiterFitPill.tsx`
and is mounted in **three** places, plus a settings page:

1. **Applicants list row** (`app.applicants.tsx`, right cell of each row)
   — small variant (`size="sm"`), one per candidate.
2. **Applicants candidate panel** (`app.applicants.tsx`) — header of the
   side panel, only when `application.stage !== "applied"`.
3. **Interview report header** (`app.applicants_.$id.report.tsx`) —
   directly under the breadcrumb, only when stage is past `applied`.
4. **`/app/settings` → "Your taste"** (`app.settings.tsx`) — not the
   pill itself; this is the *introspection* page that shows your
   current weights, lets you force a refit, or reset.

The Settings entry is wired in `Shell.tsx` (`HR_NAV` array).

### 5.2 Visual language — deliberately distinct from the AI chip

The pill is **dashed-outlined, neutral palette, sparkle icon**. This is
not a stylistic choice; it is a contract:

- The **AI recommendation chip** (green / yellow / red) is the platform's
  centrally-computed hire signal — solid border, success/warning/destructive
  palette.
- The **Recruiter-fit pill** is *your* model — dashed border, slate /
  accent-tint palette, never green/red.

A user must never confuse one for the other.

### 5.3 Pill states

The pill renders one of four visual states based on
`useRecruiterFitScore(applicationId)`:

| State | Trigger | Label | Tooltip |
|---|---|---|---|
| **Loading** | query in flight, no `data` yet | "…" with faded sparkle | none |
| **Error** | query rejected | nothing (component returns `null`) | n/a |
| **Cold** | `confidence === "cold"` | **"Learning your taste"**, dashed-muted | "Cold start — learning your preferences. Score is muted until ~10 decisions." |
| **Warming** | `0.4 ≤ score < 0.65` and not cold | **"Your fit XX%"**, neutral palette | top factors + "Warming up — based on a small history; treat as a hint." |
| **Warm-high** | `score ≥ 0.65` and not cold | **"Your fit XX%"**, accent tint | top factors + "Warm — fit reflects your historical approve/reject patterns." |
| **Warm-low** | `score < 0.4` and not cold | **"Your fit XX%"**, muted | top factors + "Warm — …". (Same footer; lower score, mute palette.) |

### 5.4 Tooltip anatomy (hover)

```
┌────────────────────────────────────────────────┐
│ RECRUITER FIT                       73%        │  ← score, or "—" if cold
├────────────────────────────────────────────────┤
│ PULLED TOWARD APPROVE                          │
│   answer coherence            +0.42            │  ← top 3 (per-feature contribution)
│   AI hiring recommendation    +0.31            │
│   technical answer quality    +0.18            │
│ PULLED TOWARD REJECT                           │
│   missing required skills     −0.27            │  ← top 2
│   cognitive load (stress)     −0.11            │
├────────────────────────────────────────────────┤
│ Warm — fit reflects your historical            │
│ approve/reject patterns.                        │  ← italics, muted
└────────────────────────────────────────────────┘
```

In **cold** state, the score row shows `—`, the two factor lists are
empty (so the headings/lists are hidden by `FactorList`'s
empty-guard), and only the cold-start footer is shown:

```
┌────────────────────────────────────────────────┐
│ RECRUITER FIT                       —          │
├────────────────────────────────────────────────┤
│ Cold start — learning your preferences.        │
│ Score is muted until ~10 decisions.            │
└────────────────────────────────────────────────┘
```

### 5.5 The Settings → Your taste page

`/app/settings` adds an introspection surface that the pill cannot
provide (the pill is per-candidate; this is per-recruiter).

Components on that page:

- **Stats row** — `Decisions captured`, `Confidence` (Learning / Warming up / Warm), `Profile version`.
- **Pulled toward approve** — top 5 positive features with weight bars.
- **Pulled toward reject** — top 3 negative features.
- **Refit now** — `POST /api/recruit/taste/me/refit`. Idempotent. Useful
  if you want to force a refresh without waiting for the next 5-multiple.
- **Reset** — `POST /api/recruit/taste/me/reset`. Wipes the profile row
  but **keeps the decision history**, so a subsequent refit reproduces
  the same model.

Cold state on the Settings page reads:
> "No decisions captured yet. Approve or reject a few candidates and the
> model will start to learn what you weight."

---

## 6. "But hovering still says 'Learning your taste'" — troubleshooting

This is the expected behaviour when the criteria below are not yet
met. Walk down the list in order:

### 6.1 Have you crossed both thresholds?

Run this against your DB (any sqlite3 client or via the CLI in §7):

```sql
-- decision count for the default recruiter
SELECT COUNT(*) FROM recruiter_decisions WHERE recruiter_id = 'hr-default';

-- profile state, if any
SELECT recruiter_id, n_decisions, confidence_level, version, updated_at
FROM recruiter_profiles WHERE recruiter_id = 'hr-default';
```

Expected outcomes:

| `recruiter_decisions` count | `recruiter_profiles` row | Pill state |
|---|---|---|
| 0–4 | none | cold (no profile) |
| 5–9 | one row, `confidence_level = 'cold'` | cold (gated by `for_n`) |
| 10+ | one row, `confidence_level = 'warming'` or `'warm'` | **active** |

If the count is 10+ but `confidence_level` is still `cold`, the refit
hasn't run. Force one with the **Refit now** button on `/app/settings`
or `POST /api/recruit/taste/me/refit`.

### 6.2 Are your decisions landing under the recruiter id you think?

The frontend doesn't currently send `X-Recruiter-Id` (no auth in the
POC), so every decision and every score query lands under
`recruiter_id = 'hr-default'`. That's by design — but it also means:

- **If you wiped the DB**, the decision count restarts.
- **If you switch to a synthetic persona via curl** (`-H "X-Recruiter-Id:
  synth-pragmatist"`) and then check the pill in the browser, the
  browser is still asking about `hr-default` — the score won't reflect
  the persona.

To check what id the browser is asking about:

```bash
# Look at the DevTools network tab, GET /api/recruit/taste/score/<id>
# There's no X-Recruiter-Id header → backend resolves it to "hr-default".
```

### 6.3 Are decisions actually being captured?

Approve **or** reject in the UI, then immediately:

```sql
SELECT created_at, decision, application_id
FROM recruiter_decisions
ORDER BY created_at DESC
LIMIT 5;
```

If nothing new appears, the capture hook is failing silently (it's
fail-soft by design — the HTTP transition doesn't error). Tail the
server logs for `agents.recruit.taste.capture` warnings.

Common capture failures:

- `application not found` — the recruit transition succeeded but the
  application row vanished between the transition and the hook (very
  unlikely; would indicate an external process).
- `feature compute failed` — usually a missing/corrupt `behavioral_json`
  or scoring report. The hook proceeds with zero values for those
  features rather than dropping the decision.
- `IntegrityError` — schema not migrated. Run `init_db()` (it's
  idempotent; safe to call any time).

### 6.4 Fast path: skip the cold start with synthetic data

If you want to *see* the pill activate during a demo without 10 real
decisions, seed a synthetic persona — see §7 below.

---

## 7. Demo paths

### 7.1 Make the pill activate immediately

```bash
cd /teamspace/studios/this_studio/Agentic-Ai-Recruiter/ai-recruiter

# Seed 100 decisions for the "pragmatist" persona under recruiter id
# `synth-pragmatist`. Creates synthetic applications + decisions, fits
# the model, and prints the resulting profile.
python -m agents.recruit.taste.cli seed_synthetic_recruiter \
    --persona pragmatist --n 100 --reset

# Inspect:
python -m agents.recruit.taste.cli list
python -m agents.recruit.taste.cli show --recruiter synth-pragmatist
```

Then ask the API for that persona's view:

```bash
curl -H "X-Recruiter-Id: synth-pragmatist" \
     http://localhost:8000/api/recruit/taste/me/profile

curl -H "X-Recruiter-Id: synth-pragmatist" \
     http://localhost:8000/api/recruit/taste/score/<some-application-id>
```

### 7.2 The four ground-truth personas (`agents/recruit/taste/synthetic.py`)

| persona | core preference | top features in truth weights |
|---|---|---|
| **Pragmatist** | delivered output, technical correctness, low struggle | `technical_avg (+1.8)`, `fit_score (+1.5)`, `cognitive_load (−0.9)` |
| **Potential-spotter** | engagement / drive over polish; tolerant of gaps | `engagement_mean (+1.7)`, `confidence_mean (+1.5)` |
| **Communicator** | coherence-first; low struggle, calm | `coherence_avg (+1.9)`, `confidence_mean (+1.1)`, `cognitive_load (−1.0)` |
| **Generalist** | balanced — control persona | all features ~0.5–0.7, `missing_skills (−0.5)` |

Each persona seeds with `--n 100 --reset` produces a *visibly different*
weight ranking — that's the demo gate (§9 in the package README).

---

## 8. API surface (under `/api/recruit/taste/*`)

All routes read `X-Recruiter-Id` (default `hr-default`).

| Method | Path | Body | Returns |
|---|---|---|---|
| `GET` | `/me/profile` | — | `RecruiterTasteProfile` — `n_decisions`, `confidence`, `version`, top 5 positive + top 3 negative features. |
| `GET` | `/score/{application_id}` | — | `RecruiterFitScore` — `score`, `confidence`, `recruiter_id`, `top_positive[3]`, `top_negative[2]`. Recomputes features on cache miss. |
| `POST` | `/me/refit` | — | The freshly-fit profile, same shape as `/me/profile`. |
| `POST` | `/me/reset` | — | `{recruiter_id, reset: true/false}`. |

All routes are no-op-safe: refit on zero decisions returns a zero-weight
COLD profile; score on a missing profile returns `0.5 / cold`.

---

## 9. Constraints honoured

The locked-spec for this subsystem committed to five constraints — all
upheld:

1. **Feature vectors are cached, not recomputed at decision time.** The
   `candidate_features` table is keyed by `(application_id, version)`;
   capture and score share the same cache.
2. **Refit-from-scratch, not online SGD.** Every refit reads the full
   decision history. No incremental updates.
3. **Simple L2-regularised logistic regression.** No GBDT, no
   hierarchical / mixed-effects models.
4. **Tiebreaker, not re-ranker.** The pill renders next to — never
   replaces — the existing recommendation chip. The scoring agent's
   output is unchanged.
5. **Removable.** Deletable as a unit (the package + 4 routes + 3
   frontend touchpoints) with no other-side effects.

---

## 10. Out of scope (deliberate, for the POC)

- **Bias auditing.** The feature set excludes demographic-correlated
  signals by construction, but no formal audit / fairness check exists.
  Real deployment needs (a) per-feature exclusion list, (b) periodic
  drift / divergence audit, (c) recruiter-facing transparency on what's
  being learned.
- **Onboarding quiz.** Cold-start uses zero prior; the UI mutes the
  pill until ~10 decisions accumulate.
- **LLM-generated style memos.** Top factors are template-rendered from
  `FEATURE_LABELS` — no LLM in this feature.
- **Hierarchical / cross-recruiter learning.** Each recruiter is fit
  independently. No shared base, no transfer.
- **Online SGD / incremental updates.** Refit only.
- **Modifying the scoring agent's output.** The scoring agent is
  untouched; no fork, no shadow mode.
- **Real auth.** `recruiter_id` is a header value with a default.

---

## 11. Tests

```bash
pytest agents/recruit/tests/ -v
```

Twelve tests across two suites:

**`test_taste_learner.py` — does the model recover what the persona
actually weights?**
- For each of the 4 personas, generate 500 decisions, fit, compare
  recovered raw-scale weights against the persona's ground truth.
  Acceptance: Pearson > 0.7. Currently passes for all four.
- Same candidate, four personas → at least one pair of scores differs
  by more than 0.1. Proves the personas learned different things.
- `score_candidate` returns ≤3 positive and ≤2 negative factors with
  plain-text labels.
- Confidence tiers (cold / warming / warm) match `n_decisions` ranges.

**`test_taste_capture.py` — is the hook safe and correct?**
- Approving / rejecting an application writes a `recruiter_decisions`
  row with the cached feature snapshot.
- Unknown decision values are ignored.
- Missing application id is fail-soft.
- Refit fires exactly on the 5th, 10th, 15th… decision.
- Even if every internal call raises, `capture_decision` returns None
  and never propagates.

---

## 12. tl;dr — why your pill says "Learning your taste"

Because the cold-start gate is `n_decisions < 10`. You either:

1. Have fewer than 5 decisions captured under `recruiter_id =
   'hr-default'` → no profile row exists at all.
2. Have 5–9 decisions → profile exists, but `confidence` is `'cold'`,
   so the UI mutes the pill on purpose.

Make 10 approve/reject decisions in the UI (or run the synthetic
seeder in §7.1 for an instant active pill), then either wait for the
next refit at the 10th decision or click **Refit now** in
`/app/settings`. The pill flips to *"Your fit XX%"* with top factors
populated.
