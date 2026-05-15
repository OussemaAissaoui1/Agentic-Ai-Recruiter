# Scoring Agent — Design

**Date:** 2026-05-04
**Status:** Approved (brainstorm)
**Owner:** scoring agent (`ai-recruiter/agents/scoring/`)

## Purpose

Generate a hiring-decision report from a completed interview. The agent reads the
recorded transcript, CV, and JD; scores every candidate answer on two axes
(technical correctness, coherence) with a short rationale; produces an overall
hire recommendation with summary, strengths, and concerns; and persists the
report so the recruiter can re-read it without re-running the model.

This replaces the current stub at `agents/scoring/agent.py`, which only appends
loose `(source, kind, value)` records and returns 501 for `compute_final`.

## Scope

In scope:
- Read interview context (transcript, CV, JD) from the recruit SQLite DB by
  `interview_id`.
- Score each answer with Groq `llama-3.3-70b-versatile`.
- Produce `InterviewReport` JSON + a human-readable Markdown rendering.
- Cache reports on disk; honour a `force` flag to regenerate.
- Write the headline numbers (recommendation + averages) back to
  `interviews.scores_json` in the recruit DB.

Out of scope:
- Persisting the live transcript into the recruit DB. The NLP agent currently
  keeps sessions in memory only. This design assumes either the live flow has
  already populated `interviews.transcript_json`, or the caller passes an
  inline `transcript` override on the request body. Wiring the NLP agent to
  persist transcripts is a separate piece of work and **must** land before the
  end-to-end flow is useful in production.
- Vision / voice / behavioural signals. The agent scores text answers only.
  Those modalities will plug in later and write into `interviews.scores_json`
  alongside the NLP-derived numbers.
- Per-job-role weight presets and rater-drift calibration. Logged for the future
  in the existing scoring README; not addressed here.

## Non-goals

- Real-time scoring during the live interview. The existing
  `agents/nlp/response_scorer.py` already does that for interview-flow guidance.
  The new scoring agent is strictly post-interview / on-demand.
- Hard authorisation. The endpoint trusts the caller to be the recruiter UI.

## Assumptions

1. The recruit SQLite DB at `agents/recruit/data/recruit.db` is the source of
   truth for `interviews`, `applications`, and `jobs`.
2. `interview_id` from the recruit DB is the canonical lookup key for the
   scoring agent. The legacy `session_id` surface on the stub is removed.
3. The `GROQ_API_KEY` is provided via environment variable. The key is never
   read from a file, never logged, and never written to git.

## Architecture

```
Recruiter UI / orchestrator
        │ POST /api/scoring/{interview_id}/run
        ▼
┌─────────────────────────────────────┐
│       ScoringAgent (FastAPI)        │
│                                     │
│  ┌─────────┐   ┌──────────────┐    │
│  │ Reader  │──▶│   Scorer     │    │   reads:  recruit DB (read-only join)
│  │ (DB)    │   │  (Groq LLM)  │    │   writes: artifacts/scoring/
│  └─────────┘   └──────────────┘    │
│        │              │             │
│        ▼              ▼             │
│  ┌─────────────────────────────┐   │
│  │  ReportBuilder (json + md)  │   │
│  └─────────────────────────────┘   │
│                │                    │
└────────────────┼────────────────────┘
                 ▼
   artifacts/scoring/<interview_id>_report.json
   artifacts/scoring/<interview_id>_report.md
   recruit DB: UPDATE interviews SET scores_json = ... WHERE id = ?
```

## Module layout

```
agents/scoring/
├── __init__.py
├── agent.py             # ScoringAgent — FastAPI router, A2A handle_task, lifecycle
├── agent_card.json      # capability advertisement
├── README.md            # rewritten for the new surface
├── reader.py            # Loads (transcript, cv, jd, candidate_name, job_title) by interview_id
├── llm_client.py        # Async Groq client wrapper, reads GROQ_API_KEY
├── scorer.py            # Builds prompts, calls LLM, parses JSON, retries on failure
├── report.py            # Aggregates per-turn results → InterviewReport + Markdown renderer
├── schema.py            # Pydantic models: TurnScore, OverallAssessment, InterviewReport, ScoreRequest
├── prompts.py           # System + user prompt templates for per-turn and overall passes
└── tests/
    ├── conftest.py      # fixture sqlite DB seeded with one interview / app / job
    ├── test_reader.py
    ├── test_scorer.py
    ├── test_report.py
    └── test_agent_routes.py
```

## Data contracts

```python
class TurnScore(BaseModel):
    turn_index: int                   # 0-based position in transcript
    question: str
    answer: str
    technical_score: int | None       # 0–5; None when scoring failed for this turn
    technical_rationale: str          # 1–2 sentences; carries error string on failure
    coherence_score: int | None       # 0–5
    coherence_rationale: str

class OverallAssessment(BaseModel):
    recommendation: Literal["strong_hire", "hire", "lean_hire", "no_hire"]
    technical_avg: float              # mean of non-None technical_score values
    coherence_avg: float              # mean of non-None coherence_score values
    summary: str                      # 3–5 sentence narrative for the recruiter
    strengths: list[str]              # 2–4 bullets
    concerns: list[str]               # 2–4 bullets

class InterviewReport(BaseModel):
    interview_id: str
    candidate_name: str | None
    job_title: str | None
    generated_at: float               # unix ts
    model: str                        # "llama-3.3-70b-versatile"
    turns: list[TurnScore]
    overall: OverallAssessment

class ScoreRequest(BaseModel):
    force: bool = False                            # bypass cache
    transcript: list[dict] | None = None           # optional inline override
                                                   # shape: [{"q": str, "a": str}, ...]
```

The transcript stored in `interviews.transcript_json` is expected to be a list
of `{"q": str, "a": str}` objects. The reader normalises adjacent variants
(`{"question", "answer"}`, `{"role", "content"}`) into the canonical shape.

## API surface

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/scoring/health` | Readiness — checks `GROQ_API_KEY` presence + recruit DB reachability |
| `POST` | `/api/scoring/{interview_id}/run` | Run scoring (or return cache); body: `ScoreRequest` |
| `GET` | `/api/scoring/{interview_id}/report` | Return cached `InterviewReport` JSON; 404 if not run |
| `GET` | `/api/scoring/{interview_id}/report.md` | Return cached Markdown; 404 if not run |
| `DELETE` | `/api/scoring/{interview_id}/report` | Drop the cached report |

A2A capabilities exposed via `handle_task`:

- `score_interview` — payload `{interview_id, force?, transcript?}`, returns the report
- `get_report` — payload `{interview_id}`, returns the cached report or 404
- `delete_report` — payload `{interview_id}`, returns `{ok: true}`

The legacy `record_score` / `get_scores` / `compute_final` capabilities are
removed. The session-keyed JSONL files at `artifacts/scoring/<session_id>.jsonl`
are no longer written. Existing files on disk are left in place; they are
ignored by the new agent and can be deleted manually.

## Data flow (one `run` call)

1. **Cache check.** If `artifacts/scoring/<interview_id>_report.json` exists
   and `force=False`, deserialise and return it. Skip every step below.
2. **Load context.** `reader.py` opens the recruit DB read-only:
   - `interviews` row by `id` → `transcript_json` (parsed). If it is empty
     and the request body carries an inline `transcript`, use that. If both
     are empty, raise 422.
   - `applications` row via `interviews.application_id` → `cv_text`,
     `candidate_name`.
   - `jobs` row via `applications.job_id` → JD text, job title.
3. **Per-turn scoring.** `scorer.py` iterates the transcript. For each
   `(question, answer)` pair, calls Groq with a strict-JSON prompt that
   asks for `{technical_score, technical_rationale, coherence_score,
   coherence_rationale}`. Concurrency is bounded by an
   `asyncio.Semaphore(4)`.
4. **Overall pass.** A second Groq call receives the transcript and the
   per-turn scores and returns the `OverallAssessment` body — recommendation,
   summary, strengths, concerns. Means (`technical_avg`, `coherence_avg`) are
   computed in code from the per-turn results, not asked from the LLM.
5. **Persist.**
   - Write `artifacts/scoring/<interview_id>_report.json` (canonical).
   - Write `artifacts/scoring/<interview_id>_report.md` (rendered for humans).
   - Update `interviews.scores_json` in the recruit DB with
     `{recommendation, technical_avg, coherence_avg, generated_at}`.
6. **Return.** The full `InterviewReport`.

## LLM prompts

Two prompt templates live in `prompts.py`. Both use the Groq chat API with
`response_format={"type": "json_object"}` to force valid JSON.

**Per-turn prompt (one call per `(q, a)` pair).** System: a calibrated
technical interviewer who returns strict JSON. User: includes the JD title,
a CV summary capped at ~600 chars, the conversation context capped at the
last three turns, the question, and the answer. Asks for the four fields
defined in `TurnScore`. Rubric guidance:

- `technical_score` 0 = no signal / nonsense; 5 = clearly demonstrates expert
  command of the topic with concrete details. The rubric description includes
  one anchor sentence per integer 0–5 to keep the model from collapsing onto
  3–4.
- `coherence_score` 0 = incoherent; 5 = articulate, well-structured, easy to
  follow regardless of technical depth. Coherence is independent of
  correctness.
- Each rationale is 1–2 sentences and references something specific from the
  answer.

**Overall prompt (one call per interview).** System: same persona, instructed
to act as the lead interviewer summarising findings. User: full transcript,
the per-turn scores rendered compactly, the JD, and the CV. Asks for
`{recommendation, summary, strengths, concerns}`. The recommendation enum is
explicitly enumerated in the prompt.

The exact wording of the prompts is not part of the design contract and may be
tuned during implementation; the JSON shape is the contract.

## Markdown report

```markdown
# Interview Report — <candidate_name>, <job_title>
**Recommendation:** <recommendation> · Technical <technical_avg>/5 · Coherence <coherence_avg>/5
**Generated:** <generated_at iso> (<model>)

## Summary
<overall.summary>

### Strengths
- <bullet>

### Concerns
- <bullet>

---

## Per-Answer Breakdown
### Turn <n> — Technical <s>/5 · Coherence <s>/5
> **Q:** <question>
> **A:** <answer>
**Technical:** <technical_rationale>
**Coherence:** <coherence_rationale>
```

Failed turns render their score as `n/a` and the rationale shows the error
string with a `_(scoring failed)_` italic suffix.

## Error handling

| Failure | Behaviour |
|---|---|
| `GROQ_API_KEY` missing at startup | `/health` reports `degraded`; `run` returns 503 with a clear message. |
| `interview_id` not found in DB | 404. |
| Transcript empty AND no inline override | 422 with "no transcript available for this interview". |
| Groq API error on a single turn | Retry once with exponential backoff. On second failure, that turn's scores are `None` and the rationale carries the error string. Other turns proceed. |
| Groq returns non-JSON despite `response_format` | One regex repair pass to extract `{...}`; if still invalid, treat as a turn failure. |
| Overall pass fails | Recommendation falls back to a deterministic rule on the averages (`>=4.0 → hire`, `>=3.0 → lean_hire`, else `no_hire`); `summary` carries the error string. The report is still returned. |
| Cache write fails | Log warning, still return the report — caching is best-effort. |
| Recruit DB write-back fails | Log warning, still return the report — the on-disk JSON is the authoritative artifact. |

Per-turn isolation is the central error-handling invariant: one bad answer
must not tank the whole report.

## Configuration

Read from environment, with sensible defaults so the agent can boot in dev:

| Var | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | _(unset)_ | Required for `run`; absence reports `degraded`. |
| `SCORING_MODEL` | `llama-3.3-70b-versatile` | Groq model id. |
| `SCORING_CONCURRENCY` | `4` | Max parallel per-turn LLM calls. |
| `SCORING_TIMEOUT_SEC` | `30` | Per-call deadline. |
| `SCORING_RETRIES` | `1` | Per-call retry budget. |

The recruit DB path resolves through the existing `configs.resolve()` helper
the way other agents already do — no new path config.

## Testing strategy

- **`test_reader.py`** — fixture SQLite DB seeded with one job, one
  application, one interview. Asserts `(transcript, cv, jd, candidate_name,
  job_title)` shape; covers `transcript_json` empty / malformed / multi-shape;
  asserts read-only behaviour.
- **`test_scorer.py`** — mock async Groq client. Asserts the prompt body
  references the question and answer, parses good JSON, repairs slightly-bad
  JSON, fails-soft on a hard parse error, and respects retry budget.
- **`test_report.py`** — feed canned `TurnScore` lists into the report
  builder. Asserts averages, the recommendation fallback rule, and that the
  Markdown contains the expected sections (header, summary, strengths,
  concerns, per-turn breakdown).
- **`test_agent_routes.py`** — FastAPI `TestClient` against the in-process
  agent with the LLM client mocked. Covers cache hit, cache miss, 404 on
  unknown `interview_id`, 422 on empty transcript with no override,
  inline-transcript override path, `force=true` regeneration, and the
  `delete_report` endpoint.
- **`test_smoke_groq.py`** — gated on `GROQ_API_KEY`. Runs against the real
  Groq API with a 3-turn fixture transcript. Skipped in CI by default.

## Design self-review notes

- **Cross-agent coupling.** The scoring agent reads the recruit SQLite DB
  directly and writes back to one column. This is a tighter coupling than the
  other agents have. Justification: the recruit DB is the only place the
  transcript / CV / JD live; introducing an A2A round-trip just to read three
  fields would add latency and a layering hop without removing the coupling
  (the schema is still shared). The write-back is a single column update
  scoped to a row the recruiter already owns.
- **Why two LLM calls per interview, not one giant one.** A single mega-prompt
  scoring 15 turns at once gives a JSON parse failure mode where the entire
  report is lost. Per-turn isolation makes failures local. The cost is real
  but small — Groq's per-call latency on 70B is ~150–400 ms; the bounded
  concurrency keeps wall time near a single call.
- **Why averages in code, not from the LLM.** Cheaper, deterministic, and
  removes a class of "model invented a number" failures. The LLM still owns
  the qualitative recommendation.
- **Why no auto-trigger on interview end.** That requires the live flow to
  reliably write transcripts to the recruit DB, which it does not today. C
  (on-demand + cached) is the smallest design that works against the current
  state of the codebase. Auto-trigger can be added later by having the
  orchestrator dispatch `score_interview` on the appropriate state edge.
