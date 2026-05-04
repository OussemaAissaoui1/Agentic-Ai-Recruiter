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
| `GROQ_API_KEY` | _(unset)_ | Required for `/run`; absence reports `fallback`. |
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

## Caching

The report is cached on first `/run` and returned by subsequent calls without
re-invoking the LLM. There is no TTL or transcript-hash invalidation: if the
underlying transcript changes after a report has been generated, the recruiter
must call `POST /{id}/run` with `{"force": true}` or `DELETE /{id}/report`
followed by `POST /{id}/run` to regenerate.

## Spec / plan

- Spec: `docs/superpowers/specs/2026-05-04-scoring-agent-design.md`
- Plan: `docs/superpowers/plans/2026-05-04-scoring-agent.md`
