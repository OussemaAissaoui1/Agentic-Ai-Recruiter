# JD Generation — Integration Notes

Graph-RAG JD agent at `agents/jd_generation/`, mounted at `/api/jd/*` by
`main.py`. Coexists with `agents/recruit/` — does NOT replace `recruit/jd_gen.py`.

## Setup

### 1. Python deps

Already added to `pyproject.toml`:

```
neo4j>=5.20
```

`httpx`, `langchain`, `pydantic`, `pyyaml`, `fastapi` are already shared with
the rest of the platform. No `openai` / `anthropic` SDK — Groq goes via a
small httpx wrapper mirroring `agents/scoring/llm_client.py`.

### 2. Environment variables

Required for the agent to come up READY:

```
JD_NEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io   # AuraDB free tier example
JD_NEO4J_USER=neo4j
JD_NEO4J_PASSWORD=<your password>
JD_NEO4J_DATABASE=neo4j                            # optional, default "neo4j"

GROQ_API_KEY=<your groq key>                       # already used by agents/scoring/
```

Optional:

```
JD_PROD_GUARD=1                                    # set in prod-like envs;
                                                   # refuses `replace` uploads
                                                   # and `python -m ...seed`.
```

If Neo4j env vars are missing, the agent comes up in `ERROR` state with a
clear `error` field on `/api/jd/health` — `main.py` still starts, other
agents still serve, the JD routes return 503 with a friendly message.

If `GROQ_API_KEY` is missing, the agent comes up in `FALLBACK` state — DB
reads (graph dump, JD lookup) still work; generation returns 502 until the
key is set.

### 3. Seed the graph

```bash
cd ai-recruiter
python -m agents.jd_generation.graph.seed --clear
# clear is dev-only; refuses to run when JD_PROD_GUARD=1
```

Or via HTTP after the app is up:

```bash
curl -X POST http://localhost:8000/api/jd/graph/seed-default
```

Both load the shipped `synthetic_employees.json` (100 employees / 294 skills
/ 15 roles / 5 teams / 10 past JDs of which 3 are rejected).

To use your own data, upload via `/api/jd/graph/upload` or the
`/app/explorer` page in HireFlow. Accepts JSON (matching the shipped file
shape) or a `.zip` bundle of CSVs.

## HTTP surface (`/api/jd/*`)

```
GET    /api/jd/health
POST   /api/jd/generate                  body: {role_title, level, team_id?, stream?}
                                         SSE: tool_call | tool_result | draft |
                                              final | final_full | error | done
POST   /api/jd/{jd_id}/reject            body: {reason_text, categories?}
POST   /api/jd/{jd_id}/approve
GET    /api/jd/{jd_id}
GET    /api/jd/graph/dump                ?filter=role_family:backend&include_rejections=true
POST   /api/jd/graph/upload              multipart: file + mode
POST   /api/jd/graph/seed-default        idempotent
```

## Python API (in-process)

Other modules in the monorepo can call the agent directly without HTTP:

```python
from agents.jd_generation import (
    generate_jd, reject_jd, approve_jd, get_jd, generate_jd_stream,
)

result = await generate_jd(
    role_title="Senior Backend Engineer",
    level="senior",
    team_id="team_platform",
)
print(result.jd_text)
print(result.rejection_checks)
```

`reject_jd` accepts free-text and (optionally) `categories`. When categories
are omitted, the reason is classified into a 7-item taxonomy via a one-shot
Groq JSON call before the write.

## Upstream contract

Vacancy inputs come either from:
- HireFlow's `/app/explorer` route (recruiter UI), or
- direct `generate_jd()` calls from another module.

Both flow through the same agent — the frontend is just a thin shell over
the SSE-streaming endpoint.

## Downstream contract

When a JD is approved, the parent's existing Live Interview flow accepts
candidates who apply through it. **No code-level integration is needed in
v1** — the JD record persists in Neo4j, the `recruit` agent's sqlite still
owns Job/Application records as today. The two systems coexist.

When the parent's Scoring & Adaptability block eventually records a hire
outcome for a candidate hired via a specific JD, it should call:

```python
# TODO(v2): wire from scoring → JD pipeline
update_jd_outcome(jd_id, outcome="good_hire" | "bad_hire" | "not_filled")
```

This is intentionally a stub in v1; the consumer side will land alongside
the scoring → JD pipeline.

## Coexistence with `agents/recruit/jd_gen.py`

| | `agents/recruit/jd_gen.py` | `agents/jd_generation/` |
|---|---|---|
| Endpoint | `POST /api/recruit/jobs/generate` | `POST /api/jd/generate` |
| Backend | vLLM (Llama-3.1-8B) via NLP agent | Groq (Llama-3.3-70B-versatile) |
| Strategy | Template + LLM prose | Graph-RAG with rejection feedback |
| State | None (stateless) | Neo4j (employee graph + rejections) |
| Frontend | HireFlow's existing job-create flow | New `/app/explorer` route |

HireFlow's existing job-create flow continues to call the old endpoint
unchanged. The new graph-RAG generator is a sibling, not a replacement —
recruiters opt into it by going to **Graph Explorer** in the sidebar.

## Testing

Unit tests (no external services) — 81 tests:

```bash
cd ai-recruiter
pytest tests/jd_generation/ -m "not integration"
```

Integration tests (require live Neo4j + Groq) — 14 tests:

```bash
JD_NEO4J_URI=... JD_NEO4J_USER=... JD_NEO4J_PASSWORD=... \
GROQ_API_KEY=... \
pytest tests/jd_generation/ -m integration
```

The make-or-break test is `test_rejection_loop.py::
test_rejection_reason_influences_next_generation` — verbatim from spec
section 14. If it passes, the rejection feedback loop is actually working.

## Out of scope for v1

See `docs/superpowers/specs/2026-05-13-jd-generation-graph-rag-design.md`
section 16. Notable absences: 30/90-day outcome feedback (only the
rejection loop in v1), market context / salary lookup, multi-agent
orchestration, real HRIS integration.
