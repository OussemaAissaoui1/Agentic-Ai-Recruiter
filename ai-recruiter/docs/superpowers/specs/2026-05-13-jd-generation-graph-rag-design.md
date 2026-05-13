# JD Generation Sub-Module — Graph-RAG Design

**Status:** Draft for review
**Date:** 2026-05-13
**Scope:** New `agents/jd_generation/` sibling agent + `/explorer` route in `apps/hireflow/`
**Replaces / overlaps:** Coexists with the existing `agents/recruit/jd_gen.py`. The recruit endpoint is untouched.

---

## 1. Goal

A recruiter inputs a vacancy (role, level, optional team). A graph-RAG agent navigates a Neo4j knowledge graph of existing employees, consults past **rejection reasons** for that role family, extracts patterns (skills, career paths, education, prior industries), reads past JDs that resulted in good hires, and produces a job description draft with an evidence panel that traces every claim back to a tool call.

If the recruiter rejects the draft, they write why. The reason is stored as a `RejectionReason` node linked to the JD. On the next generation for the same role family, the agent reads those reasons before drafting and reports back, in `rejection_checks`, how the new draft addresses each one. This is the proof-of-learning artifact.

A new `/explorer` route inside the existing HireFlow recruiter UI provides a 3D force-directed visualization of the graph, plus the ability to upload custom employee/role/skill data or run against the shipped synthetic dataset.

## 2. Locked decisions (from brainstorm)

| Question | Decision | Rationale |
|---|---|---|
| Relationship to existing `agents/recruit/jd_gen.py` | **New parallel agent at `agents/jd_generation/`** mounted at `/api/jd/*`. `recruit/jd_gen.py` untouched. | Lets HireFlow choose backend per-vacancy via a config flag. Old behavior remains intact as a safety net. |
| LLM backend | **Groq, `llama-3.3-70b-versatile`** via httpx (same pattern as `agents/scoring/llm_client.py`) | 70B handles tool-calling reliably — drops the spec's "1B unreliable" fallback path. No GPU footprint. Parent already proven this pattern. |
| 3D viz + upload feature location | **`/explorer` route inside `apps/hireflow/`** | Recruiter is the primary user; less infra than a sibling Vite app. |
| Path layout | **`agents/jd_generation/`** (not `src/jd_generation/`) | Parent has no `src/` directory. All packages live at repo root. Matches existing `agents/recruit/`, `agents/scoring/`, etc. |
| Config | **`configs/runtime.yaml` + env-var overrides** (not pydantic-settings) | Parent uses pyyaml + `lru_cache` via `configs/__init__.py`. |

## 3. Module layout

```
ai-recruiter/
├── agents/
│   └── jd_generation/
│       ├── __init__.py              # exports JDGenerationAgent + public service fns
│       ├── agent.py                 # BaseAgent subclass: lifecycle, health, card
│       ├── agent_card.json
│       ├── api.py                   # FastAPI APIRouter (mounted at /api/jd/*)
│       ├── config.py                # load_jd_config() reads runtime.yaml + JD_* env
│       ├── service.py               # public Python API
│       ├── graph/
│       │   ├── client.py            # Neo4j driver, lazy singleton
│       │   ├── schema.py            # constraints + indexes
│       │   ├── seed.py              # synthetic data loader (PROD_GUARD)
│       │   ├── queries.py           # raw Cypher (one function per query)
│       │   └── dump.py              # D3-shaped JSON for the 3D viz
│       ├── tools/
│       │   ├── retrieval.py         # 8 graph tools + read-only cypher escape hatch
│       │   └── rejection.py         # get_past_rejections, store_rejection
│       ├── runtime/
│       │   ├── prompts.py           # SYSTEM_PROMPT
│       │   ├── llm_client.py        # Groq tool-calling client
│       │   ├── runner.py            # agent loop, tool dispatch, evidence log
│       │   └── bias_check.py        # final pass: gender/age-coded language
│       ├── upload/
│       │   ├── parsers.py           # JSON / CSV-zip bundle ingestion
│       │   └── schema.py            # pydantic upload validation
│       └── data/
│           ├── synthetic_employees.json
│           └── esco_skills_subset.json

tests/
└── jd_generation/
    ├── test_graph_queries.py
    ├── test_tools.py
    ├── test_agent.py
    ├── test_rejection_loop.py       # critical acceptance test (section 12)
    └── test_upload.py
```

Naming note: the subdirectory `runtime/` (not `agent/`) holds the agent loop, to avoid a name clash with the top-level `agent.py` file that subclasses `BaseAgent`.

Registered in `main.py` alongside the others:

```python
registry.register(JDGenerationAgent())  # mounts at /api/jd/*
```

## 4. Public Python API (`agents/jd_generation/__init__.py`)

```python
from agents.jd_generation import (
    generate_jd,    # role_title, level, team_id=None -> GenerateResult
    reject_jd,      # jd_id, reason_text, categories=None -> RejectResult
    approve_jd,     # jd_id -> ApproveResult
    get_jd,         # jd_id -> JDRecord
    dump_graph,     # filter=None -> {nodes, links}
    upload_graph,   # file_bytes, mode -> UploadResult
)
```

`api.py` is a thin FastAPI wrapper. Other modules calling JD generation in-process use the Python API directly without hitting HTTP.

`GenerateResult` shape:

```python
class GenerateResult(BaseModel):
    jd_id: str
    jd_text: str
    must_have: list[str]
    nice_to_have: list[str]
    evidence: dict[str, list[ToolCallRef]]      # section -> tool calls that informed it
    rejection_checks: list[RejectionCheck]      # past reason + how addressed
    tool_call_log: list[ToolCall]               # full audit trail, ordered
    bias_warnings: list[str]                    # bias_check.py output
```

## 5. HTTP surface (`/api/jd/*`)

```
GET    /api/jd/health
POST   /api/jd/generate                 body: {role_title, level, team_id?, stream?:bool=true}
                                        SSE events: tool_call | tool_result | draft | final | error | done
POST   /api/jd/{jd_id}/reject           body: {reason_text, categories?: string[]}
POST   /api/jd/{jd_id}/approve
GET    /api/jd/{jd_id}                  -> JDRecord JSON
GET    /api/jd/graph/dump               query: filter?, include_rejections?
                                        -> {nodes: [...], links: [...]} (D3-shaped)
POST   /api/jd/graph/upload             multipart: file (json|csv|zip), mode ("augment"|"replace")
                                        -> {counts: {employees:N, roles:N, ...}, warnings: []}
POST   /api/jd/graph/seed-default       idempotent — loads synthetic_employees.json
```

SSE event shapes:

```json
{"type":"tool_call",   "tool":"get_cohort",       "args":{"role_family":"backend","level":"senior"}}
{"type":"tool_result", "tool":"get_cohort",       "node_ids":["e1","e7",...],  "summary":"23 employees"}
{"type":"draft",       "section":"description",   "text":"...partial prose..."}
{"type":"final",       "result": <GenerateResult-shaped JSON>}
{"type":"done"}
```

The `node_ids` in `tool_result` events feed the explorer's "highlight nodes the agent touched" animation.

## 6. Graph schema (Neo4j)

Verbatim from the spec; reproduced here for the design doc to be self-contained.

### Nodes

| Label | Key properties |
|---|---|
| `Employee` | `id`, `hire_date`, `tenure_years`, `level`, `status` |
| `Role` | `id`, `title`, `level`, `role_family`, `department` |
| `Skill` | `id`, `name`, `category`, `esco_id` |
| `Team` | `id`, `name`, `department`, `size` |
| `Education` | `id`, `degree`, `field`, `institution_tier` |
| `PriorCompany` | `id`, `name`, `industry`, `size_bucket` |
| `JobDescription` | `id`, `text`, `created_at`, `status`, `role_family`, `hire_outcome` (nullable) |
| `RejectionReason` | `id`, `text`, `categories`, `created_at` |

### Relationships

```
(Employee)-[:HOLDS {start_date, end_date}]->(Role)
(Employee)-[:PREVIOUSLY_HELD {start_date, end_date}]->(Role)
(Employee)-[:HAS_SKILL {proficiency: 1-5}]->(Skill)
(Employee)-[:BELONGS_TO]->(Team)
(Employee)-[:GRADUATED_FROM]->(Education)
(Employee)-[:WORKED_AT]->(PriorCompany)
(Employee)-[:HIRED_VIA]->(JobDescription)
(Role)-[:REQUIRES {weight}]->(Skill)
(Role)-[:PROGRESSES_TO {frequency}]->(Role)
(Skill)-[:ADJACENT_TO {strength}]->(Skill)
(JobDescription)-[:REJECTED_FOR]->(RejectionReason)
(JobDescription)-[:GENERATED_FOR_ROLE_FAMILY]->(Role)
```

Constraints: unique `id` on every node label. Indexes: `Role.role_family`, `Skill.name`, `RejectionReason.created_at`.

## 7. Tools — 8 retrieval + 2 rejection

### `tools/retrieval.py`

```python
find_role_family(role_title: str, level: str) -> dict
get_cohort(role_family: str, level: str, k: int = 20) -> dict
get_skill_distribution(employee_ids: list[str]) -> dict
    # returns: {universal: [skill_id, ...],   # >80% cohort frequency
    #          distinctive: [...],            # 40-80%
    #          long_tail: [...]}              # <40%
get_career_paths_into(role_family: str, depth: int = 3) -> dict
get_education_distribution(employee_ids: list[str]) -> dict
get_prior_industries(employee_ids: list[str]) -> dict
get_team_signals(team_id: str) -> dict
get_past_jds(role_family: str, outcome: str = "good_hire", k: int = 3) -> dict
cypher_query(query: str) -> dict
    # READ-ONLY enforcement: regex-rejects CREATE / MERGE / DELETE / SET / REMOVE / DROP
    # (whitespace + comment-tolerant). Run in a read transaction.
```

### `tools/rejection.py`

```python
get_past_rejections(role_family: str, k: int = 10) -> dict
    # Returns sorted-by-recency: [{"jd_snippet": "...", "reason": "...", "categories": [...], "date": "..."}]
    # MUST be called before drafting. The agent runner hardcodes this — we don't trust the model.

store_rejection(jd_id: str, reason: str, categories: list[str] | None) -> dict
    # If categories is None, classifies free-text via a small Groq JSON-mode call.
    # Taxonomy (extensible): tone, requirements, bias, culture-fit, accuracy, structure, other.
```

## 8. Agent runner

System prompt is verbatim from spec section 10. The runner enforces two non-negotiable preambles before handing control to the model:

1. **Resolve role family.** Always call `find_role_family(role_title, level)` deterministically. Result goes into the model's context.
2. **Fetch past rejections.** Always call `get_past_rejections(role_family, k=10)` deterministically. Results go into the model's context.

After that, the model drives — choosing which other retrieval tools to call, in what order, when to draft, when to self-revise. Limits: `max_tool_calls=15`, `max_revisions=3`. Each tool call is logged to `tool_call_log` for the evidence panel.

If `get_cohort` returns fewer than 3 employees even after broadening, the runner aborts with a clear error rather than letting the model hallucinate.

Final pass: `bias_check.py` scans the JD text for gender-coded and age-coded language using a small wordlist + a Groq pass for ambiguous cases. Returns a list of warnings; doesn't block — the recruiter sees them in the response.

## 9. LLM client

`runtime/llm_client.py` mirrors `agents/scoring/llm_client.py` (httpx, automatic retry on 5xx/timeout, JSON response_format) and adds two methods:

```python
class GroqClient:
    async def chat_with_tools(messages, tools, tool_choice="auto", model, temperature) -> ChatResult
    async def chat_json(system, user, model, temperature) -> dict     # existing pattern
```

`ChatResult.tool_calls` is a normalized list independent of the wire format. `chat_with_tools` uses Groq's OpenAI-compatible `tools` and `tool_choice` parameters.

The spec's "use Claude for synthesis only" abstraction is dropped because 70B is strong enough. Re-introducing Claude later is a config-only change inside `llm_client.py`.

## 10. The `/explorer` page in HireFlow

New route: `apps/hireflow/src/routes/explorer.tsx`. Recruiter-only page, accessible via a new nav entry (`Explorer`) in the HR sidebar alongside Dashboard / Jobs / Applicants / Analytics / Talent Radar / Settings.

### Layout

```
┌──────────────────────────────────────────────────────────────┐
│  Data: [Default] [Custom ▾]  · 134 employees · 51 skills · ↻ │  ← sticky bar
├──────────────────────────────┬───────────────────────────────┤
│                              │  Generate                     │
│       3D force graph         │  ─ Role title: [_______]     │  ← side
│                              │    Level: [senior ▾]          │     drawer
│   filter pills: family,      │    Team: [platform ▾]         │
│   level, team                │    [ Generate ]               │
│                              │  ─ Tool calls:                │
│   search ⌕                   │    ✓ find_role_family         │
│                              │    ✓ get_past_rejections (2)  │
│                              │    ✓ get_cohort (23)          │
│                              │  ─ Draft preview              │
│                              │    [ Approve ] [ Reject ]     │
│                              │  ─ Past rejections addressed  │
└──────────────────────────────┴───────────────────────────────┘
```

### 3D graph

- Library: `react-force-graph-3d` (Three.js underneath).
- Node colors: Employee=violet, Role=mint, Skill=amber, Team=slate, JobDescription=gradient, RejectionReason=red.
- Edge width: scales with `proficiency` for `HAS_SKILL`; with `frequency` for `PROGRESSES_TO`.
- Click node → right-side detail panel (1-hop neighbors + properties).
- Filter pills above canvas (role family, level, team).
- Search box → focus and zoom to a node.
- **Tool-call halo animation:** when the side drawer fires `/api/jd/generate` (SSE), each `tool_result` event includes `node_ids`. The frontend pulses a yellow halo on those nodes for ~3 seconds as each tool fires. This is the demo-quality moment that proves the agent is actually using the graph.

### Live agent panel

- Inputs: role title (text), level (select), team (select, populated from `/api/recruit/jobs`'s known teams).
- Generate button hits `/api/jd/generate` with `stream=true`.
- Tool-call list streams in. Final JD card shows description + must-have + nice-to-have.
- Approve → `/api/jd/{id}/approve`.
- Reject → modal with reason textarea + category multi-select (the 6-item taxonomy) → `/api/jd/{id}/reject`. After save, an inline "Re-generate to verify the loop" button regenerates immediately for the same inputs so the recruiter sees the rejection consumed in the new draft.

### Data source picker (sticky top bar)

- Toggle between **Default** (shipped synthetic data) and **Custom**.
- Custom: dropzone for `.json` / `.csv` / `.zip`, validation feedback, "Upload (augment)" and "Upload (replace)" buttons.
- Post-upload: row counts displayed inline. `Reset to defaults` button.

## 11. Data upload

Two accepted formats, validated server-side via pydantic.

### JSON (single file)

```json
{
  "employees": [{"id":"e1","hire_date":"...","level":"senior","status":"active",
                 "roles_held":[{"role_id":"r3","start":"...","end":null,"current":true}],
                 "skills":[{"skill_id":"s1","proficiency":4}],
                 "team_id":"t2","education":[...], "prior_companies":[...]}],
  "roles":     [{"id":"r3","title":"Senior Backend Engineer","level":"senior",
                 "role_family":"backend","department":"engineering"}],
  "teams":     [{"id":"t2","name":"Platform","department":"engineering","size":12}],
  "skills":    [{"id":"s1","name":"Python","category":"language","esco_id":"..."}],
  "past_jds":  [{"id":"jd1","text":"...","status":"approved","role_family":"backend",
                 "hire_outcome":"good_hire","rejections":[]}]
}
```

### CSV bundle (.zip)

Contains: `employees.csv`, `roles.csv`, `skills.csv`, `teams.csv`, plus edge CSVs (`holds.csv`, `has_skill.csv`, `worked_at.csv`, `graduated_from.csv`, optional `past_jds.csv`).

### Modes

| Mode | Behavior |
|---|---|
| `augment` | `MERGE` by `id`. Doesn't touch what's already in the graph. Safe for incremental updates. |
| `replace` | Wipes all nodes/edges **except `RejectionReason` nodes** (those are learning artifacts we never lose). Requires `confirm=true` query parameter. |

Limits: max 20 MB per upload. PROD_GUARD env var must be unset for `replace` to work — prevents accidental wipe of a production-like Neo4j instance.

## 12. Acceptance test — the rejection feedback loop

Implemented verbatim from spec section 14. Single test asserts the entire learning loop:

```python
def test_rejection_reason_influences_next_generation():
    # 1. Generate JD #1 for "Senior Backend Engineer".
    jd1 = generate_jd(role_title="Senior Backend Engineer", level="senior")

    # 2. Reject with a specific reason.
    reject_jd(
        jd_id=jd1.jd_id,
        reason_text="Too much jargon — phrases like 'synergize cross-functional "
                    "verticals' don't match our straightforward voice.",
        categories=["tone"]
    )

    # 3. Generate JD #2 for the same role family.
    jd2 = generate_jd(role_title="Senior Backend Engineer", level="senior")

    # 4. The new draft must show awareness of the prior rejection.
    assert any("jargon" in check["past_reason"].lower()
               for check in jd2.rejection_checks)

    # 5. The new draft must not repeat the rejected phrasing.
    assert "synergize" not in jd2.jd_text.lower()
    assert "cross-functional verticals" not in jd2.jd_text.lower()

    # 6. The tool call log must show get_past_rejections was called.
    tool_names = [call["tool"] for call in jd2.tool_call_log]
    assert "get_past_rejections" in tool_names
```

Marked `pytest.mark.integration` — requires a reachable Neo4j and a live Groq API key. Excluded from the default test run via `pytest -m "not integration"`; CI runs it via `pytest -m integration` against a dedicated test Neo4j instance.

## 13. Configuration

In `configs/runtime.yaml`:

```yaml
jd_generation:
  neo4j:
    uri_env: JD_NEO4J_URI
    user_env: JD_NEO4J_USER
    password_env: JD_NEO4J_PASSWORD
    database: neo4j
  llm:
    backend: groq
    model: llama-3.3-70b-versatile
    api_key_env: GROQ_API_KEY
    temperature_synthesis: 0.7
    temperature_tool_select: 0.0
  agent:
    max_tool_calls: 15
    max_revisions: 3
    rejection_lookback: 10
  upload:
    max_file_size_mb: 20
    allowed_extensions: [json, csv, zip]
```

Env vars (documented in `.env.example`):

```
JD_NEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io      # AuraDB free-tier example
JD_NEO4J_USER=neo4j
JD_NEO4J_PASSWORD=
JD_NEO4J_DATABASE=neo4j
GROQ_API_KEY=                                         # already used by scoring/
JD_PROD_GUARD=0                                       # set to 1 to refuse `replace` uploads
```

Default README guidance: **Neo4j AuraDB free tier** (zero local setup). Neo4j Desktop or Docker also work; users only need to set the URI/user/password env vars.

## 14. Dependencies

### Python (`pyproject.toml`)

Add:
```
neo4j>=5.20
```

Already present and reused: `httpx`, `langchain`, `pydantic`, `pyyaml`. We do **not** add `langchain-neo4j`, `openai`, or `anthropic` — Groq goes via the existing httpx pattern, and raw Cypher is clearer than LangChain's Neo4j abstractions for this surface.

### Frontend (`apps/hireflow/package.json`)

Add:
```
react-force-graph-3d
three
```

## 15. Integration points with the parent

- **Upstream:** Vacancy inputs come either from a recruiter via the `/explorer` panel or from other modules calling `agents.jd_generation.generate_jd()` directly.
- **Downstream:** When a JD is approved/published, the parent's existing Live Interview flow accepts candidates who apply through it. No code-level integration in v1 — the JD record persists in Neo4j, and the recruit agent's sqlite still owns Job/Application records as today.
- **Feedback edge (stubbed):** `update_jd_outcome(jd_id, outcome)` exposed but raises `NotImplementedError("v2: requires scoring → JD pipeline wiring")`. Documented in `INTEGRATION.md`.
- **Shared LLM:** Groq via `agents/jd_generation/runtime/llm_client.py` — mirrors `agents/scoring/llm_client.py` rather than importing from it, because tool-calling extension is non-trivial. If both surfaces grow further, factor into `core/llm/groq.py` in a follow-up.

## 16. Out of scope for v1

- Automatic vacancy detection
- Multi-agent orchestration of JD generation
- Vector embeddings / hybrid search
- 30/90-day outcome feedback (only rejection loop in v1; `update_jd_outcome` is a stub)
- Market context / salary lookup
- Production auth — parent handles it
- Real HRIS integration
- Replacing the existing `recruit/jd_gen.py` (parallel agent, not in-place replacement)

## 17. Definition of done

1. Module lives entirely inside `agents/jd_generation/` and `tests/jd_generation/`. The only files touched outside those directories are `main.py` (registry registration), `pyproject.toml` (neo4j dep), `configs/runtime.yaml` (jd_generation block), `.env.example` (env vars), and `apps/hireflow/` (new `/explorer` route + nav entry).
2. `python -m agents.jd_generation.graph.seed` loads synthetic data into a reachable Neo4j.
3. `uvicorn main:app` starts and `POST /api/jd/generate` returns a streaming draft with evidence panel within 30 seconds against the seeded graph.
4. `POST /api/jd/{id}/reject` stores the reason; next generation for the same role family shows it in `rejection_checks` and avoids the rejected phrasing.
5. All five test files pass: `pytest tests/jd_generation/` (unit) and `pytest tests/jd_generation/ -m integration` (with Neo4j + Groq).
6. `ruff check agents/jd_generation/` passes (parent's linter, 100-col line limit, py310 target).
7. HireFlow's `/explorer` route renders the 3D graph against `/api/jd/graph/dump`, accepts uploads, and runs the live agent panel end-to-end.
8. `agents/jd_generation/INTEGRATION.md` documents the upstream / downstream contracts.

## 18. Phased build order

| Phase | Deliverable | Verification |
|---|---|---|
| 1 | Foundation: deps, config, `graph/client.py`, `graph/schema.py`, `graph/seed.py`, synthetic data | `python -m agents.jd_generation.graph.seed` runs; constraints visible in Neo4j Browser; `test_graph_queries.py` passes |
| 2 | Tools: `queries.py`, retrieval tools, rejection tools | `test_tools.py` passes |
| 3 | LLM client + agent runner + bias check | Smoke test: `python -m agents.jd_generation.runtime.runner "Senior Backend Engineer" senior`; `test_agent.py` passes |
| 4 | API: `agent.py` (BaseAgent), `api.py`, `service.py`, `main.py` registry hook | Parent app starts; `curl /api/jd/health` returns ready; manual `POST /api/jd/generate` produces a draft |
| 5 | Rejection loop end-to-end | `test_rejection_loop.py` passes (the acceptance test) |
| 6 | Upload: `parsers.py`, `schema.py`, graph endpoints, `test_upload.py` | Upload a sample CSV bundle, see row counts; replace mode wipes everything except `RejectionReason` |
| 7 | Frontend `/explorer`: data picker, 3D graph, live agent panel, tool-call halo | Manual browser walkthrough at 1440px + 768px; reduced-motion path works; keyboard navigation works |

Each phase is independently testable; phases 1–6 are backend-only and can be merged before phase 7 ships.
