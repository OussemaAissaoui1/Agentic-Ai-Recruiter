# Agentic Interview Loop

> Turning the interview from a state-machine-with-prose-slots into a real
> ReAct-style agent loop where an LLM owns the control-flow decisions.

## TL;DR

Before this change the interview "agent" was a Python state machine
(`InterviewStateTracker.should_follow_up`) that decided when to follow
up and when to move on, while the LLM only generated the spoken text.
The audit caught this honestly: every meaningful flow-control decision
was hardcoded — the LLM was a fancy autocomplete, not an agent.

This change introduces a **second LLM that owns the decision**, in the
same vein as a planner agent in a ReAct / plan-act-observe loop.

```
                BEFORE                                 AFTER
                ──────                                 ─────
   scorer ──► RULE ENGINE ──► main LLM      scorer ──► PLANNER LLM ──► main LLM
              (deterministic)  (text)                 (decides action)  (text)
```

The result is one genuinely agentic subsystem inside HireFlow. The
overall product is still pragmatic — auditable, deterministic at the
edges, with rule-based safety nets — but the *interview strategy*
itself is now driven by an LLM that reasons about state.

This is the smallest, highest-leverage move recommended in the
"agentic-ness audit" — applied.

---

## What "agentic" means here

The five criteria from the audit, and how this change scores against
each one:

| Criterion | Before | After |
| --- | --- | --- |
| LLM-driven decisions | rule engine owns choice | **planner LLM owns choice** |
| Multi-step reasoning | none | scorer → planner → main LLM (perceive → decide → act) |
| Tool / action choice | no action vocabulary | **closed action set chosen by LLM** |
| Reasoning surfaced | none | chain-of-thought returned in `planner_output.reasoning` |
| State-aware | side effects only | planner reads a typed `StateSnapshot` |

It is still not a multi-agent system in the tool-calling sense (the
planner does not call other agents). But it is a real agent: it
**perceives** (scorer signal + state snapshot + recent turns),
**decides** an action from a closed vocabulary, **explains** its
reasoning, and the rest of the pipeline acts on its choice. That's
agentic in the strong sense for one well-scoped flow.

---

## Files touched

| Path | Change |
| --- | --- |
| `agents/nlp/interview_planner.py` | **NEW** — planner agent (LLM + JSON contract) |
| `agents/nlp/interview_state.py` | **MODIFIED** — added `apply_planner_decision()` and `remaining_topics()` |
| `agents/nlp/agent.py` | **MODIFIED** — load planner, route flow control through `_decide_next_action()`, propagate `planner_output` to the API response |
| `docs/architecture/agentic-interview-loop.md` | **NEW** — this document |

The rule engine `should_follow_up()` is **kept as a fallback** — it is
the first thing the new code reaches for if the planner is unavailable
or if its JSON cannot be parsed. Reliability does not regress.

---

## The planner agent

### Contract

```python
@dataclass(frozen=True)
class PlannerDecision:
    action: str                  # closed vocabulary, see below
    reasoning: str               # the agent's chain-of-thought
    target_topic: Optional[str]  # required when moving topics
    confidence: float            # 0–1, planner self-rated
    latency_ms: float
    raw_text: str = ""
```

### Action vocabulary (closed)

| Action | Meaning | Counts as |
| --- | --- | --- |
| `follow_up_same_topic` | dig deeper on the same CV item | follow-up |
| `ask_for_example` | answer was abstract — request a concrete example | follow-up |
| `ask_simpler` | candidate is struggling — drop to an accessible angle | follow-up |
| `clarify_previous` | candidate asked for clarity — rephrase, do not advance | follow-up |
| `move_to_new_topic` | this topic is exhausted — switch (`target_topic` required) | new topic |
| `wrap_topic_then_move` | acknowledge then move (good for partial-but-acceptable) | new topic |
| `acknowledge_skip` | candidate explicitly asked to skip | new topic |

The vocabulary is closed because we want the planner to *choose* a
strategy, not invent a new one mid-interview. `_pick_one()` snaps any
free-form output back to a legal value, mirroring the response scorer
pattern.

### Prompt

The planner receives, as a single user message:

- The interview state (turn number, current topic, depth, max
  follow-ups, topics covered, topics remaining, consecutive vague
  count).
- The scorer's structured analysis (quality, engagement, knowledge
  level, suggested focus, knowledge gaps, scorer's own recommended
  action).
- The last 4 turns of conversation history.
- The candidate's latest answer verbatim (capped at 600 chars).

It is asked to produce a single JSON object:

```json
{
  "reasoning": "...step-by-step...",
  "action": "follow_up_same_topic",
  "target_topic": "",
  "confidence": 0.82
}
```

The full prompt (`InterviewPlanner.PLANNER_PROMPT` in
`agents/nlp/interview_planner.py`) embeds the reasoning rules
(respect `max_followups`, prefer move-on for "detailed" answers, do
not move on if `topics_remaining` is empty, etc.) so the planner is a
real strategist, not a rubber stamp on the scorer's recommendation.

### Model

`Qwen/Qwen2.5-1.5B-Instruct` — the same family already used by
`ResponseScorer` and `QuestionRefiner`. By default the planner
**shares** the scorer's loaded weights via
`InterviewPlanner.attach_shared_model()`, so we don't pay a second
~3 GB of VRAM/RAM for the same model.

```python
# agents/nlp/agent.py — _ensure_models_loaded
shared = (
    self._planner_share_scorer
    and self._scorer is not None
    and self._scorer.is_ready
    and self._planner_model == self._scorer_model
)
if shared:
    self._planner = InterviewPlanner(autoload=False)
    self._planner.attach_shared_model(
        tokenizer=self._scorer._tokenizer,
        model=self._scorer._model,
        device=self._scorer.device,
    )
```

If sharing is disabled (`planner_share_scorer_model=False`) the
planner loads its own weights — useful if you want the planner on GPU
while the scorer stays on CPU.

### Async + timeout

The planner runs CPU/GPU inference inside `loop.run_in_executor` with
a 4-second `asyncio.wait_for` cap. If it doesn't return in time, the
turn falls back to the rule engine. The interview never blocks waiting
for the planner.

---

## State machine changes

`InterviewStateTracker` keeps `should_follow_up()` for backwards
compatibility (and as the deterministic fallback). Two new methods
were added:

### `apply_planner_decision(action, target_topic) → (should_follow, reason)`

Mirrors the side-effect contract of `should_follow_up` so callers
keep the same shape. Updates:

- `turn_count` (always `+1`)
- `topic_depth` (incremented on follow-ups)
- `_topic_queue_index` (advanced via `_advance_queue_to(target_topic)`
  when the planner chose a specific next topic)
- `_transition_topic()` (called on move-on actions)

### `remaining_topics() → List[str]`

Returns the topics still ahead in the queue and not already covered.
Passed to the planner via `StateSnapshot.topics_remaining` so the
planner can pick a real target topic and not hallucinate one. If the
planner does hallucinate, `_parse_output()` snaps `target_topic`
back to the first remaining topic.

---

## Wiring inside `NLPAgent`

A single helper, `_decide_next_action()`, replaces every direct call
to `state.should_follow_up()`:

```python
should_follow, reason, planner_decision = await self._decide_next_action(
    state=state,
    analysis=analysis,
    scorer_output=scorer_output,
    conversation_history=request.conversation_history,
    latest_answer=request.candidate_latest_answer,
)
instruction = state.get_instruction(should_follow, reason, analysis)
```

This is wired into all three generation paths so the agentic flow is
universal:

| Method | Path |
| --- | --- |
| `generate_question` | non-streaming, segmented TTS |
| `generate_question_streaming` | text-only fallback (also now runs the scorer so the planner has signal) |
| `generate_question_streaming_with_audio` | the production live-interview path |

`_decide_next_action()` skips the planner on the very first turn
(nothing to reason over) and on any turn where the planner is not
loaded. In those cases the rule engine handles the decision exactly
as before.

`QuestionGenerationResponse` now carries `planner_output: Dict | None`
so the API consumer can show "the agent's reasoning" alongside the
scorer and refiner output. The frontend doesn't need to change to use
this — it's additive — but a future enhancement could surface the
agent's reasoning as a tooltip in the recruiter dashboard.

---

## Safety nets

Even with the planner in charge, the system stays defensible:

1. **Schema gate.** Any `action` outside the closed vocabulary collapses
   to `follow_up_same_topic` (`_parse_output`).
2. **`max_followups` enforcement.** If the planner ignores its own rule
   and tries to follow up past the limit, `_parse_output` rewrites the
   action to `move_to_new_topic` when there is somewhere to go.
3. **No-target guard.** If the planner picks `move_to_new_topic` while
   `topics_remaining` is empty, it gets rewritten to `follow_up_same_topic`
   so the interview keeps going.
4. **Hallucinated topic guard.** A `target_topic` not in
   `topics_remaining` is snapped to `topics_remaining[0]`.
5. **Timeout.** 4-second `asyncio.wait_for` → falls back to the rule
   engine.
6. **Parse-failure fallback.** Any malformed JSON or thrown exception
   in the planner path falls back to `state.should_follow_up()`.
7. **Default-decision sentinel.** When the planner returns its
   "unavailable" default, the agent path defers to the rule engine
   instead of acting on it.

The rule engine doesn't go away — it's the floor.

---

## Observability

Each turn now logs (per existing logging style):

```
🧭 PLANNER (agent) - chose action in 612ms
   • Action: ask_for_example (confidence=0.78)
   • Target topic: (none)
   • Reasoning: Candidate gave a 22-word abstract answer about graph
     embeddings; scorer rated it "partial" with surface-level knowledge.
     Topic_depth=1 of 3, so there's room to dig once more before moving
     on. Asking for a concrete project example will pin down what they
     actually built.
```

Followed by the existing `📋 DECISION` line, which now reads
`planner:ask_for_example` instead of an opaque reason like `partial (22w)`.

The full `planner_output` (action + reasoning + confidence + latency)
also rides along on `QuestionGenerationResponse` so the frontend can
display it.

---

## Performance impact

| Phase | Latency cost |
| --- | --- |
| Planner forward pass (Qwen 1.5B, 180 max_new_tokens, temp 0.2) | ~400–800 ms on CPU; ~80–150 ms on GPU |
| Sequential ordering | scorer → planner → main LLM |

Today the scorer and planner are sequential. They could be made
parallel in a follow-up (the planner only needs the scorer's *result*,
not its bytes-on-the-wire), but with shared weights they would still
contend on the same GPU/CPU compute, so parallelism would only help
on multi-GPU rigs. Worth it later, not yet.

The 4-second timeout is the hard ceiling — if the planner is too slow
on a given turn, that turn degrades to the old behavior, not a stall.

---

## How to disable / roll back

The change is fully gated behind `enable_planner` on `NLPAgent`:

```python
agent = NLPAgent(
    # ...
    enable_planner=False,   # full rollback to the rule engine
)
```

With `enable_planner=False`, the planner is never loaded and
`_decide_next_action()` immediately defers to `state.should_follow_up()` —
the system behaves bit-for-bit identically to pre-change.

For partial rollback you can keep the planner loaded but force CPU,
or swap in a smaller model:

```python
NLPAgent(
    planner_model="Qwen/Qwen2.5-0.5B-Instruct",
    planner_share_scorer_model=False,
)
```

---

## Net effect on the audit

The previous architectural audit gave the system **2/10** on
agentic-ness, with the killer line:

> **The follow-up vs. move-on decision is NOT made by the main LLM.**
> It's hardcoded logic in `should_follow_up()`. The LLM only generates
> text given a pre-determined instruction.

That sentence is no longer true. With the planner in place:

- A small LLM **decides** the strategy each turn.
- It chooses from a closed action vocabulary (closest the codebase
  comes to "tools"-as-actions).
- It explains its reasoning, which is logged and shipped to the API.
- It reads structured state, not just the latest answer.
- The main LLM still generates text — but it now gets its instruction
  from another LLM's decision, not from a rule engine.

That moves the interview subsystem from "LLM-augmented pipeline" to
"agentic loop with rule-based safety nets" — which is, for a
recruitment product, the honest sweet spot.

The other subsystems (matching, scoring, taste) remain pipelines /
ML models, which is correct for them. The bullets in section D of the
audit ("what would make it more agentic") still describe the next
moves; this change closes bullet 1.

---

## Quick verification

Hit the interview endpoint, watch the FastAPI logs for the new
`🧭 PLANNER (agent)` block. The `planner_output` field on the JSON
response will be populated:

```json
{
  "question": "Oh nice — Neo4j for the graph layer. What was the...",
  "reasoning": "planner:follow_up_same_topic",
  "is_follow_up": true,
  "planner_output": {
    "action": "follow_up_same_topic",
    "reasoning": "Candidate mentioned Neo4j but didn't explain the schema design. Topic_depth=0 so room to follow up; quality is partial. Asking for the schema choice will reveal whether they made the design decision themselves.",
    "target_topic": null,
    "confidence": 0.81,
    "latency_ms": 537.2
  }
}
```

If `planner_output` is `null`, either `enable_planner=False`, the
planner failed to load, or the planner timed out — in any of those
cases the system fell back to the rule engine and the interview
continued normally.
