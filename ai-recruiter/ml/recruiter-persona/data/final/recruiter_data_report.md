# Recruiter Dataset Analysis and Preprocessing Report

## Summary
- Input rows: **10174**
- Parsed transcripts: **10144**
- Kept after cleaning + dedup: **8629** (84.81%)
- Train / Eval: **7749 / 880**

## Cleaning Rules Applied
- Parsed speaker turns and mapped `Interviewer` to assistant (`gpt`) and candidate to user (`human`).
- Removed stage-direction labels (`Interview Scene`, `Interview Setting`, etc.).
- Enforced alternating dialogue, minimum turn count, and interviewer-question presence.
- Removed technical quiz-style interviewer prompts to preserve recruiter persona.
- Filtered potentially illegal/sensitive interview question patterns.
- Removed highly templated repetitive follow-ups and weak behavioral-signal dialogues.
- Deduplicated exact and near-duplicate conversations.

## Top Filter Reasons
- technical_question_style: 908
- very_short_turn: 448
- too_few_turns: 169
- insufficient_role_balance: 169
- very_long_turn: 88
- weak_behavioral_signal: 77
- potentially_illegal_or_sensitive_prompt: 63
- empty_conversation: 30
- too_many_turns: 10
- no_interviewer_question: 2

## Observations
- The source is structurally rich and mostly parseable into multi-turn interview dialogue.
- There is substantial template reuse in interviewer phrasing.
- Decision labels are near-balanced (`select` vs `reject`), useful for optional downstream evaluation.
- Role naming is inconsistent in capitalization (for example, `Software Engineer` vs `software engineer`).

## Limitations and Risks
- The dataset appears partially synthetic/template-generated; style diversity may be narrower than real interviews.
- Short `Reason_for_decision` fields limit explainability signal if used for reward or critique tuning.
- Candidate responses may include grammatical noise and can transfer informal patterns to the model.
- Parsing assumes colon-prefixed speaker format; unusual formatting may be dropped.
- Even after filtering, hidden bias can remain in role-specific language and selection outcomes.

## Output Files
- `data_v2/final/recruiter_llama_train_messages.jsonl`
- `data_v2/final/recruiter_llama_eval_messages.jsonl`
- `data_v2/final/recruiter_llama_train_sharegpt.jsonl`
- `data_v2/final/recruiter_llama_eval_sharegpt.jsonl`
- `data_v2/final/recruiter_data_stats.json`