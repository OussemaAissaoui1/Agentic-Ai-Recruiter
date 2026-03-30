# Recruiter Dataset Analysis and Preprocessing Report

## Summary
- Input rows: **10174**
- Parsed transcripts: **10144**
- Kept after cleaning + dedup: **8638** (84.90%)
- Train / Eval: **7758 / 880**

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
- very_short_turn: 438
- too_few_turns: 169
- insufficient_role_balance: 169
- very_long_turn: 88
- weak_behavioral_signal: 77
- potentially_illegal_or_sensitive_prompt: 63
- empty_conversation: 30
- too_many_turns: 10
- no_interviewer_question: 2

## Output Files
- `/home/oussema/Desktop/data/f2/recruiter_llama_train_sharegpt.jsonl`
- `/home/oussema/Desktop/data/f2/recruiter_llama_eval_sharegpt.jsonl`
- `/home/oussema/Desktop/data/f2/recruiter_data_stats.json`