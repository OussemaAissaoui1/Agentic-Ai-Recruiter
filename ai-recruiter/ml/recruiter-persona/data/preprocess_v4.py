#!/usr/bin/env python3
"""
v4 data preprocessing — runs ON TOP of the existing final/ JSONLs.

Layered cleaning (we trust the v3 pipeline that produced final/ but add the
v4-specific fixes the audit caught):

  1. Drop conversations whose final assistant message ends with '?'
     (interview was cut off mid-question; Alex never learns to close)
  2. Drop conversations where the FIRST assistant message introduces a name
     other than "Alex" (~0.4% of data — directly contaminates persona)
  3. Drop conversations with stage-direction / speaker-label leakage in
     assistant turns (e.g. "Interviewer:", "Recruiter:", "**Note:**")
  4. Drop conversations with extremely short assistant turns (< 4 words)
     anywhere — those are usually "Okay." / "Got it." filler that adds no
     signal but does add tokens
  5. Normalize casing: identify whether the assistant turns are
     predominantly lowercase or properly capitalized, and rewrite the few
     mixed-style outliers in EACH conversation to match the dominant style.
     Keeps natural variation across conversations but stops within-conv drift.

Outputs:
  data/final/recruiter_llama_train_messages_v4.jsonl
  data/final/recruiter_llama_eval_messages_v4.jsonl
  data/final/preprocess_v4_stats.json

Usage:
  cd ml/recruiter-persona/data
  python preprocess_v4.py

Then point config.yaml's `train_data` / `eval_data` at the v4 files.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

HERE = Path(__file__).resolve().parent
FINAL = HERE / "final"

IN_TRAIN  = FINAL / "recruiter_llama_train_messages.jsonl"
IN_EVAL   = FINAL / "recruiter_llama_eval_messages.jsonl"
OUT_TRAIN = FINAL / "recruiter_llama_train_messages_v4.jsonl"
OUT_EVAL  = FINAL / "recruiter_llama_eval_messages_v4.jsonl"
OUT_STATS = FINAL / "preprocess_v4_stats.json"

# Speaker-label leakage we never want the assistant to emit.
SPEAKER_LABEL_RE = re.compile(
    r"\b(interviewer|recruiter|candidate|user|assistant)\s*:",
    re.IGNORECASE,
)

# Catches "Ramesh, Project Management Lead" style intros where the model
# introduces a name other than Alex in the first assistant message.
NAME_INTRO_RE = re.compile(
    r"^\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?),\s+[A-Z][a-zA-Z\s]{3,}\b"
)


def _word_count(s: str) -> int:
    return len(s.split())


def _is_predominantly_lowercase(messages: List[Dict]) -> bool:
    """Look at the assistant turns and decide whether the conversation is
    in "all lowercase" style or "properly capitalized" style. The signal: do
    sentence starts begin with lowercase letters more than half the time?"""
    starts = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for sent in re.split(r"(?<=[.!?])\s+", m.get("content", "").strip()):
            sent = sent.strip()
            if sent and sent[0].isalpha():
                starts.append(sent[0])
    if not starts:
        return False
    lowers = sum(1 for c in starts if c.islower())
    return lowers / len(starts) > 0.5


def _normalize_casing(content: str, target_lowercase: bool) -> str:
    """Lightweight normalization: capitalize sentence starts when target is
    'capitalized', or lowercase first letters when target is 'lowercase'.
    Preserves the rest of the sentence as-is (so proper-noun handling is
    minimally affected)."""
    parts = re.split(r"(?<=[.!?])\s+", content)
    fixed: List[str] = []
    for p in parts:
        if not p:
            continue
        first_alpha = next((i for i, ch in enumerate(p) if ch.isalpha()), None)
        if first_alpha is None:
            fixed.append(p)
            continue
        head = p[:first_alpha]
        ch = p[first_alpha]
        tail = p[first_alpha + 1:]
        if target_lowercase:
            ch = ch.lower()
        else:
            ch = ch.upper()
        fixed.append(head + ch + tail)
    out = " ".join(fixed)
    return out


def _filter_sample(sample: Dict, reasons: Counter) -> Tuple[Dict, bool]:
    """Apply per-conversation filters. Returns (sample, kept_flag).

    The sample may be MODIFIED (casing normalization) before being returned.
    """
    msgs = sample.get("messages", [])
    if not msgs:
        reasons["empty"] += 1
        return sample, False

    asst_turns = [m for m in msgs if m.get("role") == "assistant"]
    if len(asst_turns) < 2:
        reasons["too_few_assistant_turns"] += 1
        return sample, False

    # (1) Truncated mid-question
    last_asst = asst_turns[-1].get("content", "").rstrip()
    if last_asst.endswith("?"):
        reasons["truncated_at_question"] += 1
        return sample, False

    # (2) Name drift in first assistant message
    first_asst = asst_turns[0].get("content", "").lstrip()
    m = NAME_INTRO_RE.match(first_asst)
    if m and m.group(1).lower() != "alex":
        reasons["non_alex_name_intro"] += 1
        return sample, False

    # (3) Speaker-label leakage in any assistant turn
    leaked = False
    for t in asst_turns:
        c = t.get("content", "")
        if SPEAKER_LABEL_RE.search(c):
            leaked = True
            break
    if leaked:
        reasons["speaker_label_leak"] += 1
        return sample, False

    # (4) Tiny assistant turns — "Okay.", "Great." — add no signal
    if any(_word_count(t.get("content", "")) < 4 for t in asst_turns):
        reasons["assistant_turn_lt_4_words"] += 1
        return sample, False

    # (5) Casing normalization — pick the dominant style and fix outlier
    # sentences within the assistant turns. (User turns are untouched — they're
    # masked out by the collator and won't affect the model.)
    target_lowercase = _is_predominantly_lowercase(msgs)
    new_messages = []
    for m in msgs:
        if m.get("role") == "assistant":
            new_messages.append({
                **m,
                "content": _normalize_casing(m["content"], target_lowercase),
            })
        else:
            new_messages.append(m)
    sample = {**sample, "messages": new_messages}

    reasons["kept"] += 1
    return sample, True


def run(in_path: Path, out_path: Path, reasons: Counter) -> Tuple[int, int]:
    n_in = n_out = 0
    with open(in_path, "r", encoding="utf-8") as fi, open(out_path, "w", encoding="utf-8") as fo:
        for line in fi:
            n_in += 1
            try:
                s = json.loads(line)
            except json.JSONDecodeError:
                reasons["bad_json"] += 1
                continue
            kept_sample, keep = _filter_sample(s, reasons)
            if keep:
                fo.write(json.dumps(kept_sample, ensure_ascii=False) + "\n")
                n_out += 1
    return n_in, n_out


def main():
    if not IN_TRAIN.exists() or not IN_EVAL.exists():
        raise SystemExit(
            f"Missing input files. Expected:\n  {IN_TRAIN}\n  {IN_EVAL}"
        )

    train_reasons: Counter = Counter()
    eval_reasons: Counter = Counter()

    print(f"Processing TRAIN → {OUT_TRAIN.name}")
    n_in_t, n_out_t = run(IN_TRAIN, OUT_TRAIN, train_reasons)
    print(f"  kept {n_out_t}/{n_in_t} ({100*n_out_t/max(1,n_in_t):.1f}%)")

    print(f"\nProcessing EVAL → {OUT_EVAL.name}")
    n_in_e, n_out_e = run(IN_EVAL, OUT_EVAL, eval_reasons)
    print(f"  kept {n_out_e}/{n_in_e} ({100*n_out_e/max(1,n_in_e):.1f}%)")

    stats = {
        "train": {
            "input":  n_in_t,
            "output": n_out_t,
            "keep_rate_pct": round(100 * n_out_t / max(1, n_in_t), 2),
            "reasons": dict(train_reasons),
        },
        "eval": {
            "input":  n_in_e,
            "output": n_out_e,
            "keep_rate_pct": round(100 * n_out_e / max(1, n_in_e), 2),
            "reasons": dict(eval_reasons),
        },
    }
    OUT_STATS.write_text(json.dumps(stats, indent=2))
    print(f"\nStats written to {OUT_STATS}")

    print("\n=== Filter reason summary (train) ===")
    for r, n in train_reasons.most_common():
        print(f"  {r:32s} {n}")

    print("\nDone. Next step:")
    print(f"  In config.yaml, set:")
    print(f"    train_data: \"../data/final/{OUT_TRAIN.name}\"")
    print(f"    eval_data:  \"../data/final/{OUT_EVAL.name}\"")


if __name__ == "__main__":
    main()
