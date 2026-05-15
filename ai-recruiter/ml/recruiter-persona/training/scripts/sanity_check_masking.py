#!/usr/bin/env python3
"""
Sanity-check the multi-turn loss masking BEFORE launching a long run.

Why this exists:
    DataCollatorForCompletionOnlyLM is silent when its template tokens don't
    match the tokenized text — it just masks everything (so loss is NaN) or
    nothing (so all turns contribute). Both modes look fine until 30 minutes
    into training when the eval curve goes flat or explodes.

What it does:
    1. Loads the tokenizer + a real training sample.
    2. Tokenizes the sample with the chat template, exactly like the data loader.
    3. Builds the collator with `instruction_template` + `response_template`.
    4. Prints each token with its mask state (• masked / L loss-bearing).
    5. Aggregates: % of tokens contributing to loss, and what % of those land
       inside assistant turns vs user/system turns. The number that matters:
       "loss-bearing tokens within USER content" must be 0%.

Run:
    cd ml/recruiter-persona/training
    python scripts/sanity_check_masking.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Tuple

import yaml
from transformers import AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM


HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE.parent / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def first_training_sample(path: Path) -> dict:
    with open(path, "r") as f:
        return json.loads(f.readline())


def role_spans(messages: List[dict], rendered: str) -> List[Tuple[str, int, int]]:
    """Walk through the rendered chat-template string and return
    (role, char_start, char_end) spans of each turn's CONTENT.

    Used purely for the audit print at the end.
    """
    spans = []
    cursor = 0
    for m in messages:
        content = m.get("content", "")
        if not content:
            continue
        idx = rendered.find(content, cursor)
        if idx == -1:
            # Whitespace collapse can move it slightly; fall back to a generous search
            idx = rendered.find(content[:40], cursor)
        if idx == -1:
            continue
        spans.append((m["role"], idx, idx + len(content)))
        cursor = idx + len(content)
    return spans


def main() -> int:
    cfg = load_config()
    model_name = cfg["base_model"]
    train_path = (HERE.parent / cfg["train_data"]).resolve()

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Reading first sample from: {train_path}")
    sample = first_training_sample(train_path)
    messages = sample["messages"]
    print(f"Sample has {len(messages)} messages: roles = {[m['role'] for m in messages]}")

    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    print(f"Rendered conversation: {len(rendered)} chars")

    # Build the collator with the FIXED multi-turn templates
    instruction_template = "<|start_header_id|>user<|end_header_id|>"
    response_template    = "<|start_header_id|>assistant<|end_header_id|>"
    instruction_template_ids = tokenizer.encode(instruction_template, add_special_tokens=False)
    response_template_ids    = tokenizer.encode(response_template,    add_special_tokens=False)

    collator = DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template_ids,
        response_template=response_template_ids,
        tokenizer=tokenizer,
        mlm=False,
    )

    encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"][0]
    batch = collator([{"input_ids": input_ids.tolist()}])
    labels = batch["labels"][0]

    n_total = int(labels.numel())
    n_loss = int((labels != -100).sum().item())
    pct_loss = 100.0 * n_loss / n_total
    print()
    print("=" * 70)
    print(f"Mask summary: {n_loss}/{n_total} tokens contribute to loss ({pct_loss:.1f}%)")
    print("=" * 70)

    # Build a char-level role map for ATTRIBUTION of the loss-bearing tokens.
    spans = role_spans(messages, rendered)
    char_role = [None] * len(rendered)
    for role, s, e in spans:
        for i in range(s, min(e, len(char_role))):
            char_role[i] = role

    # Map each token to a char position via offsets — if offsets aren't
    # available, fall back to running over decode lengths.
    offsets = tokenizer(rendered, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]

    bucket = {"system": 0, "user": 0, "assistant": 0, "other": 0}
    for tok_idx, (a, b) in enumerate(offsets):
        if labels[tok_idx].item() == -100:
            continue
        # Pick the role this token belongs to using its first char position
        if a < len(char_role) and char_role[a] is not None:
            bucket[char_role[a]] += 1
        else:
            bucket["other"] += 1

    print("\nLoss-bearing tokens by source role:")
    for role, n in bucket.items():
        pct = 100.0 * n / max(1, n_loss)
        print(f"  {role:>9}: {n:>5}  ({pct:5.1f}%)")

    # The acceptance criterion
    print("\n" + "=" * 70)
    if bucket["user"] == 0 and bucket["assistant"] > 0:
        print("✅ MASKING OK — loss only over assistant turns.")
        print("=" * 70)
        return 0
    if bucket["user"] > 0:
        print(f"❌ MASKING BROKEN — {bucket['user']} loss-bearing tokens land")
        print("   inside USER (candidate) turns. The model would learn to")
        print("   roleplay the candidate. Check the instruction_template wiring.")
        print("=" * 70)
        return 2
    if bucket["assistant"] == 0:
        print("❌ MASKING OVER-AGGRESSIVE — no tokens contribute to loss.")
        print("   Templates probably didn't match the tokenized text.")
        print("=" * 70)
        return 3
    print("⚠️  Unexpected mask distribution — inspect manually.")
    print("=" * 70)
    return 1


if __name__ == "__main__":
    sys.exit(main())
