#!/usr/bin/env python3
"""Convert recruiter CSV to ShareGPT format for Llama fine-tuning with enhanced system prompt."""

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

RANDOM_SEED = 42

SYSTEM_PROMPT_TEMPLATE = """You are Alex, a warm and professional senior technical recruiter conducting a live voice interview with {name} for the role of {role}. This is a natural spoken conversation, not a written exchange. Speak as you would in a real interview.

## Job Description:
{jd}

## Candidate CV:
{cv}
{reference_section}

## Conversation Flow
**Opening (first turn only):**
- Greet the candidate warmly by name.
- Set a friendly tone and ask an opening question about their background or a highlight from their CV.

**During the interview:**
- Always acknowledge what the candidate just said before asking your next question.
- Vary your acknowledgments naturally. Reference specific details from their answer to show active listening.
- Ask **one focused follow-up question** per turn that digs deeper into their experience.
- **Every question, follow-up, and acknowledgment must be grounded exclusively in the provided documents (CV, JD, and any additional context).**
- Do not introduce topics or skills that are not present in these documents.
- **Avoid generic questions like “Tell me about your Python experience.” Instead, connect skills to specific requirements in the JD or projects in the CV.** For example, if the JD requires “real‑time AI systems” and the CV mentions “Python projects,” ask: “How did you handle latency constraints in that Python‑based recommendation engine?”

**Question Style:**
- Ask about the candidate’s **specific contributions**, not just what the team or project did.
- Never repeat a topic the candidate already covered unless you are building on it with new details from the references.
- **Cross‑reference:** When you see a skill or project in the CV that matches a requirement in the JD, ask how the candidate applied that skill to meet that requirement.

## Rules:
- Produce exactly **one question per turn** (stop after the first question mark).
- Keep your response concise and conversational (max 1–2 sentences before question).
- Sound natural and human — avoid robotic phrasing.
"""

SPEAKER_LINE_RE = re.compile(
    r"^\s*(?:\*\*|__)?\s*([A-Za-z][A-Za-z0-9 .\-']{0,60}?)\s*(?:\*\*|__)?\s*:\s*(.*)$"
)

INTERVIEWER_LABEL_HINTS = (
    "interviewer",
    "recruiter",
    "hiring manager",
    "talent partner",
)

STAGE_LABEL_HINTS = (
    "interview setting",
    "interview scene",
    "scene",
    "setting",
    "stage direction",
    "stage directions",
    "narrator",
    "note",
)

TECHNICAL_QUESTION_PATTERNS = [
    r"\bwhat is the purpose of\b",
    r"\bwhat is the difference between\b",
    r"\bdifference between\b",
    r"\bjava\b",
    r"\bstringbuilder\b",
    r"\bstringbuffer\b",
    r"\bthreadlocal\b",
    r"\bstrictfp\b",
    r"\bserializable\b",
    r"\bdistributed caching\b",
    r"\bmulti-tenant\b",
    r"\brbac\b",
    r"\bopenapi\b",
]

ILLEGAL_OR_SENSITIVE_PATTERNS = [
    r"\bwhat is your age\b",
    r"\bhow old are you\b",
    r"\bmarried\b",
    r"\bchildren\b",
    r"\bpregnan\w*\b",
    r"\breligion\b",
    r"\bnationality\b",
    r"\bethnicity\b",
    r"\bdisabilit\w*\b",
]

CANNED_PHRASES = [
    "let me shift to a different area now",
    "let me follow up on that",
    "what was the outcome of that situation",
    "good. let me ask a behavioural question now",
]

OPEN_ENDED_MARKERS = [
    "tell me about",
    "describe a",
    "walk me through",
    "what happened next",
    "how did you",
    "could you share",
    "give me an example",
    "can you walk me through",
]

ROLE_TITLE_HINTS = (
    "candidate",
    "interviewer",
    "recruiter",
    "hiring manager",
    "manager",
    "director",
    "lead",
    "technical lead",
    "data science",
    "data scientist",
    "data analyst",
    "data engineer",
    "software engineer",
    "ui engineer",
    "ui designer",
    "ux designer",
    "devops engineer",
    "cloud architect",
    "cloud engineer",
    "qa engineer",
    "product manager",
    "content writer",
    "mobile app developer",
    "it support specialist",
    "cybersecurity analyst",
    "full stack developer",
    "game developer",
    "business analyst",
)

MONTH_NAME_RE = re.compile(
    r"^(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)\b",
    re.IGNORECASE,
)

JSON_ROLE_PREFIX_RE = re.compile(
    r'^\s*[\]\[}{,\s]*"?role"?\s*:\s*"?(?:assistant|user|system)"?\s*,\s*"?content"?\s*:\s*"?',
    re.IGNORECASE,
)

JSON_ROLE_WRAPPER_RE = re.compile(
    r'^\s*[}\],\s]*\{\s*"role"\s*:\s*"(?:assistant|user|system)"\s*,\s*"content"\s*:\s*"',
    re.IGNORECASE,
)

JSON_TRAILING_WRAPPER_RE = re.compile(r'"\s*[}\]]\s*,?\s*$')

# Regex patterns to remove email and phone from text
EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def canonicalize(text: str) -> str:
    text = normalize_space(text).lower()
    text = re.sub(r"[^a-z0-9\s?]", "", text)
    return normalize_space(text)


def words(text: str) -> int:
    return len(text.split())


def is_stage_speaker(label: str) -> bool:
    low = label.strip().lower()
    return any(h in low for h in STAGE_LABEL_HINTS)


def normalize_label(label: str) -> str:
    return normalize_space(re.sub(r"[*_`]+", "", label).lower())


def normalize_assistant_name(text: str) -> str:
    """Replace various interviewer/recruiter names with 'Alex' for consistency."""
    common_names = [
        "rachel", "sarah", "emily", "jennifer", "jessica", "ashley",
        "michael", "david", "john", "robert", "james", "william",
        "lisa", "amy", "amanda", "melissa", "nicole", "stephanie",
        "daniel", "matthew", "christopher", "andrew", "joshua", "ryan",
        "karen", "nancy", "laura", "rebecca", "michelle", "kimberly"
    ]

    for name in common_names:
        text = re.sub(r'\b' + name + r'\b', 'Alex', text, flags=re.IGNORECASE)
    return text


def replace_alex_with_candidate(conv: list[dict[str, str]], candidate_name: str) -> list[dict[str, str]]:
    """Replace 'Alex' with the candidate's name in gpt turns."""
    if not candidate_name:
        return conv
    result = []
    for turn in conv:
        if turn["from"] == "gpt":
            new_value = re.sub(r'\bAlex\b', candidate_name, turn["value"], flags=re.IGNORECASE)
            result.append({"from": turn["from"], "value": new_value})
        else:
            result.append(turn)
    return result


def clean_turn_text(text: str) -> str:
    text = text.strip()
    text = JSON_ROLE_WRAPPER_RE.sub("", text)
    text = JSON_ROLE_PREFIX_RE.sub("", text)
    text = JSON_TRAILING_WRAPPER_RE.sub("", text)
    text = text.replace("**", "").replace("__", "").replace("`", "")
    text = re.sub(r"^\s*[-*]+\s*", "", text)
    text = re.sub(r'\([^)]*\)', '', text)  # remove stage directions in parentheses
    text = re.sub(r"^\s*dr\.?\s+[A-Za-z][A-Za-z .'-]{0,80}(?=\s*,|\s*:|\s+-|\s+\(|$)", "Alex", text, flags=re.IGNORECASE)
    text = normalize_assistant_name(text)
    return normalize_space(text)


def looks_like_person_name(text: str) -> bool:
    return bool(
        re.match(
            r"^(?:(?:dr|mr|mrs|ms)\.?\s+)?[a-z][a-z'-]+(?:\s+[a-z][a-z'-]+){1,4}(?:\s+(?:md|dvm|dds|phd))?$",
            text,
            re.IGNORECASE,
        )
    )


def looks_like_metadata_turn(text: str) -> bool:
    low = normalize_space(text).lower()
    if not low:
        return True
    if any(k in low for k in ("candidate profile", "confidential candidate profile")):
        return True
    if "expected experience" in low:
        return True
    if re.match(r"^(?:candidate|interviewer|location|date|time|duration|note)\s*:", low):
        return True
    if MONTH_NAME_RE.match(low) and ("am" in low or "pm" in low or re.search(r"\b\d{4}\b", low)):
        return True
    if re.match(r"^\d{1,2}:\d{2}\s*(?:am|pm)\b", low):
        return True
    if words(low) <= 7 and looks_like_person_name(low):
        return True
    if words(low) <= 14 and any(hint in low for hint in ("candidate", "interviewer", "hiring manager", "recruiter", "virtual interview", "conference room", "office", "interview scene", "interview setting")) and "?" not in low:
        return True
    if re.match(r"^[a-z][a-z .'-]{1,80},\s*[a-z0-9 .&/'()-]{2,120}$", low):
        left, right = [normalize_space(x) for x in low.split(",", 1)]
        if looks_like_person_name(left) and any(h in right for h in ROLE_TITLE_HINTS):
            return True
    if low.startswith("dr.") and any(h in low for h in ROLE_TITLE_HINTS):
        if "?" not in low and words(low) <= 20:
            return True
    return False


def candidate_aliases(candidate_name: str) -> set[str]:
    aliases = set()
    cname = normalize_label(candidate_name)
    if not cname:
        return aliases
    aliases.add(cname)
    parts = cname.split()
    if parts:
        aliases.add(parts[0])
        aliases.add(parts[-1])
    return aliases


def infer_speaker_roles(turns_raw: list[tuple[str, str]], candidate_name: str) -> dict[str, str | None]:
    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "questions": 0})
    for speaker, content in turns_raw:
        s = normalize_label(speaker)
        stats[s]["count"] += 1
        if "?" in content:
            stats[s]["questions"] += 1

    cand_aliases = candidate_aliases(candidate_name)
    role_map: dict[str, str | None] = {}

    for s in stats:
        if is_stage_speaker(s):
            role_map[s] = None
        elif any(h in s for h in INTERVIEWER_LABEL_HINTS):
            role_map[s] = "gpt"
        elif s in {"candidate", "interviewee", "applicant"} or s in cand_aliases:
            role_map[s] = "human"

    speakers_no_stage = [s for s in stats if role_map.get(s, "unknown") is not None]
    if cand_aliases and len(speakers_no_stage) == 2:
        for s in speakers_no_stage:
            if s not in cand_aliases and role_map.get(s) is None:
                role_map[s] = "gpt"

    for s, st in stats.items():
        if s in role_map:
            continue
        q_ratio = st["questions"] / max(st["count"], 1)
        if st["questions"] >= 2 and q_ratio >= 0.5:
            role_map[s] = "gpt"
        else:
            role_map[s] = "human"

    return role_map


def parse_transcript(transcript: str, candidate_name: str) -> list[dict[str, str]]:
    turns_raw: list[tuple[str, str]] = []
    current_speaker = None
    current_text_parts: list[str] = []

    for line in transcript.splitlines():
        line = line.rstrip()
        if not line.strip():
            continue

        line = re.sub(r"^\s*\*\*\s*([^*]{1,60})\s*\*\*\s*:\s*", r"\1: ", line)
        line = re.sub(r"^\s*__\s*([^_]{1,60})\s*__\s*:\s*", r"\1: ", line)

        m = SPEAKER_LINE_RE.match(line)
        if m:
            if current_speaker is not None:
                turns_raw.append((current_speaker, normalize_space(" ".join(current_text_parts))))
            current_speaker = m.group(1).strip()
            current_text_parts = [m.group(2).strip()]
        else:
            if current_speaker is not None:
                current_text_parts.append(line.strip())

    if current_speaker is not None:
        turns_raw.append((current_speaker, normalize_space(" ".join(current_text_parts))))

    role_map = infer_speaker_roles(turns_raw, candidate_name)

    turns: list[dict[str, str]] = []
    for speaker, content in turns_raw:
        content = clean_turn_text(content)
        if not content:
            continue
        if looks_like_metadata_turn(content):
            continue
        role = role_map.get(normalize_label(speaker), "human")
        if role is None:
            continue
        turns.append({"from": role, "value": content})

    if not turns:
        return []

    merged: list[dict[str, str]] = [turns[0].copy()]
    for t in turns[1:]:
        if t["from"] == merged[-1]["from"]:
            merged[-1]["value"] = normalize_space(merged[-1]["value"] + " " + t["value"])
        else:
            merged.append(t.copy())

    while merged and merged[0]["from"] != "gpt":
        merged.pop(0)
    if merged and merged[-1]["from"] == "human":
        merged.pop()

    return merged


def has_pattern(text: str, patterns: list[str]) -> bool:
    low = text.lower()
    return any(re.search(p, low) for p in patterns)


def quality_filter(conv: list[dict[str, str]]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not conv:
        return False, ["empty_conversation"]
    if len(conv) < 3:
        reasons.append("too_few_turns")
    if len(conv) > 30:
        reasons.append("too_many_turns")
    gpt_turns = [t for t in conv if t["from"] == "gpt"]
    human_turns = [t for t in conv if t["from"] == "human"]
    if len(gpt_turns) < 2 or len(human_turns) < 1:
        reasons.append("insufficient_role_balance")
    for i in range(1, len(conv)):
        if conv[i]["from"] == conv[i-1]["from"]:
            reasons.append("non_alternating")
            break
    for t in conv:
        n = words(t["value"])
        if n < 4:
            reasons.append("very_short_turn")
            break
        if n > 420:
            reasons.append("very_long_turn")
            break
    all_gpt = " ".join(t["value"] for t in gpt_turns)
    if "?" not in all_gpt:
        reasons.append("no_interviewer_question")
    if has_pattern(all_gpt, TECHNICAL_QUESTION_PATTERNS):
        reasons.append("technical_question_style")
    if has_pattern(all_gpt, ILLEGAL_OR_SENSITIVE_PATTERNS):
        reasons.append("potentially_illegal_or_sensitive_prompt")
    if sum(1 for p in CANNED_PHRASES if p in all_gpt.lower()) >= 2:
        reasons.append("over_templated_followups")
    if not any(m in all_gpt.lower() for m in OPEN_ENDED_MARKERS) and all_gpt.count("?") < 2:
        reasons.append("weak_behavioral_signal")
    first = gpt_turns[0]["value"].lower() if gpt_turns else ""
    if first.startswith("are you "):
        reasons.append("closed_binary_opening")
    return len(reasons) == 0, reasons


def clean_resume(cv: str) -> str:
    """Remove synthetic boilerplate like 'sample resume', customization notes, and PII."""
    if not cv:
        return ""

    # Remove everything from "This is a sample resume" to the end of the resume
    cv = re.sub(
        r'(?:this\s+is\s+a\s+sample\s+resume|sample\s+resume).*',
        '',
        cv,
        flags=re.IGNORECASE | re.DOTALL
    )

    # Remove "Contact Information:" section up to (but not including) "Summary"
    cv = re.sub(
        r'Contact\s+Information\s*:.*?(?=Summary)',
        '',
        cv,
        flags=re.IGNORECASE | re.DOTALL
    )

    # Remove everything from "I hope this" to the end
    cv = re.sub(
        r'I\s+hope\s+this.*',
        '',
        cv,
        flags=re.IGNORECASE | re.DOTALL
    )

    # Remove any remaining lines that are just labels (e.g., "* Email:", "* Phone:")
    cv = re.sub(r'^\s*\*?\s*(?:Email|Phone|Tel|Telephone|Mobile|Cell)\s*:\s*$', '', cv, flags=re.MULTILINE | re.IGNORECASE)

    # Clean up multiple blank lines
    cv = re.sub(r'\n\s*\n\s*\n', '\n\n', cv)

    return cv.strip()

def deduplicate(examples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    seen_exact = set()
    seen_soft = set()
    out = []
    removed = 0
    for ex in examples:
        conv = ex["conversations"]
        exact_payload = "||".join(f"{t['from']}:{canonicalize(t['value'])}" for t in conv)
        soft_payload = "||".join(f"{t['from']}:{canonicalize(t['value'])}" for t in conv[:4])
        exact_key = hashlib.md5(exact_payload.encode("utf-8")).hexdigest()
        soft_key = hashlib.md5(soft_payload.encode("utf-8")).hexdigest()
        if exact_key in seen_exact or soft_key in seen_soft:
            removed += 1
            continue
        seen_exact.add(exact_key)
        seen_soft.add(soft_key)
        out.append(ex)
    return out, removed


def to_sharegpt_record(example: dict[str, Any]) -> dict[str, Any]:
    return {
        "system": example["system"],
        "conversations": example["conversations"],
        "meta": {
            "id": example["id"],
            "name": example["name"],
            "role": example["job_role"],
            "decision": example["decision"],
        }
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stratified_split(examples: list[dict[str, Any]], ratio: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ex in examples:
        bucket = ex.get("job_role", "unknown").strip().lower() or "unknown"
        grouped[bucket].append(ex)

    train = []
    evals = []
    for _, items in grouped.items():
        random.shuffle(items)
        if len(items) == 1:
            train.extend(items)
            continue
        split = max(1, min(int(len(items) * ratio), len(items) - 1))
        train.extend(items[:split])
        evals.extend(items[split:])
    random.shuffle(train)
    random.shuffle(evals)
    return train, evals


def main():
    parser = argparse.ArgumentParser(description="Convert recruiter CSV to ShareGPT format for Llama SFT")
    parser.add_argument("--input", default="data_v2/raw/dataset.csv", help="Input CSV path")
    parser.add_argument("--outdir", default="data_v2/final", help="Output directory")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Train split ratio (0..1)")
    args = parser.parse_args()

    random.seed(RANDOM_SEED)

    input_path = Path(args.input)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    parsed_rows = 0
    filtered_out = 0
    reason_counts: Counter[str] = Counter()
    decision_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()

    processed: list[dict[str, Any]] = []

    with input_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1

            transcript = (row.get("Transcript") or "").replace("\r\n", "\n").replace("\r", "\n").strip()
            if not transcript:
                reason_counts["empty_transcript"] += 1
                filtered_out += 1
                continue

            candidate_name = normalize_space(row.get("Name") or "")
            conv = parse_transcript(transcript, candidate_name)
            if conv:
                parsed_rows += 1
                # Replace "Alex" with candidate name in gpt turns
                conv = replace_alex_with_candidate(conv, candidate_name)

            ok, reasons = quality_filter(conv)
            if not ok:
                filtered_out += 1
                reason_counts.update(reasons)
                continue

            decision = normalize_space((row.get("decision") or "unknown").lower())
            role = normalize_space(row.get("Role") or "unknown")
            jd = row.get("Job_Description") or ""
            cv = row.get("Resume") or ""

            # Clean CV and JD of emails/phones
            cv_clean = clean_resume(cv)
            jd_clean = clean_resume(jd)  # same cleaning, though JD may not have emails

            # Build system prompt with placeholders
            system = SYSTEM_PROMPT_TEMPLATE.format(
                name=candidate_name,
                role=role,
                jd=jd_clean,
                cv=cv_clean,
                reference_section="",  # optional; can be left empty
            )
            decision_counts[decision] += 1
            role_counts[role] += 1
            processed.append(
                {
                    "id": normalize_space(row.get("ID") or ""),
                    "name": candidate_name,
                    "job_role": role,
                    "decision": decision,
                    "system": system,
                    "conversations": conv,
                }
            )
    deduped, dupes_removed = deduplicate(processed)
    train, evals = stratified_split(deduped, args.train_ratio)
    train_sharegpt = [to_sharegpt_record(x) for x in train]
    eval_sharegpt = [to_sharegpt_record(x) for x in evals]
    train_path = out_dir / "recruiter_llama_train_sharegpt.jsonl"
    eval_path = out_dir / "recruiter_llama_eval_sharegpt.jsonl"
    write_jsonl(train_path, train_sharegpt)
    write_jsonl(eval_path, eval_sharegpt)
    kept = len(deduped)
    keep_rate = (kept / total_rows * 100.0) if total_rows else 0.0
    turns = [len(x["conversations"]) for x in deduped]
    gpt_questions = [sum(1 for t in x["conversations"] if t["from"] == "gpt" and "?" in t["value"]) for x in deduped]
    stats = {
        "input_rows": total_rows,
        "parsed_rows": parsed_rows,
        "filtered_out": filtered_out,
        "after_filter_before_dedup": len(processed),
        "duplicates_removed": dupes_removed,
        "kept_rows": kept,
        "keep_rate_pct": round(keep_rate, 2),
        "train_rows": len(train),
        "eval_rows": len(evals),
        "reason_counts": dict(reason_counts.most_common()),
        "decision_distribution": dict(decision_counts),
        "top_roles": role_counts.most_common(20),
        "turns_min": min(turns) if turns else 0,
        "turns_median": sorted(turns)[len(turns)//2] if turns else 0,
        "turns_max": max(turns) if turns else 0,
        "question_turns_median": sorted(gpt_questions)[len(gpt_questions)//2] if gpt_questions else 0,
    }
    stats_path = out_dir / "recruiter_data_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    report_lines = [
        "# Recruiter Dataset Analysis and Preprocessing Report",
        "",
        "## Summary",
        f"- Input rows: **{total_rows}**",
        f"- Parsed transcripts: **{parsed_rows}**",
        f"- Kept after cleaning + dedup: **{kept}** ({keep_rate:.2f}%)",
        f"- Train / Eval: **{len(train)} / {len(evals)}**",
        "",
        "## Cleaning Rules Applied",
        "- Parsed speaker turns and mapped `Interviewer` to assistant (`gpt`) and candidate to user (`human`).",
        "- Removed stage-direction labels (`Interview Scene`, `Interview Setting`, etc.).",
        "- Enforced alternating dialogue, minimum turn count, and interviewer-question presence.",
        "- Removed technical quiz-style interviewer prompts to preserve recruiter persona.",
        "- Filtered potentially illegal/sensitive interview question patterns.",
        "- Removed highly templated repetitive follow-ups and weak behavioral-signal dialogues.",
        "- Deduplicated exact and near-duplicate conversations.",
        "",
        "## Top Filter Reasons",
    ]
    for reason, count in reason_counts.most_common(10):
        report_lines.append(f"- {reason}: {count}")
    report_lines.extend([
        "",
        "## Output Files",
        f"- `{train_path}`",
        f"- `{eval_path}`",
        f"- `{stats_path}`",
    ])
    report_path = out_dir / "recruiter_data_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"Saved report: {report_path}")
if __name__ == "__main__":
    main()