"""Split JDs into ability-requirement lines and resumes into experience blocks.

Qin et al. (TAPJFNN) treat requirements / duties as individual bullet lines
and experiences as paragraph-sized blocks (one per job / project / award).
Boundaries come from the document's own structural line breaks.
"""

from __future__ import annotations

import re
from typing import List

_BULLET = re.compile(r"^\s*[•\-\*●◦▪·⁃∙\d]+[\.)\]]?\s*")
_SPLIT_SENT = re.compile(r"(?<=[\.!?])\s+(?=[A-Z])")
_HEADER = re.compile(
    r"^(experience|work experience|professional experience|employment|"
    r"projects|education|academic|skills|achievements|awards|"
    r"publications|certifications|summary|objective|contact)[:\-\s]*$",
    re.IGNORECASE,
)


def _strip_bullet(line: str) -> str:
    return _BULLET.sub("", line).strip()


def split_requirements(text: str, max_items: int = 18, max_chars: int = 400) -> List[str]:
    """Return a list of ability requirements / duties from a job description.

    A requirement is a single bullet or sentence. Long paragraphs are split on
    sentence boundaries so each entry stays sentence-length.
    """
    if not text:
        return []
    text = text.replace("\r", "\n")
    raw_lines = [ln.strip() for ln in text.split("\n")]
    items: List[str] = []
    for ln in raw_lines:
        if not ln:
            continue
        if _HEADER.match(ln):
            continue
        stripped = _strip_bullet(ln)
        if len(stripped) < 3:
            continue
        if len(stripped) > max_chars or not stripped.startswith(("-", "*", "•")):
            # Split long lines on sentence boundaries
            for sent in _SPLIT_SENT.split(stripped):
                sent = sent.strip()
                if len(sent) >= 3:
                    items.append(sent[:max_chars])
        else:
            items.append(stripped[:max_chars])
    # De-duplicate while preserving order
    seen = set()
    unique: List[str] = []
    for s in items:
        key = s.lower()
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique[:max_items]


def split_experiences(text: str, max_items: int = 18, max_chars: int = 1800) -> List[str]:
    """Return experience blocks from a resume.

    Blocks are separated by blank lines. Each block is a single work / project
    / education entry. If the resume has no blank lines we fall back to
    per-line splitting.
    """
    if not text:
        return []
    text = text.replace("\r", "\n")
    # Remove known headers
    cleaned: List[str] = []
    for ln in text.split("\n"):
        if _HEADER.match(ln.strip()):
            cleaned.append("")  # force a block boundary
        else:
            cleaned.append(ln.rstrip())
    blocks: List[str] = []
    buf: List[str] = []
    for ln in cleaned:
        if ln.strip() == "":
            if buf:
                blocks.append(" ".join(s.strip() for s in buf if s.strip()))
                buf = []
        else:
            buf.append(ln)
    if buf:
        blocks.append(" ".join(s.strip() for s in buf if s.strip()))
    if len(blocks) <= 1:
        # No paragraph structure; split on double-space or bullet lines
        blocks = [b.strip() for b in re.split(r"\s{2,}|\n", text) if b.strip()]
    blocks = [b[:max_chars] for b in blocks if len(b) > 10]
    return blocks[:max_items]
