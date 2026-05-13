"""Post-draft bias scan.

A small wordlist catches the most common gender-coded and age-coded
language without round-tripping every JD through the LLM. The heuristic
runs sub-millisecond. Anything ambiguous (no hits, or only weak hits) can
optionally be re-checked via a Groq pass — that's wired into the runner.

Wordlists adapted from the well-known Gaucher et al. (2011) gender-coded
language set plus accumulated industry practice. We don't claim exhaustive
coverage; the rejection-feedback loop catches everything the wordlist
misses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

# ---------------------------------------------------------------------------
# Coded-language buckets — keep entries lowercase, match as word boundaries
# ---------------------------------------------------------------------------
MASCULINE_CODED = [
    "aggressive", "ambitious", "analytical", "assertive", "autonomous",
    "boast", "challenging", "competitive", "confident", "courageous",
    "decide", "decisive", "determined", "dominant", "dominate",
    "driven", "fearless", "forceful", "headstrong", "hierarchical",
    "hostile", "impulsive", "independent", "individual", "intellectual",
    "lead", "logical", "objective", "outspoken", "principled",
    "self-confident", "self-reliant", "self-sufficient", "stubborn", "superior",
    "ninja", "rockstar", "guru", "wizard",
]

FEMININE_CODED = [
    "agree", "affectionate", "child", "cheer", "collab",
    "commit", "communal", "compassion", "connect", "considerate",
    "cooperat", "co-operat", "depend", "emotiona", "empath",
    "feel", "flatterable", "gentle", "honest", "interpersonal",
    "interdependent", "interpersona", "inter-personal", "inter-dependent", "kind",
    "kinship", "loyal", "modesty", "nag", "nurtur",
    "pleasant", "polite", "quiet", "respon", "sensitiv",
    "submissive", "support", "sympath", "tender", "together",
    "trust", "understand", "warm", "whin", "yield",
]

AGE_CODED = [
    "young", "youthful", "energetic", "digital native", "fresh",
    "recent graduate", "junior-only", "must be under", "no older than",
]

UNREALISTIC_REQUIREMENT_PATTERNS = [
    # 10+ years of <tech released within last X years>
    re.compile(r"(\d{2,})\+?\s*years?\s+(?:of\s+)?(?:experience\s+(?:with|in)\s+)?"
               r"(pytorch|tensorflow|kubernetes|react|next\.?js|typescript|rust|tailwind|"
               r"vllm|langchain|svelte|astro|remix)\b", re.I),
]


@dataclass
class BiasWarning:
    category: str   # "masculine_coded" | "feminine_coded" | "age_coded" | "unrealistic_requirements"
    term:     str
    snippet:  str   # 30 chars on either side


def _scan_terms(text: str, terms: List[str], category: str) -> List[BiasWarning]:
    out: List[BiasWarning] = []
    lower = text.lower()
    for term in terms:
        # Use a word-boundary regex when the term is a single word; substring
        # match for partial roots like "compassion" (matches compassionate).
        if " " in term or "-" in term:
            idx = lower.find(term)
            while idx >= 0:
                snippet = text[max(0, idx - 30): idx + len(term) + 30]
                out.append(BiasWarning(category=category, term=term, snippet=snippet))
                idx = lower.find(term, idx + len(term))
        else:
            for m in re.finditer(rf"\b{re.escape(term)}\w*", lower):
                start = m.start()
                snippet = text[max(0, start - 30): m.end() + 30]
                out.append(BiasWarning(category=category, term=text[m.start():m.end()],
                                       snippet=snippet))
    return out


def scan_bias(text: str) -> List[BiasWarning]:
    """Return all bias warnings the wordlist + patterns catch."""
    if not text:
        return []
    warnings: List[BiasWarning] = []
    warnings.extend(_scan_terms(text, MASCULINE_CODED,  "masculine_coded"))
    warnings.extend(_scan_terms(text, FEMININE_CODED,   "feminine_coded"))
    warnings.extend(_scan_terms(text, AGE_CODED,        "age_coded"))
    for pat in UNREALISTIC_REQUIREMENT_PATTERNS:
        for m in pat.finditer(text):
            warnings.append(BiasWarning(
                category="unrealistic_requirements",
                term=m.group(0),
                snippet=text[max(0, m.start() - 30): m.end() + 30],
            ))
    return warnings
