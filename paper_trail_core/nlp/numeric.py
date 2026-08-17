"""
Numeric and date conflict detection.

NLI models are systematically weak on numeric contradiction. Two sentences
that differ only in a figure — "raised the rate to 6.75 percent" vs. "held the
rate at 6.5 percent" — do not look *opposed* to a model trained on natural
language inference; they look like they are talking about loosely related
things, so the model returns `neutral` with high confidence. That is the exact
failure this pipeline was reported for, and no amount of NLI model swapping
fixes it, because the training data barely contains the pattern.

So we detect it explicitly. If a claim and its evidence discuss the same
subject and each states a quantity of the same *kind*, and those quantities
disagree by more than a tolerance, that is direct evidence of contradiction —
independent of what the NLI head thinks.

The output is a strength in [0, 1] that the verifier folds into the
contradiction probability, and which is also exposed as its own fusion feature
so the learned adjudicator can discount it if it ever misfires.
"""
from __future__ import annotations

import re

# Relative difference below which two figures are "the same number" — covers
# rounding between sources ("$1.2 billion" vs "$1.19 billion").
REL_TOLERANCE = 0.02

_MULTIPLIERS = {
    "hundred": 1e2, "thousand": 1e3, "k": 1e3,
    "lakh": 1e5, "million": 1e6, "m": 1e6, "mn": 1e6,
    "crore": 1e7, "billion": 1e9, "bn": 1e9, "b": 1e9,
    "trillion": 1e12, "tn": 1e12,
}

# Unit families. Two numbers only conflict if they belong to the same family —
# "5 percent" and "5 million" describe different things and must never be
# compared.
_UNIT_PATTERNS = [
    ("percent",  r"(?:%|per\s?cent(?:age)?\b|percentage\s+points?\b|basis\s+points?\b)"),
    ("currency", r"(?:\$|usd\b|eur\b|gbp\b|inr\b|rs\.?\b|₹|£|€|dollars?\b|euros?\b|rupees?\b|pounds?\b)"),
    ("year",     r"\b(?:19|20)\d{2}\b"),
    ("people",   r"(?:people|persons?|deaths?|cases?|patients?|voters?|residents?|employees?|jobs?|users?)\b"),
    ("distance", r"(?:km\b|kilomet(?:re|er)s?\b|miles?\b|met(?:re|er)s?\b)"),
    ("temp",     r"(?:°|degrees?\s*(?:c|f|celsius|fahrenheit)?\b)"),
]

_RE_NUMBER = re.compile(
    r"(?P<sign>[-+]?)"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"\s*(?P<mult>hundred|thousand|lakh|million|crore|billion|trillion|bn|mn|tn|[kmb]\b)?",
    re.IGNORECASE,
)

_RE_ENTITY = re.compile(r"\b[A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,})*\b")

# Words that are capitalised because they start a sentence, not because they
# name anything. We filter by this list rather than by position, because
# discarding every sentence-initial capital would throw away exactly the
# entities that matter most — "India raised...", "Apollo 11 landed..." both
# lead with their subject.
_STOP_ENTITIES = {
    "the", "this", "that", "these", "those", "there", "then", "but", "and",
    "however", "meanwhile", "according", "some", "many", "most", "several",
    "other", "another", "such", "since", "while", "after", "before", "during",
    "when", "where", "what", "which", "who", "why", "how", "his", "her",
    "its", "their", "our", "your", "new", "one", "two", "three",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
}


def _unit_family(text: str, start: int, end: int) -> str | None:
    """
    Classify the unit attached to a number by looking at a small window either
    side of it — units precede ("$5") or follow ("5 percent") the figure.
    """
    window = text[max(0, start - 12):min(len(text), end + 22)].lower()
    for family, pattern in _UNIT_PATTERNS:
        if family == "year":
            continue
        if re.search(pattern, window):
            return family
    return None


def extract_numbers(text: str) -> list[tuple[float, str]]:
    """
    Pull (value, unit_family) pairs out of `text`.

    Bare numbers with no identifiable unit are dropped: comparing them across
    two sentences produces far more false conflicts than true ones.
    """
    out: list[tuple[float, str]] = []
    for m in _RE_NUMBER.finditer(text):
        raw = m.group("num").replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            continue
        if m.group("sign") == "-":
            value = -value

        mult = (m.group("mult") or "").lower().strip()
        if mult:
            value *= _MULTIPLIERS.get(mult, 1.0)

        # A four-digit integer with no unit and no multiplier reads as a year.
        family = _unit_family(text, m.start(), m.end())
        if family is None:
            if not mult and raw.isdigit() and len(raw) == 4 and 1800 <= value <= 2100:
                family = "year"
            else:
                continue
        out.append((value, family))
    return out


def extract_entities(text: str) -> set[str]:
    """Capitalised tokens and ALLCAPS acronyms, minus sentence-initial noise."""
    ents = set()
    for m in _RE_ENTITY.finditer(text):
        phrase = m.group(0).lower()
        if phrase in _STOP_ENTITIES:
            continue
        # A multi-word phrase survives even if it opens with a stopword
        # ("New Delhi"); a bare stopword does not.
        ents.add(phrase)
    for m in re.finditer(r"\b[A-Z]{2,6}\b", text):
        ents.add(m.group(0).lower())
    return ents


def numeric_conflict(claim: str, premise: str) -> float:
    """
    Strength of numeric disagreement between a claim and its evidence, in [0, 1].

    Returns 0.0 unless the two texts plausibly discuss the same subject (they
    share at least one entity), since two unrelated sentences that happen to
    contain different percentages are not in conflict.

    Where the same unit family appears in both, the *closest* pair of figures
    is used. That matters: evidence often lists several numbers, and a claim
    should only be called contradicted if it disagrees with all of them, not
    merely with the least similar one.
    """
    if not claim or not premise:
        return 0.0

    if not (extract_entities(claim) & extract_entities(premise)):
        return 0.0

    claim_nums = extract_numbers(claim)
    premise_nums = extract_numbers(premise)
    if not claim_nums or not premise_nums:
        return 0.0

    worst = 0.0
    for c_val, c_unit in claim_nums:
        same_unit = [p for p, u in premise_nums if u == c_unit]
        if not same_unit:
            continue
        # Compare against the premise figure the claim agrees with *most*:
        # evidence often cites several numbers, and a claim is only
        # contradicted if it disagrees with all of them.
        strength = min(_conflict_strength(c_unit, c_val, p) for p in same_unit)
        worst = max(worst, strength)

    return round(worst, 4)


def _conflict_strength(unit: str, a: float, b: float) -> float:
    """
    How strongly two same-unit figures disagree, in [0, 1].

    Each family needs its own scale, because "how different is different"
    is not a single question:

    * **year** — categorical. 1969 vs 1972 is a 0.15% relative difference but a
      total contradiction, so any non-zero gap scores 1.0.
    * **percent** — measured in percentage *points*, not relative terms. A
      central bank rate of 6.75 vs 6.5 is a 3.7% relative difference and a
      decisive factual disagreement; relative scaling would score it near zero.
    * **everything else** — relative, which is right for magnitudes
      (populations, currency) where sources legitimately round.
    """
    if unit == "year":
        return 0.0 if abs(a - b) < 1 else 1.0

    if unit == "percent":
        # Ramps to full strength at a 1-point gap. A quarter-point gap (a
        # standard central-bank rate move) lands around 0.2 — a real but not
        # decisive signal, which is the honest reading: it could equally be a
        # revision or a differently-defined metric. The fusion layer learns
        # how much weight this deserves rather than us asserting it.
        points = abs(a - b)
        if points <= 0.05:
            return 0.0
        return min(1.0, (points - 0.05) / 0.95)

    rel = abs(a - b) / max(abs(a), abs(b), 1e-9)
    if rel <= REL_TOLERANCE:
        return 0.0
    return min(1.0, (rel - REL_TOLERANCE) / 0.48)
