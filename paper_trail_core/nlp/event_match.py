"""
Event corroboration for time-bound claims ("Metro to shut 57 stations from 9pm
on Friday").

A breaking story is rarely restated word for word: the next day's headlines say
"17 stations shut for third day". An NLI model reads the changed figures as
disagreement, and a keyword search rarely returns a headline that entails the
claim. What can be checked is whether the *event itself* is being reported: the
claim's distinctive terms all appear in recent items from several independent
news outlets.

This is a weaker statement than "the claim is true", so the caller reports it as
"event confirmed, details differ" and caps the confidence. It is deliberately
strict about the words: a fabricated variant that adds a term no outlet uses
("permanently", "resigns") fails the coverage check.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Iterable, Optional

from paper_trail_core.nlp.numeric import numeric_conflict

# Words that carry no event identity: time words, report-style framing and
# function words. Numbers are dropped separately.
_IGNORED = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "and", "or",
    "but", "if", "of", "at", "by", "for", "with", "about", "to", "from", "in",
    "on", "into", "over", "this", "that", "these", "those", "has", "have",
    "had", "will", "would", "could", "should", "it", "its", "as", "than",
    "so", "not", "no", "down", "up", "out", "after", "before", "amid", "all",
    "live", "update", "updates", "breaking", "news", "latest", "today",
    "tonight", "tomorrow", "yesterday", "ahead", "till", "until", "week",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "mon", "tue", "tues", "wed", "thu", "thur", "thurs", "fri",
    "sat", "sun", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug",
    "sep", "sept", "oct", "nov", "dec", "january", "february", "march",
    "april", "june", "july", "august", "september", "october", "november",
    "december", "am", "pm", "morning", "evening", "night", "day", "days",
    "led", "see", "full", "list",
}

# Suffixes folded away so "shut"/"shuts"/"shutting" and "protest"/"protests" meet.
_SUFFIXES = ("ings", "ing", "es", "ed", "s")

SNIPPET_COVERAGE = 0.75     # share of the claim's terms one item must contain
UNION_COVERAGE = 0.90       # share of the claim's terms the matching items cover


def _stem(word: str) -> str:
    for suf in _SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[: -len(suf)]
    return word


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z']*", text.replace("-", " "))


def event_terms(claim: str) -> tuple[set[str], set[str]]:
    """
    (terms, acronyms): the claim's distinctive stems, and the subset that are
    acronyms or initialisms (CJP, PM), which every matching item must contain.
    """
    terms: set[str] = set()
    acronyms: set[str] = set()
    for tok in _tokens(claim):
        base = re.sub(r"'s?$", "", tok)
        low = base.lower()
        if len(low) < 2 or low in _IGNORED:
            continue
        stem = _stem(low)
        terms.add(stem)
        if base.isupper() and len(base) >= 2:
            acronyms.add(stem)
    return terms, acronyms


def _item_stems(text: str) -> set[str]:
    return {_stem(re.sub(r"'s?$", "", t).lower()) for t in _tokens(text)}


_TIMEISH_RE = re.compile(
    r"\b\d{1,2}(?::\d{2})?\s?(?:am|pm)\b"
    r"|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}\b"
    r"|\b\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b"
    r"|\b(?:19|20)\d{2}\b",
    re.IGNORECASE,
)


def _figures(text: str) -> set[str]:
    """Quantities in the text, with clock times, dates and years removed."""
    return set(re.findall(r"\d+(?:\.\d+)?", _TIMEISH_RE.sub(" ", text)))


def _published(snip: dict) -> Optional[datetime]:
    raw = snip.get("published_at") or snip.get("published") or ""
    if not raw:
        return None
    try:
        dt = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        try:
            dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def corroboration(
    claim: str,
    snippets: Iterable[dict[str, Any]],
    *,
    min_snippets: int = 3,
    min_sources: int = 2,
    max_age_days: float = 7.0,
    now: Optional[datetime] = None,
) -> dict[str, Any]:
    """
    Is the event in `claim` being reported by recent, independent news items?

    Returns {"corroborated": bool, "sources": [domains], "matched": int,
    "coverage": float, "detail_mismatch": bool, "missing": [terms]}.
    """
    now = now or datetime.now(timezone.utc)
    terms, acronyms = event_terms(claim)
    result: dict[str, Any] = {
        "corroborated": False, "sources": [], "matched": 0,
        "coverage": 0.0, "detail_mismatch": False, "missing": sorted(terms),
    }
    if len(terms) < 3:      # too little to identify an event by
        return result

    cutoff = now - timedelta(days=max_age_days)
    matched: list[dict] = []
    covered: set[str] = set()
    for s in snippets:
        domain = str(s.get("domain", ""))
        if not domain or domain.endswith("wikipedia.org"):
            continue                       # reference pages are not news reports
        when = _published(s)
        if when is None or when < cutoff or when > now + timedelta(days=1):
            continue
        stems = _item_stems(s.get("text", ""))
        hit = terms & stems
        if acronyms - stems:
            continue
        if len(hit) / len(terms) < SNIPPET_COVERAGE:
            continue
        matched.append(s)
        covered |= hit

    coverage = len(covered) / len(terms)
    sources = sorted({str(s.get("domain")) for s in matched})
    result.update(
        matched=len(matched), sources=sources, coverage=round(coverage, 3),
        missing=sorted(terms - covered),
        detail_mismatch=bool(matched) and (
            bool(_figures(claim) - set().union(*(_figures(s.get("text", "")) for s in matched)))
            or any(numeric_conflict(claim, s.get("text", "")) > 0 for s in matched)
        ),
    )
    result["corroborated"] = (
        len(matched) >= min_snippets
        and len(sources) >= min_sources
        and coverage >= UNION_COVERAGE
    )
    return result
