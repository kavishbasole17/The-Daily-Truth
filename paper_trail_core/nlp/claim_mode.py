"""
Is this input a claim, or an article?

Short inputs (one or two sentences) are decided on evidence alone; anything
longer takes the article path (style classifier + learned fusion), which was
fitted on whole articles and is measured to work better on them. One definition
here, so retrieval, claim extraction and adjudication can never disagree about
which path a given text is on.
"""
from __future__ import annotations

import re

_SENTENCE_END = re.compile(r"[.!?]+(?=\s+[A-Z\"'(]|\s*$)")
_ABBREV = re.compile(r"\b(?:Mr|Mrs|Ms|Dr|Prof|St|Jr|Sr|vs|etc|Inc|Ltd|No|U\.S|U\.K)\.", re.IGNORECASE)


def count_sentences(text: str) -> int:
    """Rough sentence count; abbreviations and decimals do not end a sentence."""
    cleaned = _ABBREV.sub(lambda m: m.group(0).replace(".", ""), text or "")
    cleaned = re.sub(r"(?<=\d)\.(?=\d)", "", cleaned)
    n = len(_SENTENCE_END.findall(cleaned.strip()))
    return max(1, n) if cleaned.strip() else 0


def is_claim_text(text: str) -> bool:
    """True for a short input of at most CLAIM_MODE_MAX_SENTENCES sentences."""
    try:
        from config import settings
        if not settings.CLAIM_MODE:
            return False
        max_chars = int(settings.CLAIM_MODE_MAX_CHARS)
        max_sentences = int(settings.CLAIM_MODE_MAX_SENTENCES)
    except Exception:  # pragma: no cover
        return False
    text = (text or "").strip()
    return 0 < len(text) <= max_chars and count_sentences(text) <= max_sentences
