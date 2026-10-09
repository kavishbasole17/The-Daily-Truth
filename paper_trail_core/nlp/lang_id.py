"""
Language identification.

`langdetect` (pure Python, no model download) over fastText lid.176 or a
transformer-based detector: this project runs on CPU locally with a memory
budget that already forced two aborted GPU-toolkit installs this session
(see docs/ARCHITECTURE.md's compute section) — a multi-hundred-MB model
download for a task this well-served by a lightweight library isn't worth
the risk. Revisit only if detection accuracy turns out to be the bottleneck,
not preemptively.
"""
from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

# langdetect is statistical over character n-grams, which is thin evidence for a
# sentence: "Water boils at 100 degrees Celsius at sea level." comes back as
# Catalan at 0.99999, and the pipeline then refuses it as an unsupported
# language. Only English and Hindi are supported and Hindi is not Latin script,
# so for short plain-ASCII text, "English" is the right default: if it is really
# some other language the evidence comes back neutral and the verdict is
# uncertain, a safe failure, where a wrong rejection of English is not.
_ENGLISH_ANCHORS = frozenset(
    "the of and with that this from which its been were are have has by".split())
_ENGLISH_COMMON = frozenset("is was to for in it as be on at".split())
SHORT_ASCII_MAX_CHARS = 200


def _is_ascii_text(text: str) -> bool:
    return all(ord(c) < 128 for c in text if c.isalpha())


def _looks_english(text: str) -> bool:
    """Plain-ASCII text that is short, or has several ordinary English function words."""
    if not _is_ascii_text(text) or not any(c.isalpha() for c in text):
        return False
    if len(text) <= SHORT_ASCII_MAX_CHARS:
        return True
    found = set(re.findall(r"[a-z']+", text.lower()))
    anchors = found & _ENGLISH_ANCHORS
    return bool(anchors) and len(anchors | (found & _ENGLISH_COMMON)) >= 2

# MVP supports English (no-op) and Hindi (translated). Anything else is
# flagged unsupported rather than silently run through English-tuned claim
# extraction/NLI/stylistic models — see docs/ARCHITECTURE.md stretch tier
# for Telugu/Tamil.
SUPPORTED_LANGUAGES = {"en", "hi"}


def detect_language(text: str) -> tuple[str, float]:
    """
    Returns (ISO 639-1 code, confidence in [0,1]).

    ("unknown", 0.0) on empty input or detector failure — never raises, so
    callers can treat that tuple as the single "detection didn't work" case
    without a separate try/except.
    """
    text = (text or "").strip()
    if len(text) < 3:
        return "unknown", 0.0
    try:
        from langdetect import DetectorFactory, detect_langs
        DetectorFactory.seed = 0  # deterministic — langdetect samples internally
        candidates = detect_langs(text)
        if not candidates:
            return "unknown", 0.0
        best = candidates[0]
        if best.lang != "en" and _looks_english(text):
            return "en", 0.6
        return best.lang, float(best.prob)
    except Exception as exc:
        log.warning("lang_id: detection failed (%s)", exc)
        return "unknown", 0.0
