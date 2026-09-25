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

log = logging.getLogger(__name__)

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
        return best.lang, float(best.prob)
    except Exception as exc:
        log.warning("lang_id: detection failed (%s)", exc)
        return "unknown", 0.0
