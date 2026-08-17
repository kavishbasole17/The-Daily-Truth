"""
Claim-extraction agent node.

Turns an article into a short list of self-contained, externally checkable
factual claims.

Backends (config.CLAIM_BACKEND):
  heuristic — default. Robust sentence segmentation + check-worthiness ranking.
              Local, deterministic, free, offline.
  llm       — optional LLM extraction, off by default. Requires a provider key.
  naive     — the original `text.split(".")` behaviour, retained *only* so the
              regression test can demonstrate the decimal-shredding bug it
              caused. Never select this in production.

Why heuristic is the default rather than an LLM: the LLM path in the previous
version was unreachable (no key, package not installed), so the naive splitter
ran unnoticed for the project's entire life while the code advertised LLM
extraction. A default that always works and always runs the same way is worth
more here than a better default that silently degrades.
"""
from __future__ import annotations

import json
import logging
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from config import settings
    _BACKEND  = settings.CLAIM_BACKEND
    _PROVIDER = settings.LLM_PROVIDER
    _MODEL    = settings.LLM_MODEL
    _MAX      = settings.MAX_CLAIMS
    _MIN_SCORE = settings.CHECKWORTHY_MIN
    _DEDUP_COS = settings.CLAIM_DEDUP_COSINE
except ImportError:  # pragma: no cover
    _BACKEND, _PROVIDER, _MODEL = "heuristic", "anthropic", "claude-haiku-4-5-20251001"
    _MAX, _MIN_SCORE, _DEDUP_COS = 8, 0.35, 0.92

from paper_trail_core.agents.state import PipelineState
from paper_trail_core.degradations import degrade
from paper_trail_core.nlp.checkworthiness import (
    checkworthiness,
    deduplicate,
    is_eligible,
)
from paper_trail_core.nlp.segmentation import segment_sentences

log = logging.getLogger(__name__)

_PROMPT = """\
You are a professional fact-checker. Extract up to {n} specific, verifiable factual
claims from the article below. Each claim must be a single, self-contained statement
that can be checked against external sources.

Article:
\"\"\"
{text}
\"\"\"

Reply with ONLY valid JSON — no markdown, no explanation:
{{"claims": ["claim 1", "claim 2", ...]}}"""


# ── Heuristic (default) ────────────────────────────────────────────────────────

def extract_claims_heuristic(
    text: str,
    max_claims: int = _MAX,
    min_score: float = _MIN_SCORE,
) -> list[str]:
    """
    Segment the article, rank sentences by check-worthiness, return the best few.

    Claims come back in **document order**, not score order: the UI renders
    them as a "claim trail" and a trail that jumps around the article is much
    harder to follow than one that reads top to bottom.
    """
    sentences = segment_sentences(text)
    if not sentences:
        return []

    scored = [
        (pos, s, checkworthiness(s))
        for pos, s in enumerate(sentences)
        if is_eligible(s)
    ]
    keep = [c for c in scored if c[2] >= min_score]

    # Nothing cleared the bar — rather than return nothing (which would leave
    # the verdict to the style classifier alone), fall back to the best
    # available sentences. Their low scores propagate as low relevance
    # downstream, so a weak article yields a low-confidence verdict rather
    # than a confident one built on unverifiable sentences.
    if not keep:
        keep = sorted(scored, key=lambda c: -c[2])[:max_claims]

    keep = deduplicate(keep, cosine_threshold=_DEDUP_COS)
    keep = sorted(keep, key=lambda c: -c[2])[:max_claims]
    return [s for _pos, s, _score in sorted(keep, key=lambda c: c[0])]


# ── Naive (regression-test only) ───────────────────────────────────────────────

def extract_claims_naive(text: str, n: int) -> list[str]:
    """
    The original implementation, preserved verbatim as a regression fixture.

    Splitting on "." breaks every decimal: "6.75 percent" becomes "6." and
    "75 percent". `tests/test_claim_extractor.py` asserts that the heuristic
    backend does not do this and that this one still does, so the bug cannot
    silently return.
    """
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if len(s.strip()) > 20]
    selected = sentences[: min(n, len(sentences))]
    if not selected:
        selected = [text[:100]]
    return [s + "." if not s.endswith(".") else s for s in selected[:n]]


# ── LLM (optional, off by default) ─────────────────────────────────────────────

def _llm_extract(text: str, n: int) -> list[str]:
    prompt = _PROMPT.format(n=n, text=text[:4000])

    if _PROVIDER == "anthropic":
        from anthropic import Anthropic
        client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        msg = client.messages.create(
            model=_MODEL,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()

    elif _PROVIDER == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content

    else:
        raise ValueError(f"Unknown LLM provider: {_PROVIDER}")

    claims = json.loads(raw)["claims"][:n]
    return _reject_hallucinations(text, claims)


def _reject_hallucinations(text: str, claims: list[str], min_cosine: float = 0.5) -> list[str]:
    """
    Drop LLM claims that aren't grounded in the article.

    An extractor that invents claims is worse than one that misses them: the
    invented claim gets retrieved and verified like any other, and its verdict
    is folded into an article-level score it has no business influencing.
    """
    sentences = segment_sentences(text)
    if not sentences or not claims:
        return claims
    try:
        from paper_trail_core.nlp.embeddings import embed
        vecs = embed(claims + sentences)
        c_vecs, s_vecs = vecs[: len(claims)], vecs[len(claims):]
        sims = c_vecs @ s_vecs.T
        return [c for i, c in enumerate(claims) if sims[i].max() >= min_cosine]
    except Exception as exc:
        log.warning("claim_extractor: hallucination check skipped (%s)", exc)
        return claims


# ── Node ──────────────────────────────────────────────────────────────────────

def extract_claims_node(state: PipelineState) -> dict:
    """LangGraph node: extract factual claims from the article."""
    text = state.get("article_text", "")
    log.info("claim_extractor: running (backend=%s)", _BACKEND)

    if not text or len(text.strip()) < 40:
        return {
            "claims": [],
            "error": "Article text too short for claim extraction",
            "degradations": degrade(
                "claim_extractor", "text_too_short",
                f"Article is {len(text.strip())} characters; need at least 40 to "
                "extract a checkable claim.",
                severity="critical",
            ),
        }

    if _BACKEND == "llm":
        try:
            claims = _llm_extract(text, _MAX)
            if claims:
                log.info("claim_extractor: %d claims via LLM", len(claims))
                return {"claims": claims, "degradations": []}
            raise ValueError("LLM returned no usable claims")
        except Exception as exc:
            # Full detail (which can include provider SDK error text — API key
            # hints, quota messages) goes to the server log only. This
            # degradation is persisted and served back by the unauthenticated
            # GET /articles/{id} endpoint, so it gets the exception class name
            # only, never the message.
            log.error("claim_extractor: LLM path failed (%s), using heuristic", exc)
            return {
                "claims": extract_claims_heuristic(text),
                "degradations": degrade(
                    "claim_extractor", "llm_failed",
                    f"LLM extraction failed ({type(exc).__name__}); fell back to "
                    "the local heuristic.",
                ),
            }

    if _BACKEND == "naive":
        return {
            "claims": extract_claims_naive(text, _MAX),
            "degradations": degrade(
                "claim_extractor", "naive_backend",
                "CLAIM_BACKEND=naive splits on '.' and breaks decimals. "
                "This backend exists only for regression testing.",
                severity="critical",
            ),
        }

    claims = extract_claims_heuristic(text)
    log.info("claim_extractor: extracted %d claims", len(claims))

    if not claims:
        return {
            "claims": [],
            "degradations": degrade(
                "claim_extractor", "no_checkable_claims",
                "No sentence in the article was specific enough to check against "
                "external sources.",
                severity="critical",
            ),
        }
    return {"claims": claims, "degradations": []}
