"""
Claim-extraction agent node.

Mock mode  → heuristic sentence sampling (no API calls).
Real mode  → LLM call (Anthropic or OpenAI) returning structured JSON.
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
    _MOCK      = settings.MOCK_MODE
    _PROVIDER  = settings.LLM_PROVIDER
    _MODEL     = settings.LLM_MODEL
    _MAX       = settings.MAX_CLAIMS
except ImportError:
    _MOCK, _PROVIDER, _MODEL, _MAX = True, "anthropic", "claude-haiku-4-5-20251001", 5

from paper_trail_core.agents.state import PipelineState

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


# ── Mock ──────────────────────────────────────────────────────────────────────

def _mock_extract(text: str, n: int) -> list[str]:
    """Return heuristic claims based on the first few sentences.

    Deliberately does not pad short articles with a generic filler claim —
    with evidence retrieval now grounded in real sources, a fabricated
    "the sources are credible" claim would trivially find supporting
    evidence and dilute genuine signal from the article's actual content.
    """
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if len(s.strip()) > 20]
    selected = sentences[: min(n, len(sentences))]
    if not selected:
        selected = [text[:100]]
    return [s + "." if not s.endswith(".") else s for s in selected[:n]]


# ── Real ──────────────────────────────────────────────────────────────────────

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

    return json.loads(raw)["claims"][:n]


# ── Node ──────────────────────────────────────────────────────────────────────

def extract_claims_node(state: PipelineState) -> dict:
    """LangGraph node: extract factual claims from the article."""
    text = state.get("article_text", "")
    log.info("claim_extractor: running (mock=%s)", _MOCK)

    if not text or len(text.strip()) < 20:
        return {"claims": [], "error": "Article text too short for claim extraction"}

    try:
        claims = _mock_extract(text, _MAX) if _MOCK else _llm_extract(text, _MAX)
        log.info("claim_extractor: extracted %d claims", len(claims))
        return {"claims": claims}
    except Exception as exc:
        log.error("claim_extractor failed (%s), falling back to mock", exc)
        return {
            "claims": _mock_extract(text, _MAX),
            "error": f"claim_extractor degraded to mock: {exc}",
        }
