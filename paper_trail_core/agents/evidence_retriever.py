"""
Evidence-retrieval agent node.

For each extracted claim, fetches supporting/contradicting snippets from
trusted sources.

Real mode (default — no key required):
  1. NewsAPI, restricted to a curated whitelist of trusted wire services and
     outlets (Reuters, AP, BBC, NPR, ...), when NEWS_API_KEY is configured.
  2. Wikipedia's search API, always available, no key required — general
     factual/background grounding.
  3. If a claim gets zero hits from *both*, that absence is itself a
     signal: genuinely newsworthy factual claims are almost always picked
     up by at least one trusted outlet or reference source within days.
     We surface that explicitly rather than silently treating it as
     "no information" (see UNCORROBORATED_MARKER, consumed by verifier.py).

Mock mode → returns synthetic canned snippets (no network calls).

The search provider is pluggable via SEARCH_PROVIDER in .env. Total external
calls are capped at MAX_EVIDENCE_CALLS to control latency (one "call" budget
per claim, regardless of how many underlying HTTP requests that claim uses).
"""
from __future__ import annotations

import logging
import os
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from config import settings
    _MOCK      = settings.EVIDENCE_MOCK_MODE
    _PROVIDER  = settings.SEARCH_PROVIDER
    _MAX_CALLS = settings.MAX_EVIDENCE_CALLS
except ImportError:
    _MOCK, _PROVIDER, _MAX_CALLS = True, "newsapi", 5

from paper_trail_core.agents.state import PipelineState

log = logging.getLogger(__name__)

UNCORROBORATED_MARKER = "UNCORROBORATED:"

# Curated whitelist of trusted-outlet domains, used with NewsAPI's `domains`
# filter. We use `domains` rather than NewsAPI's fixed `sources` catalog
# because `domains` accepts any outlet's domain directly — NewsAPI's
# `sources` enum has very thin non-US/UK coverage (e.g. only 3 India-tagged
# entries total), and combining `sources` + `domains` in one request silently
# drops the `domains` filter, so international/regional outlets never
# actually got queried before. This list intentionally spans multiple
# regions rather than defaulting to US/UK wire services alone.
TRUSTED_NEWS_DOMAINS = [
    # Global wire services / int'l
    "reuters.com", "apnews.com", "bbc.co.uk", "aljazeera.com",
    # US
    "npr.org", "wsj.com", "washingtonpost.com", "cbsnews.com",
    "abcnews.go.com", "nbcnews.com", "axios.com", "politico.com",
    "bloomberg.com", "usatoday.com",
    # UK / Europe
    "theguardian.com", "dw.com",
    # India
    "thehindu.com", "timesofindia.indiatimes.com", "ndtv.com",
    "indianexpress.com", "hindustantimes.com", "livemint.com",
    # Asia-Pacific
    "japantimes.co.jp", "scmp.com", "abc.net.au",
]


# ── Provider adapters ──────────────────────────────────────────────────────────

def _search_newsapi(query: str) -> list[str]:
    import requests
    if not settings.NEWS_API_KEY:
        return []
    params = {
        "q": query,
        "domains": ",".join(TRUSTED_NEWS_DOMAINS),
        "sortBy": "relevancy",
        "pageSize": 3,
        "apiKey": settings.NEWS_API_KEY,
    }
    resp = requests.get("https://newsapi.org/v2/everything", params=params, timeout=10)
    resp.raise_for_status()
    articles = resp.json().get("articles", [])
    snippets = []
    for a in articles:
        source = (a.get("source") or {}).get("name", "a trusted outlet")
        title = (a.get("title") or "").strip()
        desc = (a.get("description") or "").strip()
        text = f"{source} reports: {title}. {desc}".strip()
        if title or desc:
            snippets.append(text)
    return snippets


def _search_wikipedia(query: str) -> list[str]:
    import requests
    resp = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query", "list": "search", "srsearch": query,
            "format": "json", "srlimit": 3,
        },
        timeout=10,
        headers={"User-Agent": "PaperTrail-FakeNewsDetector/1.0 (capstone project)"},
    )
    resp.raise_for_status()
    hits = resp.json().get("query", {}).get("search", [])
    snippets = []
    for h in hits:
        title = h.get("title", "")
        snippet = re.sub(r"<[^>]+>", "", h.get("snippet", ""))
        if title or snippet:
            snippets.append(f"Wikipedia — {title}: {snippet}")
    return snippets


def _search_serpapi(query: str) -> list[str]:
    import requests
    params = {
        "q": f"fact check {query}",
        "api_key": settings.SERPAPI_KEY,
        "num": 3,
    }
    resp = requests.get("https://serpapi.com/search", params=params, timeout=10)
    resp.raise_for_status()
    results = resp.json().get("organic_results", [])
    return [r.get("snippet", "") for r in results if r.get("snippet")]


def _search_tavily(query: str) -> list[str]:
    from tavily import TavilyClient
    client = TavilyClient(api_key=settings.TAVILY_API_KEY)
    resp = client.search(query=f"fact check: {query}", max_results=3)
    return [r.get("content", "") for r in resp.get("results", []) if r.get("content")]


_PROVIDER_MAP = {
    "newsapi":  _search_newsapi,
    "wikipedia": _search_wikipedia,
    "tavily":   _search_tavily,
    "serpapi":  _search_serpapi,
}


def _search_trusted(claim: str) -> list[str]:
    """
    Layered real-mode search: try the configured primary provider, then
    always supplement with Wikipedia (free, keyless) unless it *was* the
    primary. If both come back empty, surface that absence explicitly
    instead of pretending we have no signal at all.
    """
    snippets: list[str] = []

    primary_fn = _PROVIDER_MAP.get(_PROVIDER)
    if primary_fn is None:
        raise ValueError(f"Unknown search provider: {_PROVIDER}")

    try:
        snippets.extend(primary_fn(claim))
    except Exception as exc:
        log.warning("evidence_retriever: primary provider '%s' failed for claim: %s", _PROVIDER, exc)

    if _PROVIDER != "wikipedia":
        try:
            snippets.extend(_search_wikipedia(claim))
        except Exception as exc:
            log.warning("evidence_retriever: wikipedia fallback failed for claim: %s", exc)

    if not snippets:
        snippets.append(
            f"{UNCORROBORATED_MARKER} No coverage of this claim was found among major "
            "trusted news outlets (Reuters, AP, BBC, and others) or general reference "
            "sources, which is unusual for a genuinely newsworthy factual claim."
        )

    return snippets[:3]


# ── Mock ──────────────────────────────────────────────────────────────────────

_MOCK_TEMPLATES = [
    "Multiple independent sources have corroborated this claim with documented evidence.",
    "Fact-checkers found no credible evidence supporting this assertion.",
    "The statement is partially accurate but lacks important context.",
    "Official government records are consistent with this claim.",
    "Several news organisations have raised doubts about this claim.",
]


def _mock_snippets(claim: str, idx: int) -> list[str]:
    a = _MOCK_TEMPLATES[idx % len(_MOCK_TEMPLATES)]
    b = _MOCK_TEMPLATES[(idx + 2) % len(_MOCK_TEMPLATES)]
    return [a, b]


# ── Node ──────────────────────────────────────────────────────────────────────

def retrieve_evidence_node(state: PipelineState) -> dict:
    """LangGraph node: for each claim fetch supporting/contradicting snippets."""
    claims = state.get("claims", [])
    log.info("evidence_retriever: %d claims, mock=%s, provider=%s", len(claims), _MOCK, _PROVIDER)

    if not claims:
        return {"evidence": []}

    evidence = []
    calls_remaining = _MAX_CALLS

    for idx, claim in enumerate(claims):
        if calls_remaining <= 0:
            log.warning("evidence_retriever: hit MAX_EVIDENCE_CALLS (%d), stopping", _MAX_CALLS)
            evidence.append({"claim": claim, "snippets": []})
            continue

        try:
            if _MOCK:
                snippets = _mock_snippets(claim, idx)
            else:
                snippets = _search_trusted(claim)
                calls_remaining -= 1

            evidence.append({"claim": claim, "snippets": snippets})
        except Exception as exc:
            log.error("evidence_retriever: search failed for claim %d: %s", idx, exc)
            evidence.append({"claim": claim, "snippets": _mock_snippets(claim, idx)})

    log.info("evidence_retriever: done (%d evidence items)", len(evidence))
    return {"evidence": evidence}
