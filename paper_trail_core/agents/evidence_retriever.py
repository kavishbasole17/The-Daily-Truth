"""
Evidence-retrieval agent node.

For each claim, fetches candidate evidence from trusted sources and keeps only
what actually addresses the claim.

Three things changed from the previous version, each of which was producing
wrong verdicts.

**Queries.** The old code passed the entire claim sentence verbatim as the `q`
parameter to NewsAPI and Wikipedia. Both degrade badly on long natural-language
queries — you get the outlet's coverage of whatever single common word
dominated. Queries are now built from named entities, high-IDF terms and
figures, and capped short.

**Relevance.** Whatever came back was accepted as evidence. A Wikipedia article
merely *about* the Reserve Bank of India was fed to the verifier as evidence
for a specific interest-rate figure. Every candidate is now scored against the
claim with sentence embeddings and dropped below a cosine floor. A bad premise
does not produce a cautious answer from an NLI model; it produces a confident
wrong one, so this gate matters more than it looks.

**Absence of evidence.** The old code appended a synthetic "UNCORROBORATED"
snippet that the verifier turned into `contradiction` at 0.6 confidence — so
any claim our two search providers missed was scored as evidence of fakery.
Given a keyless Wikipedia lookup and a NewsAPI free tier limited to English and
roughly a one-month window, missing a claim is the *normal* outcome for
anything older, non-English or regional. That single line drove new articles
toward "fake" regardless of content. It is gone. Retrieval failure is now
reported as a status, feeds a coverage feature, and pushes toward abstention —
never toward a verdict.
"""
from __future__ import annotations

import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from config import settings
    _BACKEND   = settings.EVIDENCE_BACKEND
    _PROVIDER  = settings.SEARCH_PROVIDER
    _MAX_CALLS = settings.MAX_EVIDENCE_CALLS
    _REL_MIN   = settings.RELEVANCE_MIN
    _WORKERS   = settings.RETRIEVAL_WORKERS
except ImportError:  # pragma: no cover
    _BACKEND, _PROVIDER, _MAX_CALLS, _REL_MIN, _WORKERS = "live", "newsapi", 12, 0.35, 6

from paper_trail_core.agents import evidence_cache
from paper_trail_core.agents.sources import TRUSTED_NEWS_DOMAINS, domain_of, trust_for
from paper_trail_core.agents.state import PipelineState
from paper_trail_core.degradations import degrade
from paper_trail_core.nlp.numeric import extract_entities

log = logging.getLogger(__name__)

MAX_SNIPPETS_PER_CLAIM = 4
HTTP_TIMEOUT = 8

# Providers that search *news*. Wikipedia is deliberately not among them: it is
# the reference layer and always runs, so SEARCH_PROVIDER selects only which
# news backend (if any) runs alongside it.
_NEWS_PROVIDERS = frozenset({"newsapi", "tavily", "serpapi"})

# NewsAPI's developer tier allows 100 requests per rolling 24 hours (50 per 12).
# Its index also only reaches back about a month, which is why claims dated
# outside that window skip it entirely — see is_newsapi_reachable().
NEWSAPI_WINDOW_DAYS = 30

# Once NewsAPI returns 429 there is no point asking again for a while — the
# quota is time-based, so every further request is a guaranteed error that
# still costs a round trip. This is a *timed* cooldown rather than a permanent
# latch: the quota refills on a rolling window, and a process that stays up
# overnight should start using it again rather than run evidence-blind forever.
_NEWSAPI_COOLDOWN_S = 1800.0
_newsapi_lock = threading.Lock()
_newsapi_blocked_until = 0.0


def _newsapi_in_cooldown() -> bool:
    with _newsapi_lock:
        return time.time() < _newsapi_blocked_until


def _newsapi_start_cooldown() -> None:
    global _newsapi_blocked_until
    with _newsapi_lock:
        if time.time() >= _newsapi_blocked_until:
            log.warning(
                "evidence_retriever: NewsAPI quota exhausted — pausing news search "
                "for %.0f minutes. Wikipedia continues to be queried; claims that "
                "needed news coverage will be reported unverifiable, which lowers "
                "confidence and does not count against the article.",
                _NEWSAPI_COOLDOWN_S / 60,
            )
        _newsapi_blocked_until = time.time() + _NEWSAPI_COOLDOWN_S


def newsapi_status() -> dict[str, Any]:
    """Introspection for diagnostics / the /health endpoint."""
    with _newsapi_lock:
        remaining = max(0.0, _newsapi_blocked_until - time.time())
    return {
        "configured": bool(getattr(settings, "NEWS_API_KEY", None)),
        "in_cooldown": remaining > 0,
        "cooldown_remaining_s": round(remaining, 1),
    }

# Wikipedia throttles aggressively and will 429 a handful of concurrent
# requests, so retrieval — which runs several claims in parallel — serialises
# its calls through this gate. It treats `generator=search` + `prop=extracts`
# as an expensive query and limits it well below plain search, so the floor is
# deliberately conservative: measured at 6 workers the backoff pinned itself at
# the 8s ceiling and stages timed out; at 3 workers and 1s spacing it runs clean.
#
# Politeness aside, this matters for correctness. A 429 is indistinguishable
# downstream from "no evidence exists", so a throttle set too aggressively
# quietly converts our own impatience into apparent evidence about the article.
_WIKI_MIN_INTERVAL = 1.0
_WIKI_MAX_INTERVAL = 8.0
_wiki_lock = threading.Lock()
_wiki_last_call = 0.0
_wiki_interval = _WIKI_MIN_INTERVAL


def _wiki_throttle() -> None:
    """Serialise Wikipedia calls with an adaptive minimum spacing."""
    global _wiki_last_call
    with _wiki_lock:
        delta = time.time() - _wiki_last_call
        if delta < _wiki_interval:
            time.sleep(_wiki_interval - delta)
        _wiki_last_call = time.time()


def _wiki_backoff(hit_limit: bool) -> None:
    """
    Adapt the spacing to what Wikipedia is actually willing to serve.

    Getting rate-limited is not a cosmetic problem here: a 429 is
    indistinguishable downstream from "no evidence exists", so a throttle set
    too aggressively quietly converts our own impatience into apparent evidence
    about the article. Back off hard on a limit, recover slowly afterwards.
    """
    global _wiki_interval
    with _wiki_lock:
        if hit_limit:
            _wiki_interval = min(_WIKI_MAX_INTERVAL, _wiki_interval * 2.0)
            log.warning("evidence_retriever: wikipedia rate limit — spacing now %.1fs",
                        _wiki_interval)
        else:
            _wiki_interval = max(_WIKI_MIN_INTERVAL, _wiki_interval * 0.9)


def _wiki_get(url: str, params: dict | None, headers: dict, attempts: int = 4):
    """GET against Wikipedia with throttling and 429 backoff."""
    import requests

    last = None
    for attempt in range(attempts):
        _wiki_throttle()
        resp = requests.get(url, params=params, timeout=HTTP_TIMEOUT, headers=headers)
        if resp.status_code == 429:
            _wiki_backoff(True)
            last = resp
            time.sleep(min(_WIKI_MAX_INTERVAL, 1.0 * (2 ** attempt)))
            continue
        _wiki_backoff(False)
        return resp
    return last

_RE_TOKEN = re.compile(r"[A-Za-z][A-Za-z'-]+")
_RE_QUOTED = re.compile(r'"([^"]{4,80})"')

_QUERY_STOPWORDS = {
    "about", "after", "against", "also", "although", "among", "and", "any",
    "are", "around", "because", "been", "before", "being", "between", "both",
    "but", "came", "can", "could", "did", "does", "during", "each", "even",
    "every", "for", "from", "had", "has", "have", "her", "his", "how",
    "into", "its", "just", "like", "made", "make", "many", "may", "more",
    "most", "much", "must", "not", "now", "off", "only", "other", "our",
    "out", "over", "said", "same", "say", "says", "she", "should",
    "since", "some", "such", "than", "that", "the", "their", "them", "then",
    "there", "these", "they", "this", "those", "through", "too", "under",
    "until", "very", "was", "way", "were", "what", "when", "where", "which",
    "while", "who", "will", "with", "would", "you", "your",
}


# ── IDF from the already-trained TF-IDF vectoriser ────────────────────────────

_idf_lock = threading.Lock()
_idf: dict[str, float] | None = None
_idf_loaded = False


def _get_idf() -> dict[str, float]:
    """
    Term IDF weights lifted from models/classical/tfidf.joblib.

    Reusing the vectoriser the models were fitted on costs nothing, needs no
    new dependency, and is already domain-matched to news text — a better
    source of "which words in this sentence are distinctive" than a generic
    stopword list.
    """
    global _idf, _idf_loaded
    if _idf_loaded:
        return _idf or {}
    with _idf_lock:
        if _idf_loaded:
            return _idf or {}
        try:
            import joblib
            path = os.path.join(_ROOT, "models", "classical", "tfidf.joblib")
            vec = joblib.load(path)
            _idf = dict(zip(vec.get_feature_names_out(), vec.idf_))
            log.info("evidence_retriever: loaded IDF for %d terms", len(_idf))
        except Exception as exc:
            log.warning("evidence_retriever: IDF unavailable (%s) — using length heuristic", exc)
            _idf = {}
        _idf_loaded = True
    return _idf or {}


def _keywords(claim: str, limit: int = 5) -> list[str]:
    """Most distinctive content words in the claim, IDF-ranked."""
    idf = _get_idf()
    seen: dict[str, float] = {}
    for tok in _RE_TOKEN.findall(claim):
        low = tok.lower()
        if low in _QUERY_STOPWORDS or len(low) < 4:
            continue
        # Unknown terms are rare by definition, so rank them highly.
        seen[low] = idf.get(low, 12.0)
    return [w for w, _ in sorted(seen.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]]


def news_provider() -> str | None:
    """
    The configured news-search provider, or None if news search is disabled.

    `SEARCH_PROVIDER` selects which *news* backend to use. Wikipedia is not one
    of the options in the usual sense — it always runs, as the reference layer —
    so `SEARCH_PROVIDER=wikipedia` (or `none`) means "reference sources only,
    no news search". That is the setting to use when a news API quota is
    exhausted or you need a demo that cannot be rate-limited.
    """
    if _PROVIDER in ("wikipedia", "none", "", None):
        return None
    if _PROVIDER in _NEWS_PROVIDERS:
        return _PROVIDER
    log.warning(
        "evidence_retriever: unknown SEARCH_PROVIDER %r — falling back to newsapi. "
        "Valid values: %s, or 'wikipedia'/'none' to disable news search.",
        _PROVIDER, ", ".join(sorted(_NEWS_PROVIDERS)),
    )
    return "newsapi"


def is_newsapi_reachable(claim: str, window_days: int = NEWSAPI_WINDOW_DAYS) -> bool:
    """
    Whether NewsAPI could plausibly hold coverage of this claim.

    The developer tier indexes roughly the last month, so a claim explicitly
    dated outside that window cannot be corroborated there no matter how well
    the query is written. Spending a request on it burns quota that a checkable
    claim needs — and, worse, the resulting empty result is indistinguishable
    from "no coverage exists" unless we know why we asked.

    Claims with no date reference at all are still queried: undated claims are
    usually about current events, which is precisely what the window covers.
    """
    years = [int(y) for y in re.findall(r"\b(1[89]\d{2}|20\d{2})\b", claim)]
    if not years:
        return True

    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    # Any year older than the current one is outside a ~30-day window except in
    # the first weeks of January, which the +1 tolerance covers.
    cutoff = now.year if now.timetuple().tm_yday > window_days else now.year - 1
    return max(years) >= cutoff


def build_queries(claim: str) -> list[tuple[str, str]]:
    """
    Build (provider, query) pairs for a claim.

    NewsAPI gets **one** query, not two. The developer tier allows 100 requests
    per 24 hours in total; at two queries per claim and five claims per article
    that is ten articles a day before the quota is gone, which is not a usable
    system. One entity-anchored query per claim doubles that, and the recall
    that the second looser query used to add is largely covered by Wikipedia,
    which has no quota at all.

    Wikipedia is always queried — it is keyless, unmetered, and strong on the
    settled background facts that make up most check-worthy claims.
    """
    entities = sorted(extract_entities(claim), key=lambda e: (-len(e), e))
    quoted = _RE_QUOTED.findall(claim)
    keywords = _keywords(claim)

    anchors = [q.strip() for q in quoted[:1]] + entities[:2]
    anchors = [a for a in anchors if a][:2]

    queries: list[tuple[str, str]] = []

    news = news_provider()
    # The date-window check is a NewsAPI limitation; other providers index the
    # whole web and are worth querying regardless of how old the claim is.
    if news and (news != "newsapi" or is_newsapi_reachable(claim)):
        if anchors:
            quoted_anchors = " AND ".join(f'"{a}"' for a in anchors)
            queries.append((news, quoted_anchors if news == "newsapi"
                            else " ".join(anchors)))
        elif keywords:
            queries.append((news, " ".join(keywords[:5])))

    # Wikipedia does best with the plain subject rather than boolean syntax.
    wiki_terms = (anchors[:1] or []) + keywords[:3]
    if wiki_terms:
        queries.append(("wikipedia", " ".join(dict.fromkeys(wiki_terms))[:120]))

    # Deduplicate while preserving order.
    out: list[tuple[str, str]] = []
    for item in queries:
        if item not in out:
            out.append(item)
    return out


# ── Providers ─────────────────────────────────────────────────────────────────

def _search_newsapi(query: str) -> list[dict[str, Any]]:
    import requests

    if not getattr(settings, "NEWS_API_KEY", None):
        return []
    if _newsapi_in_cooldown():
        raise RuntimeError("newsapi_rate_limited")

    params = {
        "q": query,
        "domains": ",".join(TRUSTED_NEWS_DOMAINS),
        "language": "en",
        "searchIn": "title,description",
        "sortBy": "relevancy",
        "pageSize": 5,
        "apiKey": settings.NEWS_API_KEY,
    }
    resp = requests.get("https://newsapi.org/v2/everything", params=params, timeout=HTTP_TIMEOUT)

    if resp.status_code == 429:
        _newsapi_start_cooldown()
        raise RuntimeError("newsapi_rate_limited")
    if resp.status_code == 426:
        # "Upgrade required" — the free tier's ~1-month window was exceeded.
        # That is an absence of *coverage we can reach*, not an error.
        return []
    resp.raise_for_status()

    body = resp.json()
    if body.get("status") == "error":
        if body.get("code") in ("rateLimited", "maximumResultsReached"):
            _newsapi_start_cooldown()
            raise RuntimeError("newsapi_rate_limited")
        raise RuntimeError(f"newsapi: {body.get('code')}: {body.get('message')}")

    out = []
    for a in body.get("articles", []):
        title = (a.get("title") or "").strip()
        desc = (a.get("description") or "").strip()
        if not (title or desc):
            continue
        url = a.get("url") or ""
        source_name = (a.get("source") or {}).get("name", "")
        domain = domain_of(url) or domain_of(source_name)
        out.append({
            "text": f"{title}. {desc}".strip(),
            "url": url,
            "domain": domain,
            "trust": trust_for(domain or source_name),
            "provider": "newsapi",
            "published_at": a.get("publishedAt", ""),
        })
    return out


def _search_wikipedia(query: str) -> list[dict[str, Any]]:
    """
    Resolve the query to page titles, then fetch each page's plain-text summary.

    The old code used the `list=search` snippet, which is an HTML highlight
    fragment cut mid-sentence with markup stripped out of the middle. As an NLI
    premise it is close to useless. The REST summary endpoint returns the
    article's clean lead paragraph — one extra request for a much better premise.
    """
    headers = {"User-Agent": "PaperTrail-FakeNewsDetector/2.0 (capstone project)"}

    # `generator=search` + `prop=extracts` returns the matching pages *and*
    # their plain-text intros in a single request. The obvious implementation —
    # search, then fetch each page's summary — costs four requests per claim
    # and gets rate-limited immediately, and a rate-limited claim is
    # indistinguishable downstream from a claim nothing was written about.
    resp = _wiki_get(
        "https://en.wikipedia.org/w/api.php",
        {
            "action": "query", "format": "json",
            "generator": "search", "gsrsearch": query, "gsrlimit": 3,
            "prop": "extracts", "exintro": 1, "explaintext": 1,
            "exlimit": "max", "exsentences": 5,
        },
        headers,
    )
    if resp is None or resp.status_code == 429:
        raise RuntimeError("wikipedia_rate_limited")
    resp.raise_for_status()

    pages = (resp.json().get("query", {}) or {}).get("pages", {}) or {}

    out = []
    # Wikipedia returns pages keyed by id in arbitrary order; `index` carries
    # the search ranking. Sorting by it keeps retrieval deterministic.
    for page in sorted(pages.values(), key=lambda p: p.get("index", 999)):
        title = page.get("title", "")
        text = (page.get("extract") or "").strip()
        if not title or not text:
            continue

        out.append({
            "text": f"{title}: {text}",
            "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
            "domain": "en.wikipedia.org",
            "trust": trust_for("en.wikipedia.org"),
            "provider": "wikipedia",
            "published_at": "",
        })
    return out


def _search_tavily(query: str) -> list[dict[str, Any]]:
    from tavily import TavilyClient
    client = TavilyClient(api_key=settings.TAVILY_API_KEY)
    resp = client.search(query=query, max_results=5)
    out = []
    for r in resp.get("results", []):
        content = (r.get("content") or "").strip()
        if not content:
            continue
        url = r.get("url", "")
        domain = domain_of(url)
        out.append({"text": content, "url": url, "domain": domain,
                    "trust": trust_for(domain), "provider": "tavily", "published_at": ""})
    return out


def _search_serpapi(query: str) -> list[dict[str, Any]]:
    import requests
    resp = requests.get(
        "https://serpapi.com/search",
        params={"q": query, "api_key": settings.SERPAPI_KEY, "num": 5},
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    out = []
    for r in resp.json().get("organic_results", []):
        snippet = (r.get("snippet") or "").strip()
        if not snippet:
            continue
        url = r.get("link", "")
        domain = domain_of(url)
        out.append({"text": snippet, "url": url, "domain": domain,
                    "trust": trust_for(domain), "provider": "serpapi", "published_at": ""})
    return out


_PROVIDER_MAP = {
    "newsapi": _search_newsapi,
    "wikipedia": _search_wikipedia,
    "tavily": _search_tavily,
    "serpapi": _search_serpapi,
}


def _search_cached(provider: str, query: str) -> tuple[list[dict[str, Any]], str, bool]:
    """
    Run one provider query, going through the cache.

    Returns (results, status, cache_hit) where status is ok | rate_limited |
    error. A cache hit never touches the network, which is what makes
    evaluation replayable and the end-to-end test offline.
    """
    cached = evidence_cache.get(provider, query)
    if cached is not None:
        return cached, "ok", True

    if _BACKEND == "cache_only":
        return [], "ok", False

    fn = _PROVIDER_MAP.get(provider)
    if fn is None:
        return [], "error", False

    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            results = fn(query)
            evidence_cache.put(provider, query, results)
            return results, "ok", False
        except RuntimeError as exc:
            if "rate_limited" in str(exc):
                return [], "rate_limited", False
            last_exc = exc
        except Exception as exc:
            last_exc = exc
        # One bounded retry, for transient network/5xx only.
        if attempt == 0:
            time.sleep(0.5)

    log.warning("evidence_retriever: %s query failed (%s): %r", provider, last_exc, query[:60])
    return [], "error", False


# ── Relevance gating ──────────────────────────────────────────────────────────

def gate_by_relevance(claim: str, candidates: list[dict], min_cos: float = _REL_MIN) -> list[dict]:
    """
    Keep only candidates that semantically address the claim.

    Without this the verifier receives premises that merely share a topic with
    the claim. An NLI model given an irrelevant premise does not answer "I
    don't know" — it answers confidently, and the confident answer is noise
    that outvotes the real evidence.
    """
    if not candidates:
        return []

    texts = [c.get("text", "") for c in candidates]
    try:
        from paper_trail_core.nlp.embeddings import cosine_to
        sims = cosine_to(claim, texts)
    except Exception as exc:
        log.warning("evidence_retriever: relevance gating unavailable (%s) — keeping all", exc)
        for c in candidates:
            c["relevance"] = 0.5
        return candidates[:MAX_SNIPPETS_PER_CLAIM]

    kept = []
    for cand, sim in zip(candidates, sims):
        score = float(sim)
        if score >= min_cos:
            cand["relevance"] = round(score, 4)
            kept.append(cand)

    # Deterministic ordering: best evidence first, ties broken stably.
    kept.sort(key=lambda c: (-c["trust"] * c["relevance"], -c["relevance"], c.get("url", "")))
    return kept[:MAX_SNIPPETS_PER_CLAIM]


# ── Mock ──────────────────────────────────────────────────────────────────────

_MOCK_TEMPLATES = [
    "Multiple independent sources have corroborated this claim with documented evidence.",
    "Fact-checkers found no credible evidence supporting this assertion.",
    "The statement is partially accurate but lacks important context.",
]


def _mock_snippets(_claim: str, idx: int) -> list[dict[str, Any]]:
    return [{
        "text": _MOCK_TEMPLATES[(idx + i) % len(_MOCK_TEMPLATES)],
        "url": "", "domain": "mock.local", "trust": 0.5,
        "relevance": 0.6, "provider": "mock", "published_at": "",
    } for i in range(2)]


# ── Regional search ───────────────────────────────────────────────────────────
#
# Everything below is *additive*. The global queries above run exactly as they
# did before; when a claim has a location, regional results are gated by the
# same relevance floor and appended after them. Nothing the pipeline used to
# retrieve is displaced, and claims without a location never reach this code.

_GNEWS_MIN_INTERVAL = 1.0
_gnews_lock = threading.Lock()
_gnews_last_call = 0.0


def _cfg(name: str, default: Any) -> Any:
    return getattr(settings, name, default) if "settings" in globals() else default


def regional_provider() -> str | None:
    """The configured regional news provider, or None if regional search is off."""
    if not _cfg("REGION_AWARE", True) or _BACKEND == "mock":
        return None
    choice = (_cfg("REGIONAL_NEWS_PROVIDER", "gnews_rss") or "").lower()
    if choice in ("", "none", "off"):
        return None
    if choice == "newsdata" and _cfg("NEWSDATA_API_KEY", None):
        return "newsdata"
    return "gnews_rss"


def _strip_tags(s: str) -> str:
    import html
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", s or "")).split())


def _search_gnews_rss(query: str, country_code: str) -> list[dict[str, Any]]:
    """
    Google News RSS, scoped to a country edition.

    Keyless, and the only free source here that is genuinely regional: the
    `gl`/`ceid` parameters pick the country edition, so local outlets surface
    for local queries. Results are headlines only, which is the same grain
    NewsAPI's title+description gives the verifier.
    """
    import requests
    import xml.etree.ElementTree as ET

    global _gnews_last_call
    with _gnews_lock:
        wait = _GNEWS_MIN_INTERVAL - (time.time() - _gnews_last_call)
        if wait > 0:
            time.sleep(wait)
        _gnews_last_call = time.time()

    cc = (country_code or "us").upper()
    resp = requests.get(
        "https://news.google.com/rss/search",
        params={"q": query, "hl": f"en-{cc}", "gl": cc, "ceid": f"{cc}:en"},
        headers={"User-Agent": "Mozilla/5.0 (compatible; PaperTrail-FakeNewsDetector/2.0)"},
        timeout=HTTP_TIMEOUT,
    )
    if resp.status_code == 429:
        raise RuntimeError("gnews_rate_limited")
    resp.raise_for_status()
    body = resp.content
    if b"<!DOCTYPE" in body or b"<!ENTITY" in body:       # never expand entities
        raise RuntimeError("gnews: unexpected DTD in feed")
    root = ET.fromstring(body)

    out = []
    for item in root.iter("item"):
        title = _strip_tags(item.findtext("title") or "")
        src = item.find("source")
        src_name = (src.text or "").strip() if src is not None else ""
        src_url = src.get("url", "") if src is not None else ""
        if src_name and title.endswith(f" - {src_name}"):
            title = title[: -len(src_name) - 3].strip()
        if not title:
            continue
        desc = _strip_tags(item.findtext("description") or "")
        text = title if (not desc or desc.startswith(title) or title in desc) else f"{title}. {desc}"
        domain = domain_of(src_url) or domain_of(src_name)
        out.append({
            "text": text.rstrip(".") + ".",
            "url": item.findtext("link") or "",
            "domain": domain,
            "trust": trust_for(domain or src_name),
            "provider": "gnews_rss",
            "published_at": item.findtext("pubDate") or "",
        })
        if len(out) >= 8:
            break
    return out


def _search_newsdata(query: str, country_code: str) -> list[dict[str, Any]]:
    import requests

    params = {"apikey": settings.NEWSDATA_API_KEY, "q": query[:100], "language": "en"}
    if country_code:
        params["country"] = country_code.lower()
    resp = requests.get("https://newsdata.io/api/1/latest", params=params, timeout=HTTP_TIMEOUT)
    if resp.status_code == 429:
        raise RuntimeError("newsdata_rate_limited")
    resp.raise_for_status()
    out = []
    for r in (resp.json().get("results") or [])[:8]:
        title = (r.get("title") or "").strip()
        desc = (r.get("description") or "").strip()
        if not (title or desc):
            continue
        link = r.get("link") or ""
        domain = domain_of(link) or domain_of(r.get("source_url") or "") or domain_of(r.get("source_id") or "")
        out.append({
            "text": f"{title}. {desc}".strip(),
            "url": link, "domain": domain, "trust": trust_for(domain),
            "provider": "newsdata", "published_at": r.get("pubDate") or "",
        })
    return out


_REGIONAL_MAP = {"gnews_rss": _search_gnews_rss, "newsdata": _search_newsdata}


def _search_regional_cached(provider: str, query: str, country_code: str) -> tuple[list[dict[str, Any]], str]:
    """One regional query through the shared evidence cache. Returns (results, status)."""
    cache_provider = f"{provider}:{(country_code or 'xx').lower()}"
    cached = evidence_cache.get(cache_provider, query)
    if cached is not None:
        return cached, "ok"
    if _BACKEND == "cache_only":
        return [], "ok"

    fn = _REGIONAL_MAP[provider]
    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            results = fn(query, country_code)
            evidence_cache.put(cache_provider, query, results)
            return results, "ok"
        except RuntimeError as exc:
            if "rate_limited" in str(exc):
                return [], "rate_limited"
            last_exc = exc
        except Exception as exc:
            last_exc = exc
        if attempt == 0:
            time.sleep(0.5)
    log.warning("evidence_retriever: regional %s query failed (%s): %r",
                provider, last_exc, query[:60])
    return [], "error"


def _place_terms(location: dict[str, Any]) -> list[str]:
    """Distinct lowercase names a snippet might use for the claim's place."""
    terms = []
    for k in ("name", "locality", "city", "district", "state"):
        v = (location.get(k) or "").strip().lower()
        if len(v) >= 3 and v not in terms:
            terms.append(v)
    return terms


def _is_in_region(snippet: dict[str, Any], location: dict[str, Any]) -> bool:
    """Whether a snippet is about, or published in, the claim's region."""
    text = (snippet.get("text") or "").lower()
    for term in _place_terms(location):
        if re.search(rf"(?<![a-z]){re.escape(term)}(?![a-z])", text):
            return True
    cc = (location.get("country_code") or "").lower()
    domain = (snippet.get("domain") or "").lower()
    # ccTLD match only: ".com" says nothing about where an outlet is.
    if not cc:
        return False
    return domain.endswith(f".{cc}") or (cc == "gb" and domain.endswith((".uk", ".co.uk")))


def _scope_label(scope: dict[str, str]) -> str:
    return f"{scope['term']} ({scope['level']})"


def regional_evidence(claim: str, location: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Regional snippets for a located claim, widening the scope until there is enough.

    Starts at the most specific place the locator resolved and moves outward
    (locality -> city -> state -> country), stopping as soon as
    REGION_MIN_SNIPPETS relevant snippets are in hand. What was searched, and
    whether it had to widen, is returned so the verdict can say so.
    """
    from paper_trail_core.agents.locator import scopes_for

    info: dict[str, Any] = {"provider": None, "scopes_tried": [], "scope_used": None,
                            "widened": False, "status": "disabled", "regional_snippets": 0}
    provider = regional_provider()
    if provider is None:
        return [], info
    info["provider"] = provider
    info["status"] = "no_hits"

    scopes = scopes_for(location)[: max(1, int(_cfg("REGION_MAX_QUERIES", 3)))]
    if not scopes:
        return [], info

    # Strip the place itself from the claim so the keywords describe the *event*.
    stripped = claim
    for term in _place_terms(location):
        stripped = re.sub(rf"(?i)(?<![a-z]){re.escape(term)}(?![a-z])", " ", stripped)
    keywords = _keywords(stripped)[:4]
    if not keywords:
        return [], info

    wanted = max(1, int(_cfg("REGION_MIN_SNIPPETS", 2)))
    collected: list[dict[str, Any]] = []
    seen: set[str] = set()
    statuses: list[str] = []

    for i, scope in enumerate(scopes):
        query = f"{scope['term']} {' '.join(keywords)}"
        results, status = _search_regional_cached(provider, query, scope.get("country_code", ""))
        statuses.append(status)
        info["scopes_tried"].append(_scope_label(scope))

        fresh = []
        for r in results:
            key = r.get("url") or r.get("text", "")[:120]
            if key in seen:
                continue
            seen.add(key)
            fresh.append(r)
        gated = gate_by_relevance(claim, fresh)
        for g in gated:
            g["in_region"] = _is_in_region(g, location) or scope["level"] != "country"
            g["region_scope"] = scope["level"]
        if gated:
            collected.extend(gated)
            info["scope_used"] = _scope_label(scope)
        if len(collected) >= wanted:
            break

    if collected:
        info["status"] = "ok"
        info["widened"] = info["scope_used"] != info["scopes_tried"][0]
    elif statuses and all(s == "error" for s in statuses):
        info["status"] = "error"
    elif "rate_limited" in statuses:
        info["status"] = "rate_limited"

    collected.sort(key=lambda c: (-c["trust"] * c["relevance"], -c["relevance"], c.get("url", "")))
    return collected, info


# ── Per-claim retrieval ───────────────────────────────────────────────────────

def retrieve_for_claim(claim: str, idx: int, location: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Fetch, gate and package the evidence for a single claim.

    `location` (from the locator) adds regional search on top of the usual
    global queries; without it the behaviour is exactly what it always was.
    """
    if _BACKEND == "mock":
        return {"claim": claim, "snippets": _mock_snippets(claim, idx), "status": "ok",
                "cache_hits": 0, "queries": 0}

    candidates: list[dict[str, Any]] = []
    statuses: list[str] = []
    cache_hits = 0
    queries = build_queries(claim)

    for provider, query in queries:
        # `build_queries` already resolved which news provider to use (if any),
        # so no rerouting is needed here.
        results, status, hit = _search_cached(provider, query)
        statuses.append(status)
        cache_hits += int(hit)
        candidates.extend(results)

    # Drop duplicate URLs across queries/providers before the (costly) embedding.
    seen_urls: set[str] = set()
    deduped = []
    for c in candidates:
        key = c.get("url") or c.get("text", "")[:120]
        if key in seen_urls:
            continue
        seen_urls.add(key)
        deduped.append(c)

    snippets = gate_by_relevance(claim, deduped)

    if snippets:
        status = "ok"
    elif "rate_limited" in statuses and not deduped:
        status = "rate_limited"
    elif deduped:
        status = "no_relevant"     # results came back, none addressed the claim
    elif all(s == "error" for s in statuses) and statuses:
        status = "error"
    else:
        status = "no_hits"

    out = {"claim": claim, "snippets": snippets, "status": status,
           "cache_hits": cache_hits, "queries": len(queries)}

    if location:
        base_urls = {s.get("url") for s in snippets if s.get("url")}
        regional, region = regional_evidence(claim, location)
        extra = [s for s in regional if not s.get("url") or s["url"] not in base_urls]
        room = max(0, MAX_SNIPPETS_PER_CLAIM - len(snippets))
        take = extra[: room + max(0, int(_cfg("REGION_MAX_EXTRA_SNIPPETS", 2)))]
        for s in snippets:
            s.setdefault("in_region", _is_in_region(s, location))
        out["snippets"] = snippets + take
        region["regional_snippets"] = len(take)
        if take and out["status"] != "ok":
            out["status"] = "ok"
        out["location"] = location
        out["region"] = region
    return out


# ── Node ──────────────────────────────────────────────────────────────────────

def retrieve_evidence_node(state: PipelineState) -> dict:
    """LangGraph node: retrieve and filter evidence for every claim."""
    claims = state.get("claims", [])
    log.info("evidence_retriever: %d claims, backend=%s, provider=%s",
             len(claims), _BACKEND, _PROVIDER)

    if not claims:
        return {"evidence": [], "retrieval_stats": {}, "degradations": []}

    budget = claims[:_MAX_CALLS]
    degradations: list = []
    if len(claims) > _MAX_CALLS:
        degradations += degrade(
            "evidence_retriever", "claim_budget",
            f"Checked the {_MAX_CALLS} highest-ranked claims of {len(claims)} "
            "to stay within the retrieval budget.",
            severity="info",
        )

    started = time.time()

    # Network-bound, so threads help and the GIL is released during requests.
    # Results are reassembled by index, never by completion order — otherwise
    # the same article could produce differently-ordered evidence run to run.
    locations = state.get("claim_locations") or []

    def _fetch(p):
        idx, claim = p
        loc = locations[idx] if idx < len(locations) else None
        # Called without `location` when there is none, so claims the locator
        # found nothing for take the original two-argument path untouched.
        return retrieve_for_claim(claim, idx, location=loc) if loc else retrieve_for_claim(claim, idx)

    with ThreadPoolExecutor(max_workers=max(1, _WORKERS)) as pool:
        evidence = list(pool.map(_fetch, list(enumerate(budget))))

    for i, claim in enumerate(claims[_MAX_CALLS:], start=_MAX_CALLS):
        skipped = {"claim": claim, "snippets": [], "status": "skipped",
                   "cache_hits": 0, "queries": 0}
        if i < len(locations) and locations[i]:
            skipped["location"] = locations[i]
        evidence.append(skipped)

    n_ok = sum(1 for e in evidence if e["status"] == "ok")
    n_rate_limited = sum(1 for e in evidence if e["status"] == "rate_limited")
    n_error = sum(1 for e in evidence if e["status"] == "error")
    total_queries = sum(e.get("queries", 0) for e in evidence)
    total_hits = sum(e.get("cache_hits", 0) for e in evidence)
    relevances = [s["relevance"] for e in evidence for s in e["snippets"]]
    trusts = [s["trust"] for e in evidence for s in e["snippets"]]

    regions = [e["region"] for e in evidence if e.get("region")]
    stats = {
        "n_claims": len(evidence),
        "n_with_evidence": n_ok,
        "coverage": round(n_ok / max(1, len(evidence)), 4),
        "n_no_hits": sum(1 for e in evidence if e["status"] in ("no_hits", "no_relevant")),
        "n_rate_limited": n_rate_limited,
        "n_error": n_error,
        "mean_relevance": round(sum(relevances) / len(relevances), 4) if relevances else 0.0,
        "mean_trust": round(sum(trusts) / len(trusts), 4) if trusts else 0.0,
        "cache_hit_rate": round(total_hits / max(1, total_queries), 4),
        "wall_s": round(time.time() - started, 3),
        "n_located": sum(1 for e in evidence if e.get("location")),
        "n_regional_snippets": sum(r.get("regional_snippets", 0) for r in regions),
    }

    if n_rate_limited:
        news = newsapi_status()
        if news["in_cooldown"]:
            detail = (
                f"The NewsAPI quota is exhausted (developer tier: 100 requests per "
                f"24 hours), so {n_rate_limited} claim(s) could not be checked "
                f"against news coverage. News search resumes in about "
                f"{news['cooldown_remaining_s'] / 60:.0f} minutes; Wikipedia was "
                "still queried in the meantime."
            )
        else:
            detail = (
                f"A search provider rate-limited us, so {n_rate_limited} claim(s) "
                "could not be checked."
            )
        degradations += degrade(
            "evidence_retriever", "rate_limited",
            detail + " This lowers confidence in the verdict; it is not evidence "
            "against the article.",
        )
    if n_error:
        degradations += degrade(
            "evidence_retriever", "search_error",
            f"Search failed for {n_error} claim(s) after retrying.",
        )
    widened = [e for e in evidence if e.get("region", {}).get("widened")]
    if widened:
        first = widened[0]["region"]
        degradations += degrade(
            "evidence_retriever", "region_widened",
            f"For {len(widened)} claim(s) there was not enough local coverage at the "
            f"most specific place, so the search widened (e.g. {first['scopes_tried'][0]} "
            f"-> {first['scope_used']}).",
            severity="info",
        )
    uncovered = [e for e in evidence
                 if e.get("region") and e["region"].get("provider")
                 and e["region"].get("regional_snippets", 0) == 0]
    if uncovered:
        place = uncovered[0]["location"].get("label") or uncovered[0]["location"].get("name")
        degradations += degrade(
            "evidence_retriever", "no_regional_coverage",
            f"No relevant regional news was found for {len(uncovered)} located claim(s) "
            f"(e.g. {place}); they were checked against the usual global sources only.",
            severity="info",
        )
    if n_ok == 0:
        degradations += degrade(
            "evidence_retriever", "no_evidence",
            "No relevant evidence was found for any claim. The verdict cannot be "
            "evidence-grounded and will be reported as uncertain. Note that "
            "absence of coverage is not evidence of fabrication — recent, "
            "regional and non-English topics are routinely missed by these sources.",
            severity="critical",
        )

    log.info("evidence_retriever: %s", stats)
    return {"evidence": evidence, "retrieval_stats": stats, "degradations": degradations}
