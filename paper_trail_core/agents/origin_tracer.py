"""
Origin trail for claims the evidence does not support.

For each such claim this assembles, from what the pipeline can see:

  * **Published fact-checks** (Google Fact Check Tools API, ClaimReview data).
    A fact-check names who made the claim and when (`claimant`, `claimDate`).
    When one matches, that is the best answer to "where did this come from" we
    can give.
  * **A wider search** beyond trusted outlets (Google News RSS, plus SerpAPI /
    Tavily when keyed), because false claims usually start on low-trust sites.
    Hits are kept only if they are about the claim, and NLI decides whether
    each repeats or debunks it.
  * **A dated timeline** of everything that repeats or debunks the claim: the
    evidence snippets (the verifier's per-snippet `evidence_meta`), the
    wider-search hits and the fact-checks.

What this is not: proof of origin. The earliest source *found* is bounded by
what the search providers index. Messaging apps and most social posts are not
indexed, and evidence retrieval favours trusted outlets, which is rarely where a
false claim starts. The trace says so wherever it prints the result.

Report-only, like region metadata: nothing here feeds the verdict or the
fusion features, so measured results are unchanged with or without it.
"""
from __future__ import annotations

import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from paper_trail_core.agents import evidence_cache
from paper_trail_core.agents.sources import domain_of, fact_checker_trust, is_fact_checker
from paper_trail_core.agents.state import ClaimVerdict, PipelineState
from paper_trail_core.degradations import degrade
from paper_trail_core.nlp.event_match import _published

try:
    from config import settings
except ImportError:  # pragma: no cover
    settings = None

log = logging.getLogger(__name__)

FACTCHECK_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
FACTCHECK_PROVIDER = "google_factcheck"
HTTP_TIMEOUT = 8
TIMELINE_LIMIT = 8


def _cfg(name: str, default: Any) -> Any:
    return getattr(settings, name, default) if settings is not None else default


# ── Fact-check lookup ─────────────────────────────────────────────────────────

def classify_rating(rating: str) -> str:
    """Map a publisher's free-text rating ("Pants on Fire", "Misleading") to a class."""
    r = (rating or "").lower()
    if any(w in r for w in ("misleading", "partly", "partially", "half", "mixture",
                             "missing context", "out of context", "exaggerat")):
        return "misleading"
    if any(w in r for w in ("false", "fake", "incorrect", "wrong", "hoax", "fabricated",
                             "pants on fire", "not true", "untrue", "scam", "baseless",
                             "no evidence", "doctored", "edited", "satire")):
        return "false"
    if any(w in r for w in ("true", "correct", "accurate")):
        return "true"
    return "other"


def _parse_factchecks(payload: dict) -> list[dict[str, Any]]:
    """Flatten the API's claims[].claimReview[] into one row per review."""
    out: list[dict[str, Any]] = []
    for c in payload.get("claims", []) or []:
        for review in c.get("claimReview", []) or []:
            publisher = review.get("publisher") or {}
            url = review.get("url", "")
            site = publisher.get("site") or domain_of(url)
            out.append({
                "claim_text": c.get("text", ""),
                "claimant": c.get("claimant", ""),
                "claim_date": c.get("claimDate", ""),
                "publisher": publisher.get("name", "") or site,
                "site": site,
                "url": url,
                "rating": review.get("textualRating", ""),
                "rating_class": classify_rating(review.get("textualRating", "")),
                "review_date": review.get("reviewDate", ""),
            })
    return out


def _fetch_factchecks(query: str, key: str) -> list[dict[str, Any]]:
    resp = requests.get(
        FACTCHECK_URL,
        params={"query": query, "key": key, "languageCode": "en", "pageSize": 10},
        timeout=HTTP_TIMEOUT,
    )
    if resp.status_code == 429:
        raise RuntimeError("rate_limited")
    resp.raise_for_status()
    return _parse_factchecks(resp.json())


def search_factchecks(claim: str) -> tuple[list[dict[str, Any]], str]:
    """
    Fact-checks published about text like `claim`, and a status:
    ok | disabled | rate_limited | error.

    Goes through the evidence cache, so replays and tests need no network.
    """
    query = claim.strip()[:250]
    cached = evidence_cache.get(FACTCHECK_PROVIDER, query)
    if cached is not None:
        return cached, "ok"

    backend = _cfg("EVIDENCE_BACKEND", "live")
    key = _cfg("GOOGLE_FACTCHECK_API_KEY", None)
    if backend != "live" or not key:
        return [], "disabled"

    try:
        results = _fetch_factchecks(query, key)
    except RuntimeError as exc:
        if "rate_limited" in str(exc):
            return [], "rate_limited"
        log.warning("origin_tracer: fact-check lookup failed (%s)", exc)
        return [], "error"
    except Exception as exc:
        log.warning("origin_tracer: fact-check lookup failed (%s)", exc)
        return [], "error"
    evidence_cache.put(FACTCHECK_PROVIDER, query, results)
    return results, "ok"


def match_factchecks(claim: str, factchecks: list[dict[str, Any]],
                     min_cos: float | None = None) -> list[dict[str, Any]]:
    """
    Keep fact-checks whose reviewed claim is the same claim, best match first.

    The API's search is keyword-based and happily returns reviews of a
    neighbouring claim. Naming the wrong claimant is worse than naming none, so
    the bar is higher than for evidence, and if similarity can't be computed
    nothing is kept.
    """
    if not factchecks:
        return []
    floor = _cfg("FACTCHECK_MIN_RELEVANCE", 0.6) if min_cos is None else min_cos
    try:
        from paper_trail_core.nlp.embeddings import cosine_to
        sims = cosine_to(claim, [f.get("claim_text", "") for f in factchecks])
    except Exception as exc:
        log.warning("origin_tracer: similarity unavailable (%s); dropping fact-checks", exc)
        return []
    kept = []
    for fc, sim in zip(factchecks, sims):
        if float(sim) >= floor:
            kept.append({**fc, "relevance": round(float(sim), 4),
                         "trust": fact_checker_trust(fc.get("site") or fc.get("url", ""))})
    kept.sort(key=lambda f: (-f["relevance"], f.get("url", "")))
    return kept


# ── Wider search ──────────────────────────────────────────────────────────────
#
# Evidence retrieval asks trusted outlets, which is right for deciding whether a
# claim is true and wrong for finding where it started: false claims usually
# start on low-trust sites. This searches without a source allowlist, keeps the
# hits that are about the claim, and asks NLI whether each one repeats or
# debunks it. The hits only ever reach the origin trail, never the verdict.

WIDER_SEARCH_PER_CLAIM = 6      # relevant hits kept per claim

# Filter for wider-search hits. Chosen on 38 hand-labelled hits (FakeNewsNet
# stories 31-60) and checked once on 72 others (stories 1-30); both sets are in
# dataset/golden/wider_search_rows.jsonl. With only the 0.45 relevance gate, 18%
# of held-out rows were right; with this filter, 50% (9/18). What remains wrong
# is the same subject in a different event (Maxine Waters' 2010 ethics charges
# for a 2017 "criminally charged" hoax), which shares the claim's wording.
WIDER_MIN_RELEVANCE = 0.60      # headline <-> claim cosine
WIDER_MIN_COVERAGE = 0.40       # share of the claim's distinctive words the headline repeats
WIDER_MIN_STANCE = 0.50         # NLI probability behind "repeats" / "debunks"
_COVERAGE_SKIP = {"breaking"}   # hoax headlines lead with it; real ones rarely do
_PAID_PROVIDERS = (("serpapi", "SERPAPI_KEY"), ("tavily", "TAVILY_API_KEY"))


def wider_search_queries(claim: str) -> list[str]:
    """The claim as written, then its most distinctive words: two cuts at the same story."""
    from paper_trail_core.agents.evidence_retriever import _keywords

    queries = [claim.strip()[:150]]
    keywords = " ".join(_keywords(claim, 5))
    if keywords and keywords.lower() != queries[0].lower():
        queries.append(keywords)
    return [q for q in queries if q]


def _search_one(provider: str, query: str, country: str) -> tuple[list[dict[str, Any]], str]:
    """One query through the evidence cache. Off the network unless the backend is live."""
    from paper_trail_core.agents import evidence_retriever as er

    cache_provider = f"gnews_rss:{country.lower()}" if provider == "gnews_rss" else provider
    if _cfg("EVIDENCE_BACKEND", "live") != "live":
        cached = evidence_cache.get(cache_provider, query)
        return er._restamp_trust(cached or []), "ok"
    if provider == "gnews_rss":
        return er._search_regional_cached("gnews_rss", query, country)
    results, status, _hit = er._search_cached(provider, query)
    return results, status


def search_wider(claim: str, country: str) -> tuple[list[dict[str, Any]], str]:
    """
    Hits for `claim` from search that is not limited to trusted outlets, and a
    status: ok | disabled | rate_limited | error.

    Google News RSS needs no key and gives publish dates, which the timeline
    needs. SerpAPI and Tavily join when keyed, for the first query only since
    they are paid per call; their hits are undated.
    """
    if not _cfg("ORIGIN_WIDE_SEARCH", True) or _cfg("EVIDENCE_BACKEND", "live") == "mock":
        return [], "disabled"

    queries = wider_search_queries(claim)
    plan = [("gnews_rss", q) for q in queries]
    plan += [(p, queries[0]) for p, key in _PAID_PROVIDERS if _cfg(key, None)]

    hits: list[dict[str, Any]] = []
    statuses: list[str] = []
    seen: set[str] = set()
    for provider, query in plan:
        try:
            results, status = _search_one(provider, query, country)
        except Exception as exc:
            log.warning("origin_tracer: %s search failed (%s)", provider, exc)
            results, status = [], "error"
        statuses.append(status)
        for r in results:
            key = r.get("url") or f"{r.get('domain')}|{r.get('text', '')[:80]}"
            if key not in seen:
                seen.add(key)
                hits.append(dict(r))
    status = "ok" if "ok" in statuses else (statuses[0] if statuses else "disabled")
    return hits, status


def _content_words(text: str) -> set[str]:
    from paper_trail_core.agents.evidence_retriever import _QUERY_STOPWORDS
    return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower())
            if len(w) >= 3 and w not in _QUERY_STOPWORDS and w not in _COVERAGE_SKIP}


def claim_coverage(claim: str, text: str) -> float:
    """Share of the claim's distinctive words that `text` repeats."""
    words = _content_words(claim)
    return len(words & _content_words(text)) / len(words) if words else 0.0


def wider_hit_stance(claim: str, meta: dict[str, Any], relevance: float | None) -> str | None:
    """
    What a wider-search hit should be shown as (repeats | debunks | fact_check |
    mentions), or None to drop it.

    A different story about the same people passes the relevance gate easily
    ("Kim Jong-nam's death" for "Kim Jong Un dead") and NLI calls it a
    contradiction, so a hit must also repeat much of the claim's wording. A
    fact-checker's headline restates the claim, often as a question, and NLI
    reads that as repeating it; whatever NLI says, it is a fact-check.

    `relevance` None means the embedding model was unavailable: wording alone
    decides. A `stance` of None means NLI was unavailable: the hit is shown as
    "mentions" (about the claim, stance unknown), because guessing a stance from
    word overlap would call "Fact check: X did not..." a repeat of X.
    """
    if relevance is not None and relevance < WIDER_MIN_RELEVANCE:
        return None
    if claim_coverage(claim, meta.get("text", "")) < WIDER_MIN_COVERAGE:
        return None
    if is_fact_checker(meta.get("domain") or meta.get("url", "")):
        return "fact_check"
    if meta.get("stance") is None:
        return "mentions"
    if meta.get("stance") in ("repeats", "debunks") and meta.get("stance_score", 0.0) >= WIDER_MIN_STANCE:
        return meta["stance"]
    return None


def judge_hits(jobs: list[tuple[str, list[dict[str, Any]]]]) -> tuple[list[list[dict[str, Any]]], str | None]:
    """
    For each (claim, hits): the hits kept for that claim, shaped like
    `evidence_meta`. Second value says what was missing, or None:

      no_nli         stance unknown; kept hits are fact_check or mentions
      no_similarity  relevance unknown; claim wording alone decided
      no_models      both

    Measured on the hand-labelled hits (dataset/golden/wider_search_rows.jsonl,
    split=test), share of kept rows that are about the claim: 56% without NLI,
    46% without either model, against 50% right (topic and stance) with both.
    """
    from paper_trail_core.agents.verifier import _provenance
    from paper_trail_core.nlp.nli import infer_batch, is_available

    empty = [[] for _ in jobs]
    if not any(hits for _c, hits in jobs):
        return empty, None

    have_similarity = True
    try:
        from paper_trail_core.nlp.embeddings import cosine_to
        relevant = []
        for claim, hits in jobs:
            if not hits:
                relevant.append([])
                continue
            sims = cosine_to(claim, [h.get("text", "") for h in hits])
            scored = sorted(((h, float(s)) for h, s in zip(hits, sims)
                             if float(s) >= WIDER_MIN_RELEVANCE),
                            key=lambda hs: (-hs[1], hs[0].get("url", "")))
            relevant.append(scored[:WIDER_SEARCH_PER_CLAIM])
    except Exception as exc:
        log.warning("origin_tracer: similarity unavailable (%s); using claim wording only", exc)
        have_similarity = False
        relevant = [sorted(((h, None) for h in hits),
                           key=lambda hs: (-claim_coverage(claim, hs[0].get("text", "")),
                                           hs[0].get("url", "")))[:WIDER_SEARCH_PER_CLAIM]
                    for claim, hits in jobs]

    pairs = [(j, h, s) for j, scored in enumerate(relevant) for h, s in scored]
    if not pairs:
        return empty, None if have_similarity else "no_similarity"

    have_nli = _cfg("NLI_BACKEND", "transformer") == "transformer" and is_available()
    results: list = [None] * len(pairs)
    if have_nli:
        try:
            results = infer_batch([h.get("text", "") for _j, h, _s in pairs],
                                  [jobs[j][0] for j, _h, _s in pairs])
        except Exception as exc:
            log.warning("origin_tracer: NLI failed on wider-search hits (%s)", exc)
            have_nli = False
            results = [None] * len(pairs)

    out = [[] for _ in jobs]
    for (j, hit, sim), res in zip(pairs, results):
        # Trail-only trust: fact-checkers are weighted as such here (see sources.FACT_CHECKERS).
        # The domain, not the URL: Google News links are news.google.com redirects.
        hit = {**hit, "trust": fact_checker_trust(hit.get("domain") or hit.get("url", ""))}
        meta = {**_provenance(hit, res.entail if res else 0.0, res.contradict if res else 0.0),
                "relevance": round(sim, 4) if sim is not None else None,
                "text": hit.get("text", "")}
        if res is None:
            meta["stance"], meta["stance_score"] = None, 0.0
        shown_as = wider_hit_stance(jobs[j][0], meta, sim)
        if shown_as:
            out[j].append({**meta, "stance": shown_as})

    missing = {(True, True): None, (True, False): "no_nli",
               (False, True): "no_similarity", (False, False): "no_models"}
    return out, missing[(have_similarity, have_nli)]


# ── Assembly ──────────────────────────────────────────────────────────────────

def _date_only(raw: str) -> str:
    dt = _published({"published_at": raw}) if raw else None
    return dt.date().isoformat() if dt else ""


def _sort_key(entry: dict[str, Any]) -> tuple:
    # Dated entries oldest first; undated ones after, in the order they came.
    return (entry["date"] == "", entry["date"])


def build_origin(cv: ClaimVerdict, factchecks: list[dict[str, Any]],
                 factcheck_status: str, search_hits: list[dict[str, Any]] = (),
                 search_status: str = "skipped", stance_known: bool = True) -> dict[str, Any]:
    """
    The origin record for one claim, from its evidence, wider-search hits and fact-checks.

    `stance_known` False means the verifier ran without NLI, so its evidence
    stances are word overlap; those rows are shown as "mentions" instead.
    """
    timeline: list[dict[str, Any]] = []
    seen: set[str] = set()

    for found_by, metas in (("evidence", cv.get("evidence_meta", []) or []),
                            ("wider_search", search_hits)):
        for meta in metas:
            if meta.get("stance") not in ("repeats", "debunks", "fact_check", "mentions"):
                continue
            key = meta.get("url") or f"{meta.get('domain')}|{meta.get('published_at')}"
            if key in seen:
                continue
            seen.add(key)
            kind = meta["stance"]
            if found_by == "evidence" and not stance_known and kind in ("repeats", "debunks"):
                kind = "mentions"
            timeline.append({
                "date": _date_only(meta.get("published_at", "")),
                "domain": meta.get("domain", ""),
                "url": meta.get("url", ""),
                "kind": kind,
                # A fact-check found by search carries no rating; "reviewed" is
                # never counted as a debunk (classify_rating -> "other").
                "label": "reviewed" if kind == "fact_check" else "",
                "trust": meta.get("trust", 0.5),
                "found_by": found_by,
            })

    for fc in factchecks:
        if fc.get("url") in seen:
            continue
        seen.add(fc.get("url", ""))
        timeline.append({
            "date": _date_only(fc.get("review_date", "")),
            "domain": fc.get("site") or domain_of(fc.get("url", "")),
            "url": fc.get("url", ""),
            "kind": "fact_check",
            "label": fc.get("rating", ""),
            "trust": fc.get("trust", 0.5),
            "found_by": "fact_check",
        })

    timeline.sort(key=_sort_key)

    dated_repeats = [e for e in timeline if e["kind"] == "repeats" and e["date"]]
    debunks = [e for e in timeline if e["date"] and (
        e["kind"] == "debunks"
        or (e["kind"] == "fact_check"
            and classify_rating(e["label"]) in ("false", "misleading")))]

    best = next((f for f in factchecks if f.get("claimant")), None)
    claimant = best["claimant"] if best else None
    claim_date = _date_only(best.get("claim_date", "")) if best else ""

    if claimant:
        basis = "fact_check"
    elif dated_repeats:
        basis = "earliest_indexed"
    else:
        basis = "none"

    return {
        "basis": basis,
        "claimant": claimant,
        "claim_date": claim_date or None,
        "claimant_source": (best.get("site") or best.get("publisher")) if best else None,
        "earliest_seen": dated_repeats[0] if dated_repeats else None,
        "first_debunk": debunks[0] if debunks else None,
        "timeline": timeline[:TIMELINE_LIMIT],
        "fact_checks": factchecks,
        "fact_check_status": factcheck_status,
        "search_status": search_status,
    }


def needs_origin(cv: ClaimVerdict) -> bool:
    """A trail is only worth building for claims the evidence did not support."""
    return cv.get("verdict") != "entailment"


# ── Node ──────────────────────────────────────────────────────────────────────

def trace_origin_node(state: PipelineState) -> dict:
    """LangGraph node: attach an `origin` to each unsupported claim."""
    claim_verdicts = [dict(cv) for cv in state.get("claim_verdicts", [])]
    if not _cfg("ORIGIN_TRACE", True) or not claim_verdicts:
        return {}

    # With NLI degraded the verifier cannot see contradiction, so its "supported"
    # is word overlap and a hoax can carry it; trace every claim then.
    nli_degraded = bool(state.get("nli_degraded"))
    targets = [i for i, cv in enumerate(claim_verdicts) if nli_degraded or needs_origin(cv)]
    if not targets:
        return {}

    # Fact-check lookups and wider search are capped; claims past the cap still
    # get the trail built from their own evidence.
    budget = int(_cfg("ORIGIN_MAX_CLAIMS", 5))
    looked_up = targets[:budget]

    def _lookup(i: int) -> tuple[list[dict[str, Any]], str]:
        claim = claim_verdicts[i].get("claim", "")
        raw, status = search_factchecks(claim)
        return match_factchecks(claim, raw), status

    def _search(i: int) -> tuple[list[dict[str, Any]], str]:
        cv = claim_verdicts[i]
        country = ((cv.get("location") or {}).get("country_code")
                   or _cfg("ORIGIN_SEARCH_COUNTRY", "IN"))
        return search_wider(cv.get("claim", ""), country)

    with ThreadPoolExecutor(max_workers=3) as pool:
        lookup_futures = [pool.submit(_lookup, i) for i in looked_up]
        search_futures = [pool.submit(_search, i) for i in looked_up]
        lookups = dict(zip(looked_up, (f.result() for f in lookup_futures)))
        searches = dict(zip(looked_up, (f.result() for f in search_futures)))

    judged, judge_problem = judge_hits(
        [(claim_verdicts[i].get("claim", ""), searches[i][0]) for i in looked_up])
    search_hits = dict(zip(looked_up, judged))

    statuses, search_statuses = [], []
    for i in targets:
        factchecks, status = lookups.get(i, ([], "skipped"))
        _raw, search_status = searches.get(i, ([], "skipped"))
        if judge_problem and search_status == "ok":
            search_status = judge_problem
        statuses.append(status)
        search_statuses.append(search_status)
        claim_verdicts[i]["origin"] = build_origin(
            claim_verdicts[i], factchecks, status,
            search_hits.get(i, []), search_status, stance_known=not nli_degraded)

    degradations: list = []
    search_failed = sum(1 for s in search_statuses if s in ("error", "rate_limited"))
    if search_failed:
        degradations += degrade(
            "origin_tracer", "wider_search_failed",
            f"The wider source search failed for {search_failed} claim(s); their "
            "origin trails cover trusted outlets only.",
        )
    if judge_problem:
        missing = {
            "no_nli": "the NLI model, so whether a source repeats or debunks a "
                      "claim is unknown and such rows are shown as mentioning it",
            "no_similarity": "the embedding model, so rows were matched to claims "
                             "by wording alone",
            "no_models": "the NLI and embedding models, so rows were matched by "
                         "wording alone and shown as mentioning the claim",
        }[judge_problem]
        degradations += degrade(
            "origin_tracer", "wider_search_degraded",
            f"The wider source search ran without {missing}.",
            severity="info",
        )
    if "disabled" in statuses:
        degradations += degrade(
            "origin_tracer", "factcheck_disabled",
            "Fact-check lookup is off (no GOOGLE_FACTCHECK_API_KEY, or evidence "
            "backend is not live), so origin trails cannot name who made a claim.",
            severity="info",
        )
    failed = sum(1 for s in statuses if s in ("error", "rate_limited"))
    if failed:
        degradations += degrade(
            "origin_tracer", "factcheck_failed",
            f"Fact-check lookup failed for {failed} claim(s); their origin trails "
            "cannot name who made the claim.",
        )

    traced = sum(1 for i in targets if claim_verdicts[i]["origin"]["basis"] != "none")
    log.info("origin_tracer: %d/%d unsupported claims have an origin", traced, len(targets))
    return {"claim_verdicts": claim_verdicts, "degradations": degradations}
