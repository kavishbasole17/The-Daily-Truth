"""
Evidence-retrieval tests — queries, gating, caching, and the absence-of-evidence
contract. No network access; providers are monkeypatched.
"""
from __future__ import annotations

import json
import os

import pytest

from paper_trail_core.agents import evidence_cache, evidence_retriever as ER
from paper_trail_core.agents.sources import domain_of, trust_for


# ── Query construction ────────────────────────────────────────────────────────

def test_queries_are_short_not_whole_sentences():
    """
    The old code passed the entire claim as `q`. Both NewsAPI and Wikipedia
    degrade badly on long natural-language queries.
    """
    claim = ("India's central bank raised its benchmark repo rate by 25 basis "
             "points to 6.75 percent on Thursday, citing persistent food inflation.")
    queries = ER.build_queries(claim)
    assert queries
    for _provider, q in queries:
        assert len(q) < len(claim)
        assert len(q.split()) <= 12


def test_queries_cover_both_providers():
    providers = {p for p, _ in ER.build_queries("The World Health Organization reported 118,000 cases in 2020.")}
    assert "wikipedia" in providers


def test_queries_are_deterministic():
    claim = "Apollo 11 landed the first humans on the Moon on 20 July 1969."
    assert ER.build_queries(claim) == ER.build_queries(claim)


def test_at_most_one_newsapi_query_per_claim():
    """
    NewsAPI's developer tier allows 100 requests per 24 hours *in total*. Two
    queries per claim across five claims exhausts it in ten articles, which is
    not a usable system.
    """
    claim = "India's central bank raised the repo rate to 6.75 percent on Thursday."
    news_queries = [q for p, q in ER.build_queries(claim) if p == "newsapi"]
    assert len(news_queries) <= 1


def test_claims_outside_the_news_window_skip_newsapi():
    """
    The developer tier only indexes ~1 month, so a claim explicitly dated years
    ago cannot be corroborated there — spending quota on it starves the claims
    that could be.
    """
    old = "Apollo 11 landed the first humans on the Moon on 20 July 1969."
    assert ER.is_newsapi_reachable(old) is False
    assert not [p for p, _ in ER.build_queries(old) if p == "newsapi"]
    # Wikipedia has no quota and is strong on settled facts — still queried.
    assert [p for p, _ in ER.build_queries(old) if p == "wikipedia"]


def test_undated_claims_still_query_newsapi():
    """Undated claims are usually about current events — exactly the window."""
    assert ER.is_newsapi_reachable(
        "The central bank raised its benchmark rate by 25 basis points."
    ) is True


def test_newsapi_cooldown_expires(monkeypatch):
    """
    The cooldown is timed, not a permanent latch.

    A latch meant one exhausted quota left the process running evidence-blind
    until restart, long after the rolling window had refilled.
    """
    monkeypatch.setattr(ER, "_newsapi_blocked_until", 0.0)
    ER._newsapi_start_cooldown()
    assert ER._newsapi_in_cooldown() is True

    now = ER.time.time()
    monkeypatch.setattr(ER.time, "time", lambda: now + ER._NEWSAPI_COOLDOWN_S + 1)
    assert ER._newsapi_in_cooldown() is False


def test_no_queries_for_empty_claim():
    assert ER.build_queries("") == []


# ── Source trust ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("value,expected", [
    ("https://www.reuters.com/world/article", "reuters.com"),
    ("https://edition.cnn.com/2024/story", "edition.cnn.com"),
    ("apnews.com", "apnews.com"),
])
def test_domain_extraction(value, expected):
    assert domain_of(value) == expected


def test_trust_ordering():
    assert trust_for("reuters.com") > trust_for("en.wikipedia.org")
    assert trust_for("en.wikipedia.org") > trust_for("some-random-blog.example")


def test_subdomain_inherits_parent_trust():
    assert trust_for("edition.cnn.com") == trust_for("cnn.com")


# ── Relevance gating ──────────────────────────────────────────────────────────

def test_gating_drops_irrelevant_snippets():
    claim = "Apollo 11 landed humans on the Moon in 1969."
    candidates = [
        {"text": "Apollo 11 was the American spaceflight that first landed humans on the Moon.",
         "trust": 0.85, "domain": "en.wikipedia.org", "url": "u1"},
        {"text": "Bananas are a tropical fruit grown in warm climates worldwide.",
         "trust": 0.85, "domain": "en.wikipedia.org", "url": "u2"},
    ]
    kept = ER.gate_by_relevance(claim, candidates)
    texts = [c["text"] for c in kept]
    assert any("Apollo" in t for t in texts)
    assert not any("Bananas" in t for t in texts)


def test_gating_assigns_relevance_scores():
    kept = ER.gate_by_relevance(
        "The World Health Organization declared a pandemic in March 2020.",
        [{"text": "The WHO declared COVID-19 a pandemic on 11 March 2020.",
          "trust": 0.9, "domain": "who.int", "url": "u"}],
    )
    assert kept and 0.0 <= kept[0]["relevance"] <= 1.0


def test_gating_handles_empty_input():
    assert ER.gate_by_relevance("a claim", []) == []


# ── Absence of evidence ───────────────────────────────────────────────────────

def test_no_hits_reports_status_not_a_synthetic_snippet(monkeypatch):
    """
    Regression: the old retriever appended a synthetic "UNCORROBORATED" snippet
    that the verifier converted into contradiction at 0.6. Nothing found must
    produce a *status*, never a fabricated piece of evidence.
    """
    monkeypatch.setattr(ER, "_search_cached", lambda p, q: ([], "ok", False))
    result = ER.retrieve_for_claim("A claim nobody has written about.", 0)

    assert result["snippets"] == []
    assert result["status"] == "no_hits"
    for snippet in result["snippets"]:
        assert "UNCORROBORATED" not in snippet.get("text", "")


def test_rate_limit_is_distinguished_from_no_hits(monkeypatch):
    monkeypatch.setattr(ER, "_search_cached", lambda p, q: ([], "rate_limited", False))
    assert ER.retrieve_for_claim("Some claim.", 0)["status"] == "rate_limited"


def test_node_reports_no_evidence_as_a_degradation(monkeypatch):
    monkeypatch.setattr(ER, "_search_cached", lambda p, q: ([], "ok", False))
    out = ER.retrieve_evidence_node({"claims": ["A claim about something obscure."]})

    assert out["retrieval_stats"]["coverage"] == 0.0
    kinds = {d["kind"] for d in out["degradations"]}
    assert "no_evidence" in kinds


def test_node_with_no_claims_is_a_clean_noop():
    out = ER.retrieve_evidence_node({"claims": []})
    assert out["evidence"] == []
    assert out["degradations"] == []


# ── Cache ─────────────────────────────────────────────────────────────────────

def test_cache_roundtrip(tmp_path):
    base = str(tmp_path)
    results = [{"text": "hello", "url": "u", "domain": "d", "trust": 0.9}]

    assert evidence_cache.get("wikipedia", "some query", base_dir=base) is None
    evidence_cache.put("wikipedia", "some query", results, base_dir=base)
    assert evidence_cache.get("wikipedia", "some query", base_dir=base) == results


def test_cache_key_normalises_whitespace_and_case():
    assert evidence_cache.cache_key("wikipedia", "Some  Query") == \
           evidence_cache.cache_key("wikipedia", "some query")


def test_cache_key_separates_providers():
    assert evidence_cache.cache_key("wikipedia", "q") != evidence_cache.cache_key("newsapi", "q")


def test_expired_entry_is_a_miss(tmp_path):
    base = str(tmp_path)
    evidence_cache.put("wikipedia", "q", [{"text": "x"}], base_dir=base)

    path = evidence_cache._path_for(evidence_cache.cache_key("wikipedia", "q"), base)
    with open(path, "r+", encoding="utf-8") as fh:
        payload = json.load(fh)
        payload["fetched_at"] = 0            # 1970
        fh.seek(0); fh.truncate(); json.dump(payload, fh)

    assert evidence_cache.get("wikipedia", "q", ttl_days=7, base_dir=base) is None
    # ttl_days=0 means "never expire" — the mode evaluation replays use.
    assert evidence_cache.get("wikipedia", "q", ttl_days=0, base_dir=base) is not None


def test_corrupt_entry_is_a_miss_not_an_error(tmp_path):
    base = str(tmp_path)
    path = evidence_cache._path_for(evidence_cache.cache_key("wikipedia", "q"), base)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("{not valid json")

    assert evidence_cache.get("wikipedia", "q", base_dir=base) is None
