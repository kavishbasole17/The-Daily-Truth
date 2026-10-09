"""
Regional-retrieval tests.

The property that matters most is *additivity*: regional search may add
evidence, never remove or reorder what the global search found, and a claim
with no location must take exactly the path it always did. Providers are
stubbed; nothing here touches the network.
"""
from __future__ import annotations

import pytest

from paper_trail_core.agents import evidence_retriever as ER

CLAIM = "Heavy rain caused flooding after the Khadakwasla dam released water into the river."

PUNE = {"name": "Pune", "label": "Pune, Maharashtra, India", "level": "city", "city": "Pune",
        "state": "Maharashtra", "country": "India", "country_code": "in", "confidence": 0.95}
KOREGAON = {"name": "Koregaon Park", "label": "Koregaon Park, Pune, Maharashtra, India",
            "level": "locality", "locality": "Koregaon Park", "city": "Pune",
            "state": "Maharashtra", "country": "India", "country_code": "in", "confidence": 0.8}

GLOBAL_HIT = {"text": "Khadakwasla Dam: a dam on the Mutha River near Pune.",
              "url": "https://en.wikipedia.org/wiki/Khadakwasla_Dam", "domain": "en.wikipedia.org",
              "trust": 0.85, "provider": "wikipedia", "published_at": ""}


def _regional(url, text="Khadakwasla dam discharge floods parts of Pune after heavy rain.",
              domain="thehindu.com", trust=0.9):
    return {"text": text, "url": url, "domain": domain, "trust": trust,
            "provider": "gnews_rss", "published_at": ""}


@pytest.fixture(autouse=True)
def _stubs(monkeypatch):
    """Global search returns one Wikipedia page; the relevance gate is a pass-through."""
    from config import settings
    monkeypatch.setattr(settings, "EVIDENCE_BACKEND", "live")
    monkeypatch.setattr(ER, "_BACKEND", "live")
    monkeypatch.setattr(settings, "REGION_AWARE", True)
    monkeypatch.setattr(settings, "REGIONAL_NEWS_PROVIDER", "gnews_rss")
    monkeypatch.setattr(ER, "_search_cached", lambda p, q: ([dict(GLOBAL_HIT)], "ok", False))

    def _gate(claim, candidates, min_cos=0.0):
        for c in candidates:
            c["relevance"] = 0.8
        return candidates[: ER.MAX_SNIPPETS_PER_CLAIM]
    monkeypatch.setattr(ER, "gate_by_relevance", _gate)


# ── Additivity ────────────────────────────────────────────────────────────────

def test_claim_without_a_location_takes_the_original_path(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("regional search ran for a claim with no location")
    monkeypatch.setattr(ER, "_search_regional_cached", _boom)

    out = ER.retrieve_for_claim(CLAIM, 0)
    assert "location" not in out and "region" not in out
    assert [s["url"] for s in out["snippets"]] == [GLOBAL_HIT["url"]]


def test_regional_evidence_is_added_and_ordered_by_weight(monkeypatch):
    monkeypatch.setattr(ER, "_search_regional_cached",
                        lambda p, q, cc: ([_regional("https://thehindu.com/a"),
                                           _regional("https://ndtv.com/b", domain="ndtv.com", trust=0.8)], "ok"))
    base = ER.retrieve_for_claim(CLAIM, 0)
    out = ER.retrieve_for_claim(CLAIM, 0, location=PUNE)

    urls = [s["url"] for s in out["snippets"]]
    # Nothing retrieved globally is dropped, and the regional items are added.
    assert set(s["url"] for s in base["snippets"]) <= set(urls)
    assert {"https://thehindu.com/a", "https://ndtv.com/b"} <= set(urls)
    # Best-weighted first: trust x relevance never increases down the list.
    weights = [s["trust"] * s["relevance"] for s in out["snippets"]]
    assert weights == sorted(weights, reverse=True)
    assert out["region"]["regional_snippets"] == 2


def test_a_regional_duplicate_of_a_global_url_is_not_added_twice(monkeypatch):
    monkeypatch.setattr(ER, "_search_regional_cached",
                        lambda p, q, cc: ([_regional(GLOBAL_HIT["url"])], "ok"))
    out = ER.retrieve_for_claim(CLAIM, 0, location=PUNE)
    assert [s["url"] for s in out["snippets"]].count(GLOBAL_HIT["url"]) == 1


def test_regional_snippets_are_capped(monkeypatch):
    from config import settings
    monkeypatch.setattr(settings, "REGION_MAX_EXTRA_SNIPPETS", 2)
    many = [_regional(f"https://thehindu.com/{i}") for i in range(8)]
    monkeypatch.setattr(ER, "_search_regional_cached", lambda p, q, cc: (many, "ok"))

    def _no_truncate(claim, candidates, min_cos=0.0):          # isolate the extra-snippet cap
        for c in candidates:
            c["relevance"] = 0.8
        return candidates
    monkeypatch.setattr(ER, "gate_by_relevance", _no_truncate)

    out = ER.retrieve_for_claim(CLAIM, 0, location=PUNE)
    room = ER.MAX_SNIPPETS_PER_CLAIM - 1                    # one global snippet
    assert len(out["snippets"]) == 1 + room + 2


def test_regional_evidence_can_rescue_a_claim_global_search_missed(monkeypatch):
    monkeypatch.setattr(ER, "_search_cached", lambda p, q: ([], "ok", False))
    monkeypatch.setattr(ER, "_search_regional_cached",
                        lambda p, q, cc: ([_regional("https://thehindu.com/a")], "ok"))
    assert ER.retrieve_for_claim(CLAIM, 0)["status"] == "no_hits"
    rescued = ER.retrieve_for_claim(CLAIM, 0, location=PUNE)
    assert rescued["status"] == "ok" and len(rescued["snippets"]) == 1


def test_no_regional_hits_leaves_the_global_result_unchanged(monkeypatch):
    monkeypatch.setattr(ER, "_search_regional_cached", lambda p, q, cc: ([], "ok"))
    base = ER.retrieve_for_claim(CLAIM, 0)
    out = ER.retrieve_for_claim(CLAIM, 0, location=PUNE)
    assert [s["url"] for s in out["snippets"]] == [s["url"] for s in base["snippets"]]
    assert out["status"] == base["status"]
    assert out["region"]["status"] == "no_hits" and out["region"]["regional_snippets"] == 0


# ── Widening ──────────────────────────────────────────────────────────────────

def test_search_widens_until_enough_regional_evidence(monkeypatch):
    queries = []

    def _search(provider, query, cc):
        queries.append(query)
        if query.startswith("Koregaon Park Pune"):
            return [], "ok"
        if query.startswith("Pune"):
            return [_regional("https://thehindu.com/1")], "ok"
        return [_regional("https://thehindu.com/2"), _regional("https://ndtv.com/3", domain="ndtv.com")], "ok"
    monkeypatch.setattr(ER, "_search_regional_cached", _search)

    _, info = ER.regional_evidence(CLAIM, KOREGAON)
    assert info["widened"] is True
    assert info["scopes_tried"][0] == "Koregaon Park Pune (locality)"
    assert info["scope_used"].endswith("(state)") or info["scope_used"].endswith("(city)")
    assert len(queries) >= 2


def test_search_stops_widening_once_it_has_enough(monkeypatch):
    queries = []

    def _search(provider, query, cc):
        queries.append(query)
        return [_regional(f"https://thehindu.com/{len(queries)}-{i}") for i in range(3)], "ok"
    monkeypatch.setattr(ER, "_search_regional_cached", _search)

    _, info = ER.regional_evidence(CLAIM, KOREGAON)
    assert len(queries) == 1 and info["widened"] is False


def test_scope_attempts_are_bounded(monkeypatch):
    from config import settings
    monkeypatch.setattr(settings, "REGION_MAX_QUERIES", 2)
    queries = []
    monkeypatch.setattr(ER, "_search_regional_cached",
                        lambda p, q, cc: (queries.append(q) or [], "ok"))
    ER.regional_evidence(CLAIM, KOREGAON)
    assert len(queries) == 2


def test_place_words_are_removed_from_the_keywords(monkeypatch):
    queries = []
    monkeypatch.setattr(ER, "_search_regional_cached",
                        lambda p, q, cc: (queries.append(q) or [], "ok"))
    ER.regional_evidence("Floods hit Pune after the dam released water.", PUNE)
    assert queries and queries[0].count("Pune") == 1 or queries[0].lower().count("pune") == 1


# ── Switches and failure modes ────────────────────────────────────────────────

def test_region_aware_off_disables_regional_search(monkeypatch):
    from config import settings
    monkeypatch.setattr(settings, "REGION_AWARE", False)

    def _boom(*a, **k):
        raise AssertionError("regional search ran while disabled")
    monkeypatch.setattr(ER, "_search_regional_cached", _boom)
    out = ER.retrieve_for_claim(CLAIM, 0, location=PUNE)
    assert out["region"]["status"] == "disabled"
    assert [s["url"] for s in out["snippets"]] == [GLOBAL_HIT["url"]]


def test_provider_none_disables_regional_search(monkeypatch):
    from config import settings
    monkeypatch.setattr(settings, "REGIONAL_NEWS_PROVIDER", "none")
    assert ER.regional_provider() is None


def test_newsdata_needs_a_key(monkeypatch):
    from config import settings
    monkeypatch.setattr(settings, "REGIONAL_NEWS_PROVIDER", "newsdata")
    monkeypatch.setattr(settings, "NEWSDATA_API_KEY", None)
    assert ER.regional_provider() == "gnews_rss"
    monkeypatch.setattr(settings, "NEWSDATA_API_KEY", "k")
    assert ER.regional_provider() == "newsdata"


def test_provider_failure_degrades_to_global_results(monkeypatch):
    monkeypatch.setattr(ER, "_search_regional_cached", lambda p, q, cc: ([], "error"))
    out = ER.retrieve_for_claim(CLAIM, 0, location=PUNE)
    assert out["status"] == "ok" and len(out["snippets"]) == 1
    assert out["region"]["status"] == "error"


def test_cache_only_backend_never_hits_the_network(monkeypatch, tmp_path):
    from config import settings
    monkeypatch.setattr(ER, "_BACKEND", "cache_only")
    monkeypatch.setattr(settings, "EVIDENCE_CACHE_DIR", str(tmp_path))

    def _boom(*a, **k):
        raise AssertionError("network call under cache_only")
    monkeypatch.setitem(ER._REGIONAL_MAP, "gnews_rss", _boom)
    results, status = ER._search_regional_cached("gnews_rss", "Pune flood", "in")
    assert results == [] and status == "ok"


def test_regional_results_are_cached_per_country(monkeypatch, tmp_path):
    from paper_trail_core.agents import evidence_cache
    monkeypatch.setattr(evidence_cache, "_DIR", str(tmp_path))
    calls = []
    monkeypatch.setitem(ER._REGIONAL_MAP, "gnews_rss",
                        lambda q, cc: calls.append((q, cc)) or [_regional("https://thehindu.com/a")])

    ER._search_regional_cached("gnews_rss", "Pune flood", "in")
    ER._search_regional_cached("gnews_rss", "Pune flood", "in")
    ER._search_regional_cached("gnews_rss", "Pune flood", "gb")
    assert calls == [("Pune flood", "in"), ("Pune flood", "gb")]


# ── Region tagging ────────────────────────────────────────────────────────────

def test_snippet_mentioning_the_place_is_in_region():
    assert ER._is_in_region({"text": "Floods in Pune worsen", "domain": "example.com"}, PUNE)
    assert not ER._is_in_region({"text": "Floods in Lisbon worsen", "domain": "example.com"}, PUNE)


def test_country_tld_counts_but_dot_com_does_not():
    assert ER._is_in_region({"text": "Dam released water", "domain": "mid-day.in"}, PUNE)
    assert not ER._is_in_region({"text": "Dam released water", "domain": "mid-day.com"}, PUNE)


def test_place_name_must_match_as_a_whole_word():
    assert not ER._is_in_region({"text": "The Punekar tradition", "domain": "x.com"}, PUNE)


# ── Providers ─────────────────────────────────────────────────────────────────

_RSS = b"""<?xml version="1.0" encoding="UTF-8"?><rss version="2.0"><channel>
<item><title>Khadakwasla Dam discharge floods Pune - The Hindu</title>
<link>https://news.google.com/rss/articles/abc</link>
<description>&lt;a href="x"&gt;Khadakwasla Dam discharge floods Pune&lt;/a&gt;</description>
<pubDate>Mon, 17 Aug 2026 07:00:00 GMT</pubDate>
<source url="https://www.thehindu.com">The Hindu</source></item>
<item><title></title><link>https://x</link></item>
</channel></rss>"""


class _Resp:
    def __init__(self, content=b"", status=200, payload=None):
        self.content, self.status_code, self._payload = content, status, payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("http error")


def test_google_news_rss_is_parsed(monkeypatch):
    seen = {}

    def _get(url, params=None, headers=None, timeout=None):
        seen["params"] = params
        return _Resp(_RSS)
    monkeypatch.setattr("requests.get", _get)
    monkeypatch.setattr(ER.time, "sleep", lambda *_: None)

    out = ER._search_gnews_rss("Pune flood dam", "in")
    assert len(out) == 1
    assert out[0]["text"] == "Khadakwasla Dam discharge floods Pune."
    assert out[0]["domain"] == "thehindu.com"
    assert out[0]["trust"] == pytest.approx(0.95)
    assert seen["params"]["gl"] == "IN" and seen["params"]["ceid"] == "IN:en"


def test_feed_with_a_dtd_is_rejected(monkeypatch):
    evil = b'<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa">]><rss><channel/></rss>'
    monkeypatch.setattr("requests.get", lambda *a, **k: _Resp(evil))
    monkeypatch.setattr(ER.time, "sleep", lambda *_: None)
    with pytest.raises(RuntimeError):
        ER._search_gnews_rss("q", "in")


def test_rate_limited_feed_is_reported_as_rate_limited(monkeypatch):
    monkeypatch.setattr("requests.get", lambda *a, **k: _Resp(status=429))
    monkeypatch.setattr(ER.time, "sleep", lambda *_: None)
    with pytest.raises(RuntimeError, match="rate_limited"):
        ER._search_gnews_rss("q", "in")


def test_newsdata_results_are_parsed(monkeypatch):
    from config import settings
    monkeypatch.setattr(settings, "NEWSDATA_API_KEY", "k")
    payload = {"results": [{"title": "Dam opens", "description": "Gates opened in Pune.",
                            "link": "https://www.thehindu.com/x", "pubDate": "2026-08-17"}]}
    monkeypatch.setattr("requests.get", lambda *a, **k: _Resp(payload=payload))
    out = ER._search_newsdata("Pune dam", "in")
    assert out[0]["domain"] == "thehindu.com" and out[0]["provider"] == "newsdata"


# ── The node ──────────────────────────────────────────────────────────────────

def test_node_passes_locations_only_to_located_claims(monkeypatch):
    calls = []

    def _retrieve(claim, idx, location=None):
        calls.append((idx, location))
        return {"claim": claim, "snippets": [dict(GLOBAL_HIT, relevance=0.8)], "status": "ok",
                "cache_hits": 0, "queries": 1, **({"location": location, "region": {
                    "provider": "gnews_rss", "scopes_tried": ["Pune (city)"], "scope_used": "Pune (city)",
                    "widened": False, "status": "ok", "regional_snippets": 1}} if location else {})}
    monkeypatch.setattr(ER, "retrieve_for_claim", _retrieve)

    out = ER.retrieve_evidence_node({"claims": ["claim a", "claim b"],
                                     "claim_locations": [PUNE, None]})
    assert sorted(calls, key=lambda c: c[0]) == [(0, PUNE), (1, None)]
    assert out["evidence"][0]["location"] == PUNE and "location" not in out["evidence"][1]
    assert out["retrieval_stats"]["n_located"] == 1
    assert out["retrieval_stats"]["n_regional_snippets"] == 1


def test_node_without_locations_calls_the_two_argument_form(monkeypatch):
    """Old stubs (and old callers) take (claim, idx) only."""
    monkeypatch.setattr(ER, "retrieve_for_claim",
                        lambda claim, idx: {"claim": claim, "snippets": [], "status": "no_hits",
                                            "cache_hits": 0, "queries": 0})
    out = ER.retrieve_evidence_node({"claims": ["claim a"]})
    assert out["evidence"][0]["status"] == "no_hits"


def test_node_reports_widening_and_missing_regional_coverage(monkeypatch):
    def _retrieve(claim, idx, location=None):
        widened = idx == 0
        return {"claim": claim, "snippets": [dict(GLOBAL_HIT, relevance=0.8)], "status": "ok",
                "cache_hits": 0, "queries": 1, "location": location, "region": {
                    "provider": "gnews_rss", "scopes_tried": ["Koregaon Park Pune (locality)", "Pune (city)"],
                    "scope_used": "Pune (city)" if widened else None, "widened": widened,
                    "status": "ok" if widened else "no_hits",
                    "regional_snippets": 2 if widened else 0}}
    monkeypatch.setattr(ER, "retrieve_for_claim", _retrieve)

    out = ER.retrieve_evidence_node({"claims": ["a", "b"], "claim_locations": [KOREGAON, PUNE]})
    kinds = {d["kind"] for d in out["degradations"]}
    assert {"region_widened", "no_regional_coverage"} <= kinds
    assert all(d["severity"] == "info" for d in out["degradations"]
               if d["kind"] in ("region_widened", "no_regional_coverage"))
