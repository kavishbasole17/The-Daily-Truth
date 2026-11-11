"""
Origin trail tests.

The trail is report-only, so the properties pinned here are about honesty
rather than accuracy: a fact-check about a *different* claim must never name
the claimant, an undated source must never be called the earliest, and the
verdict must come out of the node exactly as it went in.

Offline: the embedder is stubbed and the cache points at a temp directory.
"""
from __future__ import annotations

import pytest

from paper_trail_core.agents import evidence_cache
from paper_trail_core.agents import origin_tracer as ot
from paper_trail_core.nlp import embeddings as emb


@pytest.fixture(autouse=True)
def _offline(monkeypatch, tmp_path):
    monkeypatch.setattr(evidence_cache, "_DIR", str(tmp_path))
    monkeypatch.setattr(ot.settings, "GOOGLE_FACTCHECK_API_KEY", None, raising=False)
    monkeypatch.setattr(ot.settings, "EVIDENCE_BACKEND", "live", raising=False)
    monkeypatch.setattr(ot.settings, "ORIGIN_TRACE", True, raising=False)


def _similarity(monkeypatch, scores: dict[str, float]):
    """cosine_to stub: score per reviewed-claim text, 0 for anything else."""
    monkeypatch.setattr(emb, "cosine_to", lambda _c, texts: [scores.get(t, 0.0) for t in texts])


def meta(domain, stance, published_at="", trust=0.5):
    return {"url": f"https://{domain}/a", "domain": domain, "published_at": published_at,
            "provider": "newsapi", "trust": trust, "stance": stance, "stance_score": 0.9}


def factcheck(claim_text="Drinking hot water cures the virus.", claimant="WhatsApp forward",
              site="altnews.in", rating="False", review_date="2026-10-30T00:00:00Z"):
    return {"claim_text": claim_text, "claimant": claimant, "claim_date": "2026-10-27T00:00:00Z",
            "publisher": "Alt News", "site": site, "url": f"https://{site}/fc",
            "rating": rating, "rating_class": ot.classify_rating(rating),
            "review_date": review_date, "relevance": 0.9, "trust": 0.9}


CLAIM = "Drinking hot water cures the virus."


def contradicted(evidence_meta=()):
    return {"claim": CLAIM, "verdict": "contradiction", "confidence": 0.8,
            "evidence_snippets": ["x"] * len(evidence_meta), "support": -0.6,
            "evidence_meta": list(evidence_meta)}


# ── Ratings and parsing ───────────────────────────────────────────────────────

@pytest.mark.parametrize("rating,expected", [
    ("False", "false"), ("Fake", "false"), ("Pants on Fire!", "false"),
    ("Not true", "false"), ("Misleading", "misleading"), ("Partly false", "misleading"),
    ("True", "true"), ("Correct", "true"), ("Unproven", "other"),
])
def test_classify_rating(rating, expected):
    assert ot.classify_rating(rating) == expected


def test_parse_flattens_one_row_per_review():
    payload = {"claims": [{
        "text": CLAIM, "claimant": "Facebook post", "claimDate": "2026-10-27T00:00:00Z",
        "claimReview": [
            {"publisher": {"name": "BOOM", "site": "boomlive.in"}, "url": "https://boomlive.in/x",
             "textualRating": "False", "reviewDate": "2026-10-29T00:00:00Z"},
            {"publisher": {"name": "Factly"}, "url": "https://factly.in/y",
             "textualRating": "Misleading"},
        ],
    }]}
    rows = ot._parse_factchecks(payload)
    assert [r["site"] for r in rows] == ["boomlive.in", "factly.in"]
    assert all(r["claimant"] == "Facebook post" for r in rows)
    assert [r["rating_class"] for r in rows] == ["false", "misleading"]
    assert ot._parse_factchecks({}) == []


# ── Matching ──────────────────────────────────────────────────────────────────

def test_factcheck_about_a_different_claim_is_dropped(monkeypatch):
    same, other = factcheck(), factcheck(claim_text="Hot water is good for colds.")
    _similarity(monkeypatch, {same["claim_text"]: 0.85, other["claim_text"]: 0.45})
    kept = ot.match_factchecks(CLAIM, [other, same])
    assert [k["claim_text"] for k in kept] == [same["claim_text"]]
    assert kept[0]["trust"] == pytest.approx(0.9)


def test_no_similarity_means_no_attribution(monkeypatch):
    def boom(*_a):
        raise RuntimeError("embedder unavailable")
    monkeypatch.setattr(emb, "cosine_to", boom)
    assert ot.match_factchecks(CLAIM, [factcheck()]) == []


# ── Lookup ────────────────────────────────────────────────────────────────────

def test_lookup_disabled_without_key(monkeypatch):
    def no_network(*_a, **_k):
        raise AssertionError("network used")
    monkeypatch.setattr(ot, "_fetch_factchecks", no_network)
    assert ot.search_factchecks(CLAIM) == ([], "disabled")


def test_lookup_disabled_when_backend_not_live(monkeypatch):
    monkeypatch.setattr(ot.settings, "GOOGLE_FACTCHECK_API_KEY", "k")
    monkeypatch.setattr(ot.settings, "EVIDENCE_BACKEND", "cache_only")
    assert ot.search_factchecks(CLAIM) == ([], "disabled")


def test_lookup_is_cached(monkeypatch):
    monkeypatch.setattr(ot.settings, "GOOGLE_FACTCHECK_API_KEY", "k")
    calls = []
    monkeypatch.setattr(ot, "_fetch_factchecks", lambda q, k: calls.append(q) or [factcheck()])
    first = ot.search_factchecks(CLAIM)
    second = ot.search_factchecks(CLAIM)
    assert first == second and first[1] == "ok"
    assert len(calls) == 1


def test_lookup_failure_is_reported_not_raised(monkeypatch):
    monkeypatch.setattr(ot.settings, "GOOGLE_FACTCHECK_API_KEY", "k")
    def down(*_a):
        raise ConnectionError("down")
    monkeypatch.setattr(ot, "_fetch_factchecks", down)
    assert ot.search_factchecks(CLAIM) == ([], "error")


# ── Assembly ──────────────────────────────────────────────────────────────────

def test_timeline_is_dated_oldest_first_with_undated_last():
    cv = contradicted([
        meta("late.in", "repeats", "2026-10-29T10:00:00Z"),
        meta("undated.in", "repeats"),
        meta("early.in", "repeats", "Tue, 28 Oct 2026 08:00:00 GMT"),
        meta("reuters.com", "neutral", "2026-10-20T00:00:00Z"),
    ])
    origin = ot.build_origin(cv, [], "disabled")
    assert [e["domain"] for e in origin["timeline"]] == ["early.in", "late.in", "undated.in"]
    assert origin["earliest_seen"]["domain"] == "early.in"
    assert origin["basis"] == "earliest_indexed"


def test_undated_source_is_never_called_earliest():
    origin = ot.build_origin(contradicted([meta("undated.in", "repeats")]), [], "disabled")
    assert origin["earliest_seen"] is None
    assert origin["basis"] == "none"
    assert [e["domain"] for e in origin["timeline"]] == ["undated.in"]


def test_factcheck_claimant_and_first_debunk():
    cv = contradicted([meta("viral.in", "repeats", "2026-10-28T00:00:00Z"),
                       meta("thehindu.com", "debunks", "2026-10-31T00:00:00Z")])
    origin = ot.build_origin(cv, [factcheck()], "ok")
    assert origin["basis"] == "fact_check"
    assert origin["claimant"] == "WhatsApp forward"
    assert origin["claim_date"] == "2026-10-27"
    assert origin["claimant_source"] == "altnews.in"
    assert origin["first_debunk"]["domain"] == "altnews.in"     # 30th, before the Hindu's 31st
    assert [e["kind"] for e in origin["timeline"]] == ["repeats", "fact_check", "debunks"]


def test_true_rating_is_not_a_debunk():
    origin = ot.build_origin(contradicted(), [factcheck(rating="True")], "ok")
    assert origin["first_debunk"] is None


# ── Node ──────────────────────────────────────────────────────────────────────

def test_node_leaves_verdicts_untouched_and_skips_supported_claims(monkeypatch):
    _similarity(monkeypatch, {})
    supported = {"claim": "Tokyo is the capital of Japan.", "verdict": "entailment",
                 "confidence": 0.9, "evidence_snippets": [], "evidence_meta": []}
    refuted = contradicted([meta("viral.in", "repeats", "2026-10-28T00:00:00Z")])
    state = {"claim_verdicts": [supported, refuted]}

    out = ot.trace_origin_node(state)

    a, b = out["claim_verdicts"]
    assert "origin" not in a
    assert b["origin"]["earliest_seen"]["domain"] == "viral.in"
    for before, after in zip(state["claim_verdicts"], out["claim_verdicts"]):
        assert {k: v for k, v in after.items() if k != "origin"} == before
    assert "origin" not in state["claim_verdicts"][1]           # input not mutated
    assert any(d["kind"] == "factcheck_disabled" for d in out["degradations"])


def test_node_caps_lookups_but_traces_every_unsupported_claim(monkeypatch):
    monkeypatch.setattr(ot.settings, "ORIGIN_MAX_CLAIMS", 1, raising=False)
    _similarity(monkeypatch, {})
    looked = []
    monkeypatch.setattr(ot, "search_factchecks", lambda c: looked.append(c) or ([], "ok"))
    state = {"claim_verdicts": [contradicted(), contradicted()]}
    out = ot.trace_origin_node(state)
    assert len(looked) == 1
    assert [cv["origin"]["fact_check_status"] for cv in out["claim_verdicts"]] == ["ok", "skipped"]


def test_node_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(ot.settings, "ORIGIN_TRACE", False)
    assert ot.trace_origin_node({"claim_verdicts": [contradicted()]}) == {}


# ── Trace ─────────────────────────────────────────────────────────────────────

def test_trace_prints_origin_trail():
    from paper_trail_core.agents.adjudicator import _build_trace

    cv = contradicted([meta("viral.in", "repeats", "2026-10-28T00:00:00Z")])
    cv["origin"] = ot.build_origin(cv, [factcheck()], "ok")
    trace = _build_trace({}, 0.2, "learned", "likely_fake", 0.8, [cv], 0.35, 0.65, [])

    assert "ORIGIN TRAIL (earliest source found, not proof of origin)" in trace
    assert "claimed by WhatsApp forward on 2026-10-27 (per altnews.in fact-check)" in trace
    assert "2026-10-28  viral.in" in trace
    assert "FACT-CHECK: False" in trace


# ── Wider search ──────────────────────────────────────────────────────────────

def hit(domain, text, published_at="", url=None):
    return {"text": text, "url": url or f"https://{domain}/p", "domain": domain,
            "trust": 0.5, "provider": "gnews_rss", "published_at": published_at}


def test_queries_are_claim_then_keywords():
    qs = ot.wider_search_queries("Drinking hot water every hour kills the coronavirus in the throat.")
    assert qs[0].startswith("Drinking hot water")
    assert len(qs) == 2 and qs[1] != qs[0] and len(qs[1].split()) <= 5


def test_wider_search_off_by_setting_and_in_mock(monkeypatch):
    called = []
    monkeypatch.setattr(ot, "_search_one", lambda *a: called.append(a) or ([], "ok"))
    monkeypatch.setattr(ot.settings, "ORIGIN_WIDE_SEARCH", False)
    assert ot.search_wider(CLAIM, "IN") == ([], "disabled")
    monkeypatch.setattr(ot.settings, "ORIGIN_WIDE_SEARCH", True)
    monkeypatch.setattr(ot.settings, "EVIDENCE_BACKEND", "mock")
    assert ot.search_wider(CLAIM, "IN") == ([], "disabled")
    assert called == []


def test_wider_search_dedupes_and_survives_a_failing_provider(monkeypatch):
    monkeypatch.setattr(ot.settings, "ORIGIN_WIDE_SEARCH", True)
    monkeypatch.setattr(ot.settings, "SERPAPI_KEY", "k", raising=False)
    monkeypatch.setattr(ot.settings, "TAVILY_API_KEY", None, raising=False)
    calls = []

    def fake(provider, query, country):
        calls.append((provider, query, country))
        if provider == "serpapi":
            raise ConnectionError("down")
        return [hit("viral.in", "Hot water kills virus"), hit("viral.in", "dup")], "ok"

    monkeypatch.setattr(ot, "_search_one", fake)
    hits, status = ot.search_wider(CLAIM, "IN")
    assert status == "ok"
    assert len(hits) == 1                                # same URL from both queries
    assert [c[0] for c in calls].count("serpapi") == 1   # paid provider: first query only
    assert all(c[2] == "IN" for c in calls)


def test_cache_only_never_touches_the_network(monkeypatch):
    from paper_trail_core.agents import evidence_retriever as er
    monkeypatch.setattr(ot.settings, "EVIDENCE_BACKEND", "cache_only")
    monkeypatch.setattr(er, "_search_regional_cached",
                        lambda *a: (_ for _ in ()).throw(AssertionError("network")))
    evidence_cache.put("gnews_rss:in", "q", [hit("viral.in", "cached")])
    results, status = ot._search_one("gnews_rss", "q", "IN")
    assert status == "ok" and results[0]["domain"] == "viral.in"


def _stub_models(monkeypatch, sims, results):
    from paper_trail_core.nlp import nli
    monkeypatch.setattr(ot.settings, "NLI_BACKEND", "transformer", raising=False)
    monkeypatch.setattr(nli, "is_available", lambda: True)
    monkeypatch.setattr(emb, "cosine_to", lambda _c, texts: [sims[t] for t in texts])
    monkeypatch.setattr(nli, "infer_batch", lambda prem, hyp: [results[p] for p in prem])


def test_judge_keeps_relevant_hits_with_stance(monkeypatch):
    from paper_trail_core.nlp.nli import NLIResult
    spreads = "Drinking hot water cures the virus, doctors confirm"
    debunks = "No, drinking hot water does not cure the virus"
    factcheck = "Fact check: does drinking hot water cure the virus?"
    _stub_models(monkeypatch,
                 sims={spreads: 0.8, debunks: 0.75, factcheck: 0.85, "Monsoon reaches Kerala": 0.1},
                 results={spreads: NLIResult(0.9, 0.05, 0.05),
                          debunks: NLIResult(0.02, 0.08, 0.9),
                          factcheck: NLIResult(0.7, 0.2, 0.1)})
    hits = [hit("viral.in", spreads, "2026-10-28T00:00:00Z"), hit("thehindu.com", debunks),
            hit("altnews.in", factcheck), hit("x.in", "Monsoon reaches Kerala")]
    judged, problem = ot.judge_hits([(CLAIM, hits), (CLAIM, [])])
    assert problem is None
    assert [(m["domain"], m["stance"]) for m in judged[0]] == [
        ("altnews.in", "fact_check"), ("viral.in", "repeats"), ("thehindu.com", "debunks")]
    assert judged[1] == []


def test_without_nli_hits_are_mentions_never_a_guessed_stance(monkeypatch):
    """Word overlap would call "Fact check: X did not..." a repeat of X; so no stance is guessed."""
    debunk = "No, drinking hot water does not cure the virus"
    factcheck = "Fact check: does drinking hot water cure the virus?"
    monkeypatch.setattr(ot.settings, "NLI_BACKEND", "lexical", raising=False)
    monkeypatch.setattr(emb, "cosine_to", lambda _c, texts: [0.8 for _ in texts])
    judged, problem = ot.judge_hits([(CLAIM, [hit("thehindu.com", debunk),
                                              hit("altnews.in", factcheck),
                                              hit("x.in", "Monsoon reaches Kerala")])])
    assert problem == "no_nli"
    assert sorted((m["domain"], m["stance"]) for m in judged[0]) == [
        ("altnews.in", "fact_check"), ("thehindu.com", "mentions")]


def test_without_any_model_wording_alone_decides(monkeypatch):
    from paper_trail_core.nlp import nli
    def boom(*_a):
        raise RuntimeError("no embedder")
    monkeypatch.setattr(emb, "cosine_to", boom)
    monkeypatch.setattr(nli, "is_available", lambda: False)
    judged, problem = ot.judge_hits([(CLAIM, [hit("viral.in", "Drinking hot water cures the virus"),
                                              hit("x.in", "Monsoon reaches Kerala")])])
    assert problem == "no_models"
    assert [(m["domain"], m["stance"], m["relevance"]) for m in judged[0]] == [
        ("viral.in", "mentions", None)]


def test_mentions_never_become_earliest_or_a_debunk():
    found = [meta("viral.in", "mentions", "2026-10-20T00:00:00Z")]
    origin = ot.build_origin(contradicted(), [], "disabled", found, "no_nli")
    assert origin["timeline"][0]["kind"] == "mentions"
    assert origin["earliest_seen"] is None and origin["first_debunk"] is None
    assert origin["basis"] == "none"


def test_no_nli_fallback_on_held_out_hits():
    """Pinned measurement: without NLI, 10 of 18 kept test rows are about the claim (56%)."""
    kept = about = 0
    for r in _labelled("test"):
        shown = ot.wider_hit_stance(r["claim"], {**r, "stance": None}, r["relevance"])
        if shown:
            kept += 1
            about += r["about"]
    assert (about, kept) == (10, 18)
    kept = about = 0
    for r in _labelled("test"):
        shown = ot.wider_hit_stance(r["claim"], {**r, "stance": None}, None)
        if shown:
            kept += 1
            about += r["about"]
    assert (about, kept) == (11, 24)                  # no models at all: 46%


def test_wider_search_hit_can_become_the_earliest_source():
    cv = contradicted([meta("thehindu.com", "debunks", "2026-10-31T00:00:00Z", trust=0.95)])
    found = [{**meta("viralbharat.co.in", "repeats", "2026-10-26T00:00:00Z", trust=0.65),
              "relevance": 0.8}]
    origin = ot.build_origin(cv, [], "disabled", found, "ok")
    assert origin["earliest_seen"]["domain"] == "viralbharat.co.in"
    assert origin["earliest_seen"]["found_by"] == "wider_search"
    assert origin["search_status"] == "ok"
    assert [e["found_by"] for e in origin["timeline"]] == ["wider_search", "evidence"]


def test_node_runs_wider_search_and_reports_failure(monkeypatch):
    monkeypatch.setattr(ot, "search_wider", lambda claim, country: ([], "error"))
    out = ot.trace_origin_node({"claim_verdicts": [contradicted()]})
    assert out["claim_verdicts"][0]["origin"]["search_status"] == "error"
    assert any(d["kind"] == "wider_search_failed" for d in out["degradations"])


def test_node_uses_the_claims_country(monkeypatch):
    seen = []
    monkeypatch.setattr(ot, "search_wider", lambda claim, country: seen.append(country) or ([], "ok"))
    cv = contradicted(); cv["location"] = {"country_code": "us"}
    ot.trace_origin_node({"claim_verdicts": [cv, contradicted()]})
    assert sorted(seen) == ["IN", "us"]          # searches run in a thread pool


def test_wider_search_weights_fact_checkers_as_such(monkeypatch):
    from paper_trail_core.nlp.nli import NLIResult
    text = "Fact check: drinking hot water does not cure the virus"
    _stub_models(monkeypatch, sims={text: 0.8},
                 results={text: NLIResult(0.02, 0.08, 0.9)})
    judged, _ = ot.judge_hits([(CLAIM, [hit("thequint.com", text,
                                            url="https://news.google.com/rss/articles/CBMi")])])
    assert judged[0][0]["trust"] == pytest.approx(0.8)     # the publisher, not Google


# ── Wider-search filter, against hand-labelled hits ───────────────────────────
#
# dataset/golden/wider_search_rows.jsonl: every hit the unfiltered wider search
# produced for 60 FakeNewsNet stories, labelled by hand before the filter was
# chosen. `about`: the headline is about this claim. `label`: about it, and the
# stance shown is right. The filter was tuned on split=tune and checked once on
# split=test; these tests pin those measured numbers.

import json as _json
import os as _os

_ROWS_PATH = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                           "dataset", "golden", "wider_search_rows.jsonl")


def _labelled(split):
    with open(_ROWS_PATH, encoding="utf-8") as fh:
        return [r for r in map(_json.loads, fh) if r["split"] == split]


def _filter_precision(split):
    kept = right = 0
    for r in _labelled(split):
        shown = ot.wider_hit_stance(r["claim"], r, r["relevance"])
        if shown is None:
            continue
        kept += 1
        # A fact-checker row shown as a fact-check is right whenever it is about the claim.
        right += r["about"] if shown == "fact_check" else r["label"]
    return right, kept


def test_filter_on_tuning_hits():
    assert _filter_precision("tune") == (12, 15)


def test_filter_on_held_out_hits():
    unfiltered = _labelled("test")
    assert sum(r["label"] for r in unfiltered) / len(unfiltered) < 0.2   # 13/72 before
    assert _filter_precision("test") == (9, 18)                           # 50% after


def test_different_event_same_people_is_dropped():
    claim = "Kim Jong Un Dead: North Koreans Calling Trump An Assassin"
    m = {"text": "Kim Jong-nam death: Unravelling the mystery.", "domain": "bbc.com",
         "stance": "debunks", "stance_score": 1.0}
    assert ot.wider_hit_stance(claim, m, 0.62) is None


def test_fact_checker_headline_is_a_fact_check_not_a_repeat():
    claim = "Australia Becomes First Country To Begin Microchipping Its Citizens"
    m = {"text": "Australia Becomes First Country to Begin Microchipping Its Public?",
         "domain": "snopes.com", "stance": "repeats", "stance_score": 0.95}
    assert ot.wider_hit_stance(claim, m, 0.89) == "fact_check"


def test_search_fact_check_row_is_not_counted_as_a_debunk():
    found = [{**meta("snopes.com", "fact_check", "2026-10-29T00:00:00Z", trust=0.85)}]
    origin = ot.build_origin(contradicted(), [], "disabled", found, "ok")
    assert origin["timeline"][0]["kind"] == "fact_check"
    assert origin["timeline"][0]["label"] == "reviewed"
    assert origin["first_debunk"] is None and origin["earliest_seen"] is None


def test_degraded_nli_traces_supported_claims_too(monkeypatch):
    _similarity(monkeypatch, {})
    supported = {"claim": CLAIM, "verdict": "entailment", "confidence": 0.4,
                 "evidence_snippets": [], "evidence_meta": []}
    assert ot.trace_origin_node({"claim_verdicts": [supported]}) == {}
    out = ot.trace_origin_node({"claim_verdicts": [supported], "nli_degraded": True})
    assert "origin" in out["claim_verdicts"][0]


def test_degraded_nli_evidence_rows_are_mentions():
    cv = contradicted([meta("en.wikipedia.org", "repeats", "2026-10-20T00:00:00Z")])
    origin = ot.build_origin(cv, [], "disabled", stance_known=False)
    assert origin["timeline"][0]["kind"] == "mentions"
    assert origin["earliest_seen"] is None
