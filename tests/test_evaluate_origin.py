"""
Origin-evaluation harness tests.

The harness produces the numbers quoted for the origin trail, so its scoring
must be right before any run is believed: a lenient claimant matcher would
inflate precision, a strict one would hide a working feature. Offline; the
pipeline is never run.
"""
from __future__ import annotations

import importlib.util
import os

import pytest

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "evaluate_origin.py")
_spec = importlib.util.spec_from_file_location("evaluate_origin", _PATH)
eo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eo)


# ── Claimant matching ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("gold,predicted,expected", [
    ("facebook-posts", "Facebook posts", True),
    ("facebook-posts", "Viral Facebook post", True),
    ("facebook-posts", "Instagram post", False),
    ("chain-email", "Chain email", True),
    ("chain-email", "Facebook post", False),
    ("tweets", "Tweet", True),
    ("blog-posting", "Bloggers", True),
    ("viral-image", "Viral image", True),
    ("viral-image", "Instagram post", True),       # images circulate on platforms
    ("viral-image", "Chain email", False),
    ("donald-trump", "Donald Trump", True),
    ("donald-trump", "Ivanka Trump", False),
    ("rick-perry", "Facebook posts", False),
    ("facebook-posts", None, False),
    ("facebook-posts", "", False),
])
def test_claimant_matches(gold, predicted, expected):
    assert eo.claimant_matches(gold, predicted) is expected


def test_same_site_allows_subdomains_only():
    assert eo.same_site("www.yournewswire.com", "yournewswire.com")
    assert eo.same_site("m.abcnews.com.co", "abcnews.com.co")
    assert not eo.same_site("abcnews.com.co", "abcnews.com")   # the classic imposter
    assert not eo.same_site("", "x.com")
    assert eo.same_site("uspoln.com:80", "uspoln.com")


# ── Folding and scoring ───────────────────────────────────────────────────────

def _origin(claimant=None, earliest=None, timeline=(), basis="none", status="ok"):
    return {"claimant": claimant, "basis": basis, "fact_check_status": status,
            "earliest_seen": {"domain": earliest[0], "date": earliest[1]} if earliest else None,
            "timeline": [{"domain": d} for d in timeline]}


def test_summarise_takes_the_earliest_across_claims():
    cvs = [
        {"verdict": "contradiction", "origin": _origin(earliest=("late.com", "2017-05-02"),
                                                       timeline=["late.com"], basis="earliest_indexed")},
        {"verdict": "neutral", "origin": _origin(claimant="Facebook posts",
                                                 earliest=("early.com", "2017-05-01"),
                                                 timeline=["early.com", "late.com"], basis="fact_check")},
        {"verdict": "entailment"},
    ]
    s = eo.summarise_origins(cvs)
    assert s["claimant"] == "Facebook posts"
    assert s["earliest_domain"] == "early.com"
    assert s["timeline_domains"] == ["late.com", "early.com"]
    assert s["basis"] == "fact_check"


def test_summarise_with_no_origin():
    s = eo.summarise_origins([{"verdict": "entailment"}])
    assert s == {"has_origin": False, "basis": "none", "fact_check_status": "not_run",
                 "claimant": None, "earliest_domain": None, "timeline_domains": [],
                 "domains_without_search": [], "wider_hits": 0, "search_status": "not_run"}


def test_publisher_scoring():
    row = {"slice": "publishers", "gold": "yournewswire.com"}
    found_not_first = {"basis": "earliest_indexed", "claimant": None,
                       "earliest_domain": "snopes.com",
                       "timeline_domains": ["snopes.com", "www.yournewswire.com"]}
    s = eo.score(row, found_not_first)
    assert s == {"traced": True, "publisher_found": True,
                 "publisher_found_without_search": False, "earliest_correct": False}


def test_summarise_separates_what_the_wider_search_added():
    timeline = [{"domain": "thehindu.com", "found_by": "evidence"},
                {"domain": "yournewswire.com", "found_by": "wider_search"}]
    s = eo.summarise_origins([{"origin": {"basis": "earliest_indexed", "timeline": timeline,
                                          "search_status": "ok"}}])
    assert s["timeline_domains"] == ["thehindu.com", "yournewswire.com"]
    assert s["domains_without_search"] == ["thehindu.com"]
    assert s["wider_hits"] == 1 and s["search_status"] == "ok"
    row = {"slice": "publishers", "gold": "yournewswire.com"}
    scored = eo.score(row, s)
    assert scored["publisher_found"] and not scored["publisher_found_without_search"]


def test_report_precision_counts_only_named_claimants():
    recs = [
        {"slice": "claimants", "gold": "facebook-posts", "traced": True,
         "claimant_named": True, "claimant_correct": True, "fact_check_status": "ok"},
        {"slice": "claimants", "gold": "facebook-posts", "traced": True,
         "claimant_named": True, "claimant_correct": False, "fact_check_status": "ok"},
        {"slice": "claimants", "gold": "donald-trump", "traced": False,
         "claimant_named": False, "claimant_correct": False, "fact_check_status": "ok"},
        {"slice": "claimants", "gold": "donald-trump", "error": "RuntimeError"},
    ]
    r = eo.build_report("claimants", recs)
    assert r["errors"] == 1
    assert (r["claimant_precision"]["k"], r["claimant_precision"]["n"]) == (1, 2)
    assert (r["claimant_recall"]["k"], r["claimant_recall"]["n"]) == (1, 3)
    assert r["claimant_recall_channel"]["n"] == 2 and r["claimant_recall_person"]["n"] == 1


def test_wilson_interval():
    assert eo.wilson(0, 0) == (0.0, 0.0)
    lo, hi = eo.wilson(5, 10)
    assert lo < 0.5 < hi and 0.2 < lo and hi < 0.8
    assert eo.wilson(10, 10)[1] == 1.0


def test_disabled_lookup_is_called_out():
    recs = [{"slice": "claimants", "gold": "tweets", "traced": False, "claimant_named": False,
             "claimant_correct": False, "fact_check_status": "disabled"}]
    assert "GOOGLE_FACTCHECK_API_KEY" in eo.render_markdown(eo.build_report("claimants", recs))


# ── Loaders (real dataset files) ──────────────────────────────────────────────

_HAS_DATA = os.path.exists(eo.FAKENEWSNET_PATH) and os.path.isdir(eo.LIAR_DIR)


@pytest.mark.skipif(not _HAS_DATA, reason="datasets not present")
def test_load_publishers_is_deterministic_with_domain_gold():
    a, b = eo.load_publishers(20), eo.load_publishers(20)
    assert [r["id"] for r in a] == [r["id"] for r in b] and len(a) == 20
    assert all(r["gold"] and "/" not in r["gold"] for r in a)


@pytest.mark.skipif(not _HAS_DATA, reason="datasets not present")
def test_load_claimants_balances_channels_and_people():
    rows = eo.load_claimants(20)
    channel = [r for r in rows if r["gold"] in eo.CHANNEL_SPEAKERS]
    assert len(rows) == 20 and len(channel) == 10
    assert all(r["claim"] for r in rows)


@pytest.mark.parametrize("url,expected", [
    ("web.archive.org/web/20171016014242/http://nypost.com/2017/x", "http://nypost.com/2017/x"),
    ("https://web.archive.org/web/2018/https://www.cnn.com/a", "https://www.cnn.com/a"),
    ("speedtalk.com/forum/x", "http://speedtalk.com/forum/x"),
])
def test_unwrap_archive(url, expected):
    assert eo._unwrap_archive(url) == expected
