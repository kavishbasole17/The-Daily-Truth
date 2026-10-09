from datetime import datetime, timezone

from paper_trail_core.nlp.event_match import corroboration, event_terms

CLAIM = ("CJP Protest Live Updates: Delhi Metro to shut down 57 stations ahead of "
         "CJP's Oct 10 protest, from 9pm on Friday till Saturday")
D = "Fri, 24 Jul 2026 07:00:00 GMT"
NOW = datetime(2026, 7, 25, tzinfo=timezone.utc)
NEWS = [
    {"domain": "timesofindia.indiatimes.com", "published_at": D,
     "text": "CJP protest: Delhi Metro shuts 17 stations for third day; see full list."},
    {"domain": "thehindu.com", "published_at": D,
     "text": "Delhi Metro: Full list of 17 stations closed amid CJP-led protests."},
    {"domain": "timesofindia.indiatimes.com", "published_at": D,
     "text": "CJP protest continues: Delhi Metro curbs to continue for 4th straight day; "
             "18 stations to remain shut fro."},
]


def test_event_terms_drop_time_words_and_keep_acronyms():
    terms, acronyms = event_terms(CLAIM)
    assert {"cjp", "delhi", "metro", "station", "protest", "shut"} <= terms
    assert not terms & {"friday", "oct", "live", "updates", "9pm"}
    assert acronyms == {"cjp"}


def test_real_story_is_corroborated_and_flags_changed_figures():
    c = corroboration(CLAIM, NEWS, now=NOW)
    assert c["corroborated"] and c["detail_mismatch"]
    assert c["sources"] == ["thehindu.com", "timesofindia.indiatimes.com"]


def test_stale_reports_do_not_corroborate():
    late = datetime(2026, 10, 9, tzinfo=timezone.utc)
    assert not corroboration(CLAIM, NEWS, now=late)["corroborated"]


def test_fabricated_variants_are_not_corroborated():
    for fake in (
        "Delhi Metro to shut all 57 stations permanently from 9pm on Friday amid CJP protest",
        "PM resigns ahead of CJP protest on Friday, Delhi Metro stations shut",
        "Delhi Metro to shut 57 stations for the Taylor Swift concert on Friday",
    ):
        assert not corroboration(fake, NEWS, now=NOW)["corroborated"], fake


def test_single_outlet_or_wikipedia_does_not_corroborate():
    one = [dict(s, domain="thehindu.com") for s in NEWS]
    assert not corroboration(CLAIM, one, now=NOW)["corroborated"]
    wiki = [dict(s, domain="en.wikipedia.org") for s in NEWS]
    assert not corroboration(CLAIM, wiki, now=NOW)["corroborated"]


def test_adjudicator_calls_corroborated_event_real_at_capped_confidence():
    from paper_trail_core.agents.adjudicator import compute_claim_verdict

    cv = {"claim": CLAIM, "verdict": "neutral", "confidence": 0.8, "support": 0.1,
          "entail_mass": 0.1, "contradict_mass": 0.0, "top_relevance": 0.9,
          "numeric_conflict": 0.0, "sources": [], "in_region": [], "status": "ok",
          "evidence_snippets": ["x"], "event_corroborated": True}
    out = compute_claim_verdict(0.5, [cv])
    assert out["verdict"] == "likely_real" and out["confidence"] <= 0.65
    cv["event_corroborated"] = False
    assert compute_claim_verdict(0.5, [cv])["verdict"] == "uncertain"
