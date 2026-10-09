"""
Claim-mode tests.

Claim mode is what a person typing one sentence gets: the verdict is decided on
evidence alone (the article-trained style classifier is ignored), and only a page
about the claim's own subject may refute it. These tests pin the rules down, and
pin the boundary that keeps whole articles on the article path.
"""
from __future__ import annotations

import pytest

from paper_trail_core.agents import adjudicator as ADJ
from paper_trail_core.agents import evidence_retriever as ER
from paper_trail_core.agents import pipeline as P
from paper_trail_core.agents import verifier as V
from paper_trail_core.agents.claim_extractor import extract_claims_node, split_compound
from paper_trail_core.nlp.claim_mode import count_sentences, is_claim_text
from paper_trail_core.nlp.lang_id import detect_language
from paper_trail_core.nlp.nli import NLIResult


@pytest.fixture
def claim_mode(monkeypatch):
    from config import settings
    monkeypatch.setattr(settings, "CLAIM_MODE", True)
    return settings


# ── Which path is a text on? ──────────────────────────────────────────────────

def test_sentence_counting_ignores_abbreviations_and_decimals():
    assert count_sentences("Tokyo is the capital of Japan.") == 1
    assert count_sentences("Dr. Singh said the value is 3.5 percent. Prices rose.") == 2
    assert count_sentences("No full stop here") == 1
    assert count_sentences("") == 0


def test_one_or_two_short_sentences_are_claims(claim_mode):
    assert is_claim_text("Tokyo is the capital of Japan.")
    assert is_claim_text("Tokyo is the capital of Japan. It lies on Honshu.")


def test_articles_stay_on_the_article_path(claim_mode):
    three = "Prices rose on Monday. Markets fell on Tuesday. Analysts were surprised on Wednesday."
    assert not is_claim_text(three)
    assert not is_claim_text("A very long sentence. " * 2 + "x" * 400)


def test_claim_mode_can_be_switched_off(monkeypatch):
    from config import settings
    monkeypatch.setattr(settings, "CLAIM_MODE", False)
    assert not is_claim_text("Tokyo is the capital of Japan.")


# ── Compound claims ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("claim,expected", [
    ("The Gateway of India in Mumbai was completed in 1924 and overlooks the Arabian Sea.",
     ["The Gateway of India in Mumbai was completed in 1924.",
      "The Gateway of India in Mumbai overlooks the Arabian Sea."]),
    ("Chennai is the capital of Tamil Nadu and is located on the Coromandel Coast.",
     ["Chennai is the capital of Tamil Nadu.", "Chennai is located on the Coromandel Coast."]),
])
def test_compound_claims_are_split_into_their_parts(claim, expected):
    assert split_compound(claim) == expected


@pytest.mark.parametrize("claim", [
    "Mount Everest lies on the border between Nepal and China.",        # 'and' joins nouns
    "The statue that stands in Gujarat is tall and is old.",            # clause in the subject
    "Rain fell and was heavy and was cold across the region.",          # three parts
    "Kolkata, the capital of West Bengal, is home to the Howrah Bridge.",
    "",
])
def test_ambiguous_sentences_are_left_alone(claim):
    assert split_compound(claim) == [claim]


def test_claim_node_checks_a_typed_claim_even_if_it_is_not_check_worthy(claim_mode):
    out = extract_claims_node({"article_text": "Tokyo is the capital of Japan."})
    assert out["claims"] == ["Tokyo is the capital of Japan."]
    assert not any(d["kind"] in ("text_too_short", "no_checkable_claims")
                   for d in out.get("degradations", []))


def test_article_extraction_is_unchanged_when_claim_mode_is_off():
    out = extract_claims_node({"article_text": "Tokyo is the capital of Japan."})
    assert any(d["kind"] == "text_too_short" for d in out.get("degradations", []))


# ── The evidence-only verdict ─────────────────────────────────────────────────

def _cv(verdict, ent=0.0, con=0.0, relevance=0.7, support=None, status="ok"):
    return {"claim": "c", "verdict": verdict, "confidence": max(ent, con),
            "evidence_snippets": ["e"], "entail_mass": ent, "contradict_mass": con,
            "support": (ent - con) if support is None else support,
            "top_relevance": relevance, "numeric_conflict": 0.0, "sources": ["x"], "status": status}


def verdict_of(cvs, ensemble=0.5, **kw):
    return ADJ.compute_claim_verdict(ensemble, cvs, {}, **kw)


def test_a_supported_claim_is_likely_real():
    r = verdict_of([_cv("entailment", ent=0.8)])
    assert r["verdict"] == "likely_real" and r["fusion_method"] == "evidence_rule"


def test_a_refuted_claim_is_likely_fake():
    r = verdict_of([_cv("contradiction", con=0.7)])
    assert r["verdict"] == "likely_fake"


def test_inconclusive_evidence_is_uncertain_never_fake():
    assert verdict_of([_cv("neutral")])["verdict"] == "uncertain"
    assert verdict_of([_cv("unverifiable", status="no_hits")])["verdict"] == "uncertain"
    assert verdict_of([])["verdict"] == "uncertain"


def test_the_style_classifier_is_ignored():
    """The point of claim mode: 0.99 'real' from the article model must not matter."""
    assert verdict_of([_cv("neutral")], ensemble=0.99)["verdict"] == "uncertain"
    assert verdict_of([_cv("contradiction", con=0.7)], ensemble=0.99)["verdict"] == "likely_fake"
    assert verdict_of([_cv("entailment", ent=0.8)], ensemble=0.01)["verdict"] == "likely_real"


def test_weak_signals_do_not_commit():
    assert verdict_of([_cv("entailment", ent=0.3)])["verdict"] == "uncertain"
    assert verdict_of([_cv("contradiction", con=0.4)])["verdict"] == "uncertain"


def test_mixed_evidence_is_uncertain():
    r = verdict_of([_cv("entailment", ent=0.8), _cv("contradiction", con=0.7)])
    assert r["verdict"] == "uncertain" and r["forced_reason"]


def test_most_parts_must_be_supported_and_none_refuted():
    assert verdict_of([_cv("entailment", ent=0.8), _cv("neutral")])["verdict"] == "likely_real"
    assert verdict_of([_cv("entailment", ent=0.8), _cv("neutral"), _cv("neutral")])["verdict"] == "uncertain"
    assert verdict_of([_cv("entailment", ent=0.8), _cv("contradiction", con=0.7)])["verdict"] == "uncertain"


def test_a_refuted_part_with_no_support_makes_the_claim_fake():
    assert verdict_of([_cv("contradiction", con=0.7), _cv("neutral")])["verdict"] == "likely_fake"


def test_degraded_nli_never_commits():
    r = verdict_of([_cv("entailment", ent=0.9)], nli_degraded=True)
    assert r["verdict"] == "uncertain"


def test_off_topic_evidence_is_not_decisive():
    assert verdict_of([_cv("contradiction", con=0.9, relevance=0.2)])["verdict"] == "uncertain"


def test_thresholds_come_from_settings(monkeypatch):
    from config import settings
    monkeypatch.setattr(settings, "CLAIM_ENTAIL_MIN", 0.9)
    assert verdict_of([_cv("entailment", ent=0.8)])["verdict"] == "uncertain"


def test_node_chooses_the_claim_path_by_text_length(claim_mode):
    state = {"article_text": "Tokyo is the capital of Japan.", "ensemble_score": 0.99,
             "claim_verdicts": [_cv("neutral")], "retrieval_stats": {}, "nli_degraded": False,
             "degradations": []}
    out = ADJ.adjudicate_node(state)
    assert out["fusion_method"] == "evidence_rule" and out["final_verdict"] == "uncertain"
    assert "short claim" in out["reasoning_trace"]


# ── Retrieval ─────────────────────────────────────────────────────────────────

def test_named_things_are_found_in_order():
    ents = ER._subject_entities("The Gateway of India in Mumbai overlooks the Arabian Sea.")
    assert ents == ["Gateway of India", "Mumbai", "Arabian Sea"]


def test_claim_mode_looks_up_each_subject_separately():
    claim = "Chennai is the capital of Kerala and lies on the Ganges."
    plain = ER.build_queries(claim)
    queries = ER.build_queries(claim, claim_mode=True)
    assert set(plain) <= set(queries)
    assert {("wikipedia", "Chennai"), ("wikipedia", "Kerala"), ("wikipedia", "Ganges")} <= set(queries)


def test_article_queries_are_unchanged():
    claim = "India's central bank raised its benchmark repo rate by 25 basis points on Thursday."
    assert ER.build_queries(claim) == ER.build_queries(claim, claim_mode=False)


def _page(title, extra=""):
    return {"text": f"{title}: {extra or 'A page.'}", "provider": "wikipedia", "trust": 0.85,
            "domain": "en.wikipedia.org", "url": f"https://en.wikipedia.org/wiki/{title}"}


def test_a_page_titled_in_the_claim_is_anchored():
    claim = "The Gateway of India in Delhi was completed in 1924."
    assert ER._title_anchored(_page("Gateway of India"), claim)
    assert not ER._title_anchored(_page("Taj Mahal"), claim)


def test_a_page_about_something_else_with_the_same_name_is_not_anchored():
    claim = "The Gateway of India in Mumbai was completed in 1924."
    assert not ER._title_anchored(_page("Gateway of India (film)"), claim)
    assert not ER._title_anchored(_page("Gateway of India (disambiguation)"), claim)
    assert ER._title_anchored(_page("Gateway of India (film)"), claim + " The film is not it, film aside.") is True


def test_non_wikipedia_results_are_never_anchored():
    assert not ER._title_anchored({"text": "Gateway of India: x", "provider": "gnews_rss"},
                                  "The Gateway of India is old.")


def test_subject_rank_follows_order_of_mention_and_containment():
    claim = "The River Nile flows through Egypt and Sudan."
    assert ER._subject_rank(_page("Nile"), claim) == 0          # "Nile" is the page for "River Nile"
    assert ER._subject_rank(_page("Egypt"), claim) == 1
    assert ER._subject_rank(_page("Blue Nile"), claim) == 99


def test_disambiguation_pages_are_recognised():
    assert ER._is_disambiguation({"text": "Mughal: Mughal or Moghul may refer to:"})
    assert ER._is_disambiguation({"text": "Delhi (disambiguation): Delhi may be"})
    assert not ER._is_disambiguation({"text": "Delhi: Delhi is a city in India."})


@pytest.fixture
def fake_cosine(monkeypatch):
    """Embedding stand-in: a fixed cosine per candidate, read from its 'cos' field."""
    import paper_trail_core.nlp.embeddings as emb
    monkeypatch.setattr(emb, "cosine_to", lambda claim, texts: [float(t.rsplit("||", 1)[1]) for t in texts])


def _cand(cos, title, **kw):
    c = _page(title)
    c["text"] = f"{title}: text ||{cos}"
    c.update(kw)
    return c


def test_claim_mode_admits_a_titled_page_below_the_usual_gate(fake_cosine):
    claim = "The Gateway of India in Delhi was completed in 1924."
    pages = [_cand(0.35, "Gateway of India"), _cand(0.35, "Taj Mahal")]
    kept = ER.gate_by_relevance(claim, pages, claim_mode=True)
    assert [k["text"].split(":")[0] for k in kept] == ["Gateway of India"]
    assert kept[0]["anchored"] and kept[0]["relevance"] >= ER.ANCHOR_RELEVANCE
    assert kept[0]["subject_rank"] == 0 and kept[0]["claim_mode"]


def test_the_usual_gate_is_unchanged_outside_claim_mode(fake_cosine):
    claim = "The Gateway of India in Delhi was completed in 1924."
    assert ER.gate_by_relevance(claim, [_cand(0.35, "Gateway of India")]) == []


def test_disambiguation_pages_are_dropped_in_claim_mode(fake_cosine):
    claim = "The Mughal emperors lived in Delhi."
    page = _cand(0.8, "Mughal")
    page["text"] = "Mughal: Mughal or Moghul may refer to: ||0.8"
    assert ER.gate_by_relevance(claim, [page], claim_mode=True) == []


# ── Who may refute ────────────────────────────────────────────────────────────

def test_only_pages_about_the_subject_or_a_paraphrase_may_refute():
    subject = {"claim_mode": True, "anchored": True, "subject_rank": 0, "relevance": 0.7}
    place = {"claim_mode": True, "anchored": True, "subject_rank": 1, "relevance": 0.7}
    passing = {"claim_mode": True, "anchored": True, "subject_rank": 2, "relevance": 0.7}
    neighbour = {"claim_mode": True, "relevance": 0.67}
    paraphrase = {"claim_mode": True, "relevance": 0.85}
    assert V._may_refute(subject) and V._may_refute(place) and V._may_refute(paraphrase)
    assert not V._may_refute(passing) and not V._may_refute(neighbour)


def _refuting_nli():
    return [NLIResult(entail=0.0, neutral=0.0, contradict=1.0)]


def test_a_neighbouring_page_cannot_refute_a_claim_in_claim_mode():
    snip = {"text": "Blue Nile: x", "relevance": 0.67, "trust": 0.85, "claim_mode": True}
    cv = V.aggregate_claim("The Nile flows through Sudan.", [snip], _refuting_nli())
    assert cv["contradict_mass"] == 0.0 and cv["verdict"] != "contradiction"


def test_the_same_page_can_refute_on_the_article_path():
    snip = {"text": "Blue Nile: x", "relevance": 0.67, "trust": 0.85}      # no claim_mode flag
    cv = V.aggregate_claim("The Nile flows through Sudan.", [snip], _refuting_nli())
    assert cv["contradict_mass"] > 0.3


def test_a_neighbouring_page_can_still_support():
    snip = {"text": "Blue Nile: x", "relevance": 0.7, "trust": 0.85, "claim_mode": True}
    cv = V.aggregate_claim("c", [snip], [NLIResult(entail=0.9, neutral=0.1, contradict=0.0)])
    assert cv["entail_mass"] > 0.4


# ── Language detection on short claims ────────────────────────────────────────

def test_short_english_is_english():
    assert detect_language("Water boils at 100 degrees Celsius at sea level.")[0] == "en"
    assert detect_language("Tokyo is the capital of Japan.")[0] == "en"


def test_non_latin_and_accented_text_is_not_forced_to_english():
    assert detect_language("भारत की राजधानी नई दिल्ली है और यह यमुना नदी के किनारे बसी है।")[0] == "hi"
    assert detect_language("Le gouvernement a annoncé de nouvelles mesures pour le pays.")[0] == "fr"


# ── End to end ────────────────────────────────────────────────────────────────

def test_a_very_short_claim_is_accepted_in_claim_mode(claim_mode, monkeypatch):
    def _retrieve(claim, idx, **kw):
        return {"claim": claim, "status": "ok", "cache_hits": 0, "queries": 1, "snippets": [
            {"text": "Japan: Tokyo is the capital of Japan.", "url": "u", "domain": "en.wikipedia.org",
             "trust": 0.85, "relevance": 0.8, "provider": "test", "claim_mode": True, "anchored": True,
             "subject_rank": 0}]}
    monkeypatch.setattr(ER, "retrieve_for_claim", _retrieve)
    monkeypatch.setattr(V, "is_available", lambda: True)
    monkeypatch.setattr(V, "infer_batch",
                        lambda premises, hyps: [NLIResult(entail=0.95, neutral=0.05, contradict=0.0)
                                                for _ in premises])
    vr = P.run_verification_pipeline("Tokyo is the capital of Japan.")
    assert vr.verdict == "likely_real" and vr.fusion_method == "evidence_rule"


def test_the_same_short_claim_is_rejected_outside_claim_mode():
    vr = P.run_verification_pipeline("Tokyo is the capital of Japan.")
    assert vr.verdict == "uncertain" and any(d["kind"] == "text_too_short" for d in vr.degradations)


def test_time_bound_claims_cannot_be_refuted_by_other_days_news():
    from paper_trail_core.agents.verifier import _is_news_source, _is_time_bound

    assert _is_time_bound(
        "Delhi Metro to shut down 57 stations from 9pm on Friday till Saturday"
    )
    assert _is_time_bound("CJP Protest Live Updates: metro shut ahead of Oct 10 protest")
    assert not _is_time_bound("The Gateway of India was completed in 1924.")
    assert _is_news_source({"domain": "thehindu.com", "in_region": True})
    assert not _is_news_source({"domain": "thehindu.com"})
    assert not _is_news_source({"domain": "en.wikipedia.org"})
