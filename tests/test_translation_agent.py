"""
Language-ID + translation agent tests.

Runs offline: `_llm_translate` is monkeypatched at the module boundary
(paper_trail_core.agents.translation_agent._llm_translate), so no Anthropic
API key or network call is required — same pattern as test_ocr_agent.py.

English input must never call the LLM at all (it's a no-op pass-through);
several tests assert this directly by making the mock raise if invoked,
since a silent unnecessary LLM call would be a cost/latency regression that
a "did it return the right text" assertion alone wouldn't catch.
"""
from __future__ import annotations

import pytest

from paper_trail_core.agents import pipeline as P
from paper_trail_core.agents import translation_agent as TA
from paper_trail_core.nlp.lang_id import detect_language

HINDI_ARTICLE = (
    "भारतीय रिज़र्व बैंक ने गुरुवार को अपनी बेंचमार्क रेपो दर में 25 आधार अंकों "
    "की बढ़ोतरी करते हुए इसे 6.75 प्रतिशत कर दिया, जिसका कारण लगातार खाद्य "
    "मुद्रास्फीति को बताया गया।"
)
ENGLISH_ARTICLE = (
    "The Reserve Bank of India raised its benchmark repo rate by 25 basis "
    "points to 6.75 percent on Thursday, citing persistent food inflation."
)
FRENCH_ARTICLE = (
    "La Banque de Réserve de l'Inde a relevé son taux de repo de référence "
    "de 25 points de base à 6,75 %, invoquant une inflation alimentaire "
    "persistante qui pèse sur l'économie nationale depuis plusieurs mois."
)


def _raise_if_called(*_a, **_k):
    raise AssertionError("LLM translation should not have been called")


# ── detect_language ──────────────────────────────────────────────────────────

def test_detect_language_english():
    lang, conf = detect_language(ENGLISH_ARTICLE)
    assert lang == "en"
    assert conf > 0.5


def test_detect_language_hindi():
    lang, conf = detect_language(HINDI_ARTICLE)
    assert lang == "hi"
    assert conf > 0.5


def test_detect_language_empty():
    assert detect_language("") == ("unknown", 0.0)


# ── translate_node ───────────────────────────────────────────────────────────

def test_english_passes_through_without_llm_call(monkeypatch):
    monkeypatch.setattr(TA, "_llm_translate", _raise_if_called)
    out = TA.translate_node({"article_text": ENGLISH_ARTICLE})
    assert out["detected_language"] == "en"
    assert out["article_text"] == ENGLISH_ARTICLE
    assert out["translated_text"] == ENGLISH_ARTICLE
    assert out["original_text"] == ENGLISH_ARTICLE
    assert out["degradations"] == []


def test_hindi_is_translated(monkeypatch):
    monkeypatch.setattr(TA, "_llm_translate", lambda text: "The RBI raised rates.")
    out = TA.translate_node({"article_text": HINDI_ARTICLE})
    assert out["detected_language"] == "hi"
    assert out["original_text"] == HINDI_ARTICLE
    assert out["translated_text"] == "The RBI raised rates."
    assert out["article_text"] == "The RBI raised rates."
    assert out["degradations"] == []


def test_hindi_translation_failure_abstains(monkeypatch):
    def _boom(text):
        raise RuntimeError("provider timeout")
    monkeypatch.setattr(TA, "_llm_translate", _boom)
    out = TA.translate_node({"article_text": HINDI_ARTICLE})
    assert out["article_text"] == ""
    assert out["original_text"] == HINDI_ARTICLE  # preserved even on failure
    assert out["degradations"][0]["kind"] == "llm_failed"
    assert out["degradations"][0]["severity"] == "critical"


def test_unsupported_language_abstains(monkeypatch):
    monkeypatch.setattr(TA, "_llm_translate", _raise_if_called)
    out = TA.translate_node({"article_text": FRENCH_ARTICLE})
    assert out["article_text"] == ""
    assert out["degradations"][0]["kind"] == "unsupported_language"
    assert out["degradations"][0]["severity"] == "critical"


def test_backend_off_abstains_hindi(monkeypatch):
    monkeypatch.setattr(TA, "_BACKEND", "off")
    out = TA.translate_node({"article_text": HINDI_ARTICLE})
    assert out["article_text"] == ""
    assert out["degradations"][0]["kind"] == "backend_off"


def test_backend_indictrans2_not_implemented(monkeypatch):
    monkeypatch.setattr(TA, "_BACKEND", "indictrans2")
    out = TA.translate_node({"article_text": HINDI_ARTICLE})
    assert out["article_text"] == ""
    assert out["degradations"][0]["kind"] == "backend_not_implemented"


def test_empty_text_is_defensive_not_a_crash():
    out = TA.translate_node({"article_text": ""})
    assert out["article_text"] == ""
    assert out["degradations"][0]["kind"] == "no_text"


# ── Pipeline-level ────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_fusion():
    from paper_trail_core.agents import fusion
    fusion._payload, fusion._loaded = None, True
    yield
    fusion._loaded = False


def test_unsupported_language_never_reaches_fusion(monkeypatch):
    monkeypatch.setattr(TA, "_llm_translate", _raise_if_called)
    result = P.run_verification_pipeline(FRENCH_ARTICLE)
    assert result.verdict == "uncertain"
    assert result.fusion_method == "not_run"
    assert any(d["stage"] == "translation" for d in result.degradations)


def test_translated_hindi_reaches_claim_extraction(monkeypatch):
    """A successful translation should proceed into the normal pipeline, not the abstain shortcut."""
    monkeypatch.setattr(TA, "_llm_translate", lambda text: ENGLISH_ARTICLE)

    from paper_trail_core.agents import evidence_retriever as ER

    def _no_network(*_a, **_k):
        raise AssertionError("test attempted a real network call")
    monkeypatch.setattr(ER, "_search_newsapi", _no_network)
    monkeypatch.setattr(ER, "_search_wikipedia", _no_network)

    def _retrieve(claim, idx):
        return {"claim": claim, "snippets": [], "status": "no_hits", "cache_hits": 0, "queries": 1}
    monkeypatch.setattr(ER, "retrieve_for_claim", _retrieve)

    result = P.run_verification_pipeline(HINDI_ARTICLE)
    assert result.fusion_method != "not_run"
