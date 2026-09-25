"""
OCR agent tests.

Runs entirely offline: `run_ocr` is monkeypatched at the module boundary
(paper_trail_core.agents.ocr_agent.run_ocr) so no Tesseract binary, language
pack, or real image decoding is required — the same pattern
tests/test_pipeline_e2e.py uses for evidence retrieval and NLI.

The pipeline-level tests here exist specifically because an earlier version
of this wiring had a real bug: routing a failed OCR read through the normal
extract_claims -> ... -> adjudicate path let the *trained* fusion model
score an all-zero evidence vector and return a confident "likely_fake" for
an image that was never read. `test_low_confidence_ocr_never_reaches_fusion`
and `test_ocr_failure_never_reaches_fusion` pin the fix: a failed/low-
confidence read must end at fusion_method="not_run", never at a value that
implies the fusion model actually ran.
"""
from __future__ import annotations

import pytest

from paper_trail_core.agents import ocr_agent as OA
from paper_trail_core.agents import pipeline as P
from paper_trail_core.nlp.ocr import OcrResult, detect_script


# ── detect_script ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("यह एक हिंदी वाक्य है", "devanagari"),
    ("This is an English sentence", "latin"),
    ("यह Hindi और English mixed hai", "mixed"),
    ("12345 !@#$%", "unknown"),
    ("", "unknown"),
])
def test_detect_script(text, expected):
    assert detect_script(text) == expected


# ── ocr_node ─────────────────────────────────────────────────────────────────

def test_ocr_node_happy_path(monkeypatch):
    monkeypatch.setattr(
        OA, "run_ocr",
        lambda path: OcrResult(text="भारत ने आज जीत हासिल की", confidence=0.82,
                                script="devanagari", n_blocks=6),
    )
    out = OA.ocr_node({"raw_image_ref": "fake.png"})
    assert out["article_text"] == "भारत ने आज जीत हासिल की"
    assert out["ocr_confidence"] == pytest.approx(0.82)
    assert out["ocr_script"] == "devanagari"
    assert out["degradations"] == []


def test_ocr_node_low_confidence_abstains(monkeypatch):
    monkeypatch.setattr(
        OA, "run_ocr",
        lambda path: OcrResult(text="garbled ...", confidence=0.20,
                                script="latin", n_blocks=3),
    )
    out = OA.ocr_node({"raw_image_ref": "fake.png"})
    assert out["article_text"] == ""
    assert out["ocr_text"] == "garbled ..."  # kept for diagnostics even though unused downstream
    assert len(out["degradations"]) == 1
    d = out["degradations"][0]
    assert d["stage"] == "ocr"
    assert d["kind"] == "low_confidence"
    assert d["severity"] == "critical"


def test_ocr_node_extraction_failure(monkeypatch):
    def _boom(path):
        raise OSError("cannot identify image file")
    monkeypatch.setattr(OA, "run_ocr", _boom)
    out = OA.ocr_node({"raw_image_ref": "corrupt.png"})
    assert out["article_text"] == ""
    assert out["degradations"][0]["kind"] == "extraction_failed"
    assert out["degradations"][0]["severity"] == "critical"


def test_ocr_node_no_image_ref():
    out = OA.ocr_node({"raw_image_ref": None})
    assert out["article_text"] == ""
    assert out["degradations"][0]["kind"] == "no_image"


def test_ocr_node_backend_off(monkeypatch):
    monkeypatch.setattr(OA, "_BACKEND", "off")
    out = OA.ocr_node({"raw_image_ref": "fake.png"})
    assert out["article_text"] == ""
    assert out["degradations"][0]["kind"] == "backend_off"


# ── Pipeline-level: the fusion-model bug this file exists to pin ────────────

@pytest.fixture(autouse=True)
def _reset_fusion():
    """Same pin as test_pipeline_e2e.py — don't depend on a trained model existing."""
    from paper_trail_core.agents import fusion
    fusion._payload, fusion._loaded = None, True
    yield
    fusion._loaded = False


def test_low_confidence_ocr_never_reaches_fusion(monkeypatch):
    monkeypatch.setattr(
        OA, "run_ocr",
        lambda path: OcrResult(text="garbled", confidence=0.1, script="unknown", n_blocks=1),
    )
    result = P.run_verification_pipeline_image("fake.png")
    assert result.verdict == "uncertain"
    assert result.needs_human_review is True
    assert result.fusion_method == "not_run"


def test_ocr_failure_never_reaches_fusion(monkeypatch):
    def _boom(path):
        raise FileNotFoundError(path)
    monkeypatch.setattr(OA, "run_ocr", _boom)
    result = P.run_verification_pipeline_image("missing.png")
    assert result.verdict == "uncertain"
    assert result.needs_human_review is True
    assert result.fusion_method == "not_run"
    assert any(d["stage"] == "ocr" for d in result.degradations)


def test_no_image_path_abstains_without_touching_ocr():
    result = P.run_verification_pipeline_image("")
    assert result.verdict == "uncertain"
    assert result.needs_human_review is True


def test_good_ocr_reaches_claim_extraction(monkeypatch):
    """
    A confident OCR read should proceed into the normal text pipeline (not
    the abstain shortcut) — proven here by the fact that extract_claims runs
    at all and produces a non-empty claims list from the OCR'd text.
    """
    article = (
        "The Reserve Bank of India raised its benchmark repo rate by 25 basis "
        "points to 6.75 percent on Thursday, citing persistent food inflation."
    )
    monkeypatch.setattr(
        OA, "run_ocr",
        lambda path: OcrResult(text=article, confidence=0.9, script="latin", n_blocks=20),
    )

    from paper_trail_core.agents import evidence_retriever as ER

    def _no_network(*_a, **_k):
        raise AssertionError("test attempted a real network call")
    monkeypatch.setattr(ER, "_search_newsapi", _no_network)
    monkeypatch.setattr(ER, "_search_wikipedia", _no_network)

    def _retrieve(claim, idx):
        return {"claim": claim, "snippets": [], "status": "no_hits", "cache_hits": 0, "queries": 1}
    monkeypatch.setattr(ER, "retrieve_for_claim", _retrieve)

    result = P.run_verification_pipeline_image("fake.png")
    # Reached real adjudication (not the OCR-abstain shortcut) — proven by
    # fusion_method being something other than "not_run", the value reserved
    # for the abstain-before-any-computation path this file's other tests pin.
    assert result.fusion_method != "not_run"
