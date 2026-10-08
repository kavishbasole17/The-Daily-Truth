"""
End-to-end pipeline tests.

Retrieval and NLI are stubbed, so these run offline, on CPU, in about a second.
What they assert is the *contract* — that whatever happens inside, the pipeline
returns an object the API can persist and the frontend can render, and that it
never returns a confident verdict it cannot justify.

`frontend/src/app/verify/page.jsx` dereferences `verdict.claim_verdicts.length`
and `cv.evidence_snippets.length` without null checks, so a null in either
position is a blank page for the user rather than a caught error. That contract
is asserted explicitly here.
"""
from __future__ import annotations

import pytest

from paper_trail_core.agents import evidence_retriever as ER
from paper_trail_core.agents import pipeline as P
from paper_trail_core.agents import verifier as V
from paper_trail_core.nlp.nli import NLIResult

REAL_ARTICLE = (
    "The World Health Organization declared the COVID-19 outbreak a pandemic on "
    "11 March 2020. Director-General Tedros Adhanom Ghebreyesus said the agency "
    "was deeply concerned by the alarming levels of spread. At the time of the "
    "announcement more than 118,000 cases had been reported in 114 countries."
)

FAKE_ARTICLE = (
    "The Federal Reserve announced Wednesday it will begin issuing a mandatory "
    "digital dollar in March, replacing all physical currency and requiring "
    "citizens to link bank accounts to a national biometric identity registry. "
    "Officials said cash transactions would be phased out within eighteen months."
)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Every provider call fails loudly if a test forgets to stub retrieval."""
    def _boom(*_a, **_k):
        raise AssertionError("test attempted a real network call")
    monkeypatch.setattr(ER, "_search_newsapi", _boom)
    monkeypatch.setattr(ER, "_search_wikipedia", _boom)


@pytest.fixture(autouse=True)
def _reset_fusion():
    """Pin the deterministic fallback so results don't depend on a trained model."""
    from paper_trail_core.agents import fusion
    fusion._payload, fusion._loaded = None, True
    yield
    fusion._loaded = False


def stub_evidence(monkeypatch, text: str, relevance=0.75, trust=0.9):
    def _retrieve(claim, idx):
        return {
            "claim": claim,
            "snippets": [{"text": text, "url": "https://reuters.com/x",
                          "domain": "reuters.com", "trust": trust,
                          "relevance": relevance, "provider": "test"}],
            "status": "ok", "cache_hits": 1, "queries": 1,
        }
    monkeypatch.setattr(ER, "retrieve_for_claim", _retrieve)


def stub_nli(monkeypatch, entail=0.9, contradict=0.05):
    monkeypatch.setattr(V, "is_available", lambda: True)
    monkeypatch.setattr(
        V, "infer_batch",
        lambda premises, hypotheses: [
            NLIResult(entail=entail, neutral=max(0.0, 1 - entail - contradict),
                      contradict=contradict)
            for _ in premises
        ],
    )


# ── The frontend contract ─────────────────────────────────────────────────────

def assert_contract(vr):
    assert vr.verdict in ("likely_real", "likely_fake", "uncertain")
    assert isinstance(vr.confidence, float) and 0.0 <= vr.confidence <= 1.0
    assert isinstance(vr.claim_verdicts, list)
    for cv in vr.claim_verdicts:
        assert isinstance(cv.get("claim"), str)
        assert cv.get("verdict") in ("entailment", "contradiction", "neutral", "unverifiable")
        assert isinstance(cv.get("confidence"), float)
        assert isinstance(cv.get("evidence_snippets"), list)
        assert all(isinstance(s, str) for s in cv["evidence_snippets"])
    assert isinstance(vr.reasoning_trace, str) and vr.reasoning_trace
    assert isinstance(vr.degradations, list)


def test_contract_holds_for_supported_article(monkeypatch):
    stub_evidence(monkeypatch, "The WHO declared COVID-19 a pandemic on 11 March 2020.")
    stub_nli(monkeypatch, entail=0.95, contradict=0.02)
    assert_contract(P.run_verification_pipeline(REAL_ARTICLE))


def test_contract_holds_for_refuted_article(monkeypatch):
    stub_evidence(monkeypatch, "The Federal Reserve has no plans to issue a digital dollar.")
    stub_nli(monkeypatch, entail=0.02, contradict=0.96)
    assert_contract(P.run_verification_pipeline(FAKE_ARTICLE))


def test_contract_holds_when_everything_fails(monkeypatch):
    monkeypatch.setattr(ER, "retrieve_for_claim",
                        lambda c, i: {"claim": c, "snippets": [], "status": "error",
                                      "cache_hits": 0, "queries": 0})
    monkeypatch.setattr(V, "is_available", lambda: False)
    assert_contract(P.run_verification_pipeline(REAL_ARTICLE))


# ── Directional behaviour ─────────────────────────────────────────────────────

def test_refuting_evidence_scores_lower_than_supporting(monkeypatch):
    stub_evidence(monkeypatch, "Corroborating report from a wire service.")
    stub_nli(monkeypatch, entail=0.95, contradict=0.02)
    supported = P.run_verification_pipeline(REAL_ARTICLE)

    stub_evidence(monkeypatch, "The claim is contradicted by the official record.")
    stub_nli(monkeypatch, entail=0.02, contradict=0.96)
    refuted = P.run_verification_pipeline(REAL_ARTICLE)

    assert refuted.fused_score < supported.fused_score


def test_degraded_nli_never_commits(monkeypatch):
    """A verifier that cannot detect contradiction must not deliver a verdict."""
    stub_evidence(monkeypatch, "Strongly corroborating evidence from Reuters.")
    monkeypatch.setattr(V, "is_available", lambda: False)

    vr = P.run_verification_pipeline(REAL_ARTICLE)
    assert vr.verdict == "uncertain"
    assert any(d["kind"] == "nli_fallback_lexical" for d in vr.degradations)


def test_no_evidence_does_not_produce_likely_fake(monkeypatch):
    """Absence of coverage is not evidence of fabrication."""
    monkeypatch.setattr(ER, "retrieve_for_claim",
                        lambda c, i: {"claim": c, "snippets": [], "status": "no_hits",
                                      "cache_hits": 0, "queries": 2})
    stub_nli(monkeypatch)
    vr = P.run_verification_pipeline(REAL_ARTICLE)
    assert vr.verdict == "uncertain"


# ── Robustness ────────────────────────────────────────────────────────────────

def test_short_input_is_rejected_without_a_verdict():
    vr = P.run_verification_pipeline("Too short.")
    assert vr.verdict == "uncertain"
    assert vr.degradations


def test_stage_failure_is_contained_and_reported(monkeypatch):
    def _explode(_state):
        raise RuntimeError("simulated retrieval outage")
    monkeypatch.setattr(P, "retrieve_evidence_node", _explode)
    monkeypatch.setattr(P, "_app", None)
    P._app = None

    vr = P.run_verification_pipeline(REAL_ARTICLE)
    try:
        assert_contract(vr)
        assert vr.verdict == "uncertain"
        assert any(d["severity"] == "critical" for d in vr.degradations)
    finally:
        P._app = None


def test_pipeline_is_deterministic(monkeypatch):
    stub_evidence(monkeypatch, "A corroborating wire report about the pandemic declaration.")
    stub_nli(monkeypatch, entail=0.9, contradict=0.03)

    a = P.run_verification_pipeline(REAL_ARTICLE)
    b = P.run_verification_pipeline(REAL_ARTICLE)
    assert (a.verdict, a.confidence, a.fused_score) == (b.verdict, b.confidence, b.fused_score)


def test_result_is_json_serialisable(monkeypatch):
    import json
    stub_evidence(monkeypatch, "Some evidence about the pandemic declaration.")
    stub_nli(monkeypatch)
    vr = P.run_verification_pipeline(REAL_ARTICLE)

    json.loads(vr.claim_verdicts_json())
    json.loads(vr.degradations_json())
    json.dumps(vr.to_dict())
