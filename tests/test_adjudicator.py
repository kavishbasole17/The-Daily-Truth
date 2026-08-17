"""
Adjudicator tests.

Every test pins `fusion_payload=None` to force the deterministic heuristic
path, so results don't depend on whether a fusion model happens to be trained
on the machine running the suite.

(The previous version of this file described an additive ±0.05/±0.10-per-claim
scoring model that had already been replaced by a weighted blend, so its
comments explained arithmetic the code no longer did. Comments here describe
the behaviour being asserted, not an implementation that may move.)
"""
from __future__ import annotations

import pytest

from paper_trail_core.agents.adjudicator import compute_verdict
from paper_trail_core.agents.state import ClaimVerdict


def cv(verdict: str, support: float = 0.0, confidence: float = 0.8,
       status: str = "ok", relevance: float = 0.8) -> ClaimVerdict:
    return ClaimVerdict(
        claim="Test claim",
        verdict=verdict,
        confidence=confidence,
        evidence_snippets=["snippet"],
        support=support,
        entail_mass=max(0.0, support),
        contradict_mass=max(0.0, -support),
        top_relevance=relevance,
        numeric_conflict=0.0,
        sources=["reuters.com"],
        status=status,
    )


def verdict_of(**kwargs):
    kwargs.setdefault("fusion_payload", None)
    return compute_verdict(**kwargs)


# ── Thresholds with no evidence at all ────────────────────────────────────────

def test_no_claims_falls_back_to_ensemble():
    high = verdict_of(ensemble_score=0.85, claim_verdicts=[])
    low = verdict_of(ensemble_score=0.15, claim_verdicts=[])
    assert high["verdict"] == "likely_real"
    assert low["verdict"] == "likely_fake"


def test_mid_ensemble_is_uncertain():
    r = verdict_of(ensemble_score=0.5, claim_verdicts=[])
    assert r["verdict"] == "uncertain"
    assert r["needs_human_review"] is True


# ── Evidence moves the verdict ────────────────────────────────────────────────

def test_contradicting_evidence_overrides_a_confident_ensemble():
    """
    The core failure this rebuild addresses: the style classifier calls a
    well-written fabrication "real", and refuting evidence has to be able to
    outvote it.
    """
    claims = [cv("contradiction", support=-0.8) for _ in range(3)]
    r = verdict_of(
        ensemble_score=0.95,
        claim_verdicts=claims,
        retrieval_stats={"mean_relevance": 0.8, "mean_trust": 1.0},
    )
    assert r["verdict"] in ("likely_fake", "uncertain")
    assert r["final_score"] < 0.95


def test_supporting_evidence_raises_the_score():
    claims = [cv("entailment", support=0.8) for _ in range(3)]
    with_ev = verdict_of(ensemble_score=0.5, claim_verdicts=claims)
    without = verdict_of(ensemble_score=0.5, claim_verdicts=[])
    assert with_ev["final_score"] > without["final_score"]


def test_neutral_evidence_does_not_move_the_score_much():
    claims = [cv("neutral", support=0.0) for _ in range(4)]
    r = verdict_of(ensemble_score=0.5, claim_verdicts=claims)
    assert r["final_score"] == pytest.approx(0.5, abs=0.05)


# ── Absence of evidence must not imply "fake" ─────────────────────────────────

def test_unverifiable_claims_do_not_push_toward_fake():
    """
    Regression for the UNCORROBORATED-as-contradiction defect. An article
    nobody wrote about must not be scored lower than one we never checked.
    """
    unverifiable = [cv("unverifiable", support=0.0, status="no_hits") for _ in range(4)]
    with_none = verdict_of(ensemble_score=0.7, claim_verdicts=unverifiable)
    baseline = verdict_of(ensemble_score=0.7, claim_verdicts=[])
    assert with_none["final_score"] >= baseline["final_score"] - 0.01
    assert with_none["verdict"] != "likely_fake"


def test_zero_coverage_forces_abstention():
    unverifiable = [cv("unverifiable", status="no_hits") for _ in range(3)]
    r = verdict_of(ensemble_score=0.95, claim_verdicts=unverifiable)
    assert r["verdict"] == "uncertain"
    assert r["needs_human_review"] is True
    assert r["forced_reason"] is not None


# ── Degraded NLI is hard-gated ────────────────────────────────────────────────

def test_degraded_nli_forces_abstention():
    """
    A verifier that cannot detect contradiction does not merely add noise — it
    adds confidently wrong signal. It must never produce a committed verdict.
    """
    claims = [cv("entailment", support=0.9) for _ in range(4)]
    r = verdict_of(ensemble_score=0.95, claim_verdicts=claims, nli_degraded=True)
    assert r["verdict"] == "uncertain"
    assert r["needs_human_review"] is True
    assert "degraded" in r["forced_reason"]


def test_degraded_nli_zeroes_the_evidence_channel():
    claims = [cv("contradiction", support=-0.9) for _ in range(4)]
    clean = verdict_of(ensemble_score=0.8, claim_verdicts=claims, nli_degraded=False)
    degraded = verdict_of(ensemble_score=0.8, claim_verdicts=claims, nli_degraded=True)
    assert degraded["final_score"] > clean["final_score"]
    assert degraded["final_score"] == pytest.approx(0.8, abs=0.01)


# ── Confidence semantics ──────────────────────────────────────────────────────

def test_confidence_is_the_probability_of_the_reported_class():
    real = verdict_of(ensemble_score=0.9, claim_verdicts=[])
    fake = verdict_of(ensemble_score=0.1, claim_verdicts=[])
    assert real["confidence"] == pytest.approx(real["final_score"], abs=0.01)
    assert fake["confidence"] == pytest.approx(1 - fake["final_score"], abs=0.01)


def test_uncertain_confidence_is_not_pinned_near_zero():
    """
    The old formula capped uncertain confidence at 0.1, so the UI showed
    "10% Confidence" on every single uncertain verdict regardless of the score.
    """
    r = verdict_of(ensemble_score=0.55, claim_verdicts=[])
    assert r["verdict"] == "uncertain"
    assert 0.5 <= r["confidence"] <= 1.0


def test_confidence_is_always_a_valid_probability():
    for score in (0.0, 0.25, 0.5, 0.75, 1.0):
        r = verdict_of(ensemble_score=score, claim_verdicts=[])
        assert 0.0 <= r["confidence"] <= 1.0
        assert 0.0 <= r["final_score"] <= 1.0


# ── Bounds and thresholds ─────────────────────────────────────────────────────

def test_score_stays_in_range_under_extreme_input():
    for support, ensemble in ((1.0, 1.0), (-1.0, 0.0)):
        claims = [cv("entailment" if support > 0 else "contradiction", support=support)] * 50
        r = verdict_of(ensemble_score=ensemble, claim_verdicts=claims)
        assert 0.0 <= r["final_score"] <= 1.0


def test_custom_thresholds_widen_the_abstention_band():
    r = verdict_of(ensemble_score=0.65, claim_verdicts=[],
                   escalation_low=0.3, escalation_high=0.8)
    assert r["verdict"] == "uncertain"
    assert r["needs_human_review"] is True


def test_needs_human_review_iff_uncertain():
    assert verdict_of(ensemble_score=0.95, claim_verdicts=[])["needs_human_review"] is False
    assert verdict_of(ensemble_score=0.05, claim_verdicts=[])["needs_human_review"] is False
    assert verdict_of(ensemble_score=0.50, claim_verdicts=[])["needs_human_review"] is True


def test_features_are_always_complete():
    from paper_trail_core.agents.features import FEATURE_NAMES
    r = verdict_of(ensemble_score=0.5, claim_verdicts=[cv("entailment", 0.5)])
    assert set(r["features"]) == set(FEATURE_NAMES)
