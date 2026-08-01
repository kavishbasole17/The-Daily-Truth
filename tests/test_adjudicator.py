"""
Unit tests for the adjudicator's scoring logic.

These tests are pure-function tests — no LangGraph, no DB, no model loading.
Run with: pytest tests/test_adjudicator.py -v
"""
import pytest
from paper_trail_core.agents.adjudicator import compute_verdict
from paper_trail_core.agents.state import ClaimVerdict


# ── Helpers ────────────────────────────────────────────────────────────────────

def _cv(verdict: str, confidence: float = 0.8) -> ClaimVerdict:
    return ClaimVerdict(
        claim="Test claim",
        verdict=verdict,
        confidence=confidence,
        evidence_snippets=["snippet"],
    )


# ── Basic verdict thresholds ───────────────────────────────────────────────────

def test_high_ensemble_score_is_real():
    result = compute_verdict(ensemble_score=0.85, claim_verdicts=[])
    assert result["verdict"] == "likely_real"
    assert result["confidence"] >= 0.6


def test_low_ensemble_score_is_fake():
    result = compute_verdict(ensemble_score=0.15, claim_verdicts=[])
    assert result["verdict"] == "likely_fake"
    assert result["confidence"] >= 0.6


def test_mid_ensemble_score_is_uncertain():
    result = compute_verdict(ensemble_score=0.5, claim_verdicts=[])
    assert result["verdict"] == "uncertain"
    assert result["needs_human_review"] is True


# ── NLI adjustments ────────────────────────────────────────────────────────────

def test_contradictions_push_toward_fake():
    """Multiple contradictions should drive a borderline score below the fake threshold."""
    claims = [_cv("contradiction", 0.9)] * 5
    result = compute_verdict(ensemble_score=0.55, claim_verdicts=claims)
    # 5 × 0.10 × 0.9 = 0.45 adjustment → 0.55 - 0.45 = 0.10 → likely_fake
    assert result["verdict"] == "likely_fake"


def test_entailments_push_toward_real():
    """Strong entailments should lift a borderline score above the real threshold."""
    claims = [_cv("entailment", 0.9)] * 5
    result = compute_verdict(ensemble_score=0.45, claim_verdicts=claims)
    # 5 × 0.05 × 0.9 = 0.225 → 0.45 + 0.225 = 0.675 → likely_real
    assert result["verdict"] == "likely_real"


def test_neutral_claims_no_change():
    """Neutral NLI verdicts must not change the final score."""
    claims = [_cv("neutral", 1.0)] * 10
    result_with    = compute_verdict(ensemble_score=0.5, claim_verdicts=claims)
    result_without = compute_verdict(ensemble_score=0.5, claim_verdicts=[])
    assert result_with["final_score"] == result_without["final_score"]


def test_mixed_claims():
    """Mixed signals produce a blended score."""
    claims = [
        _cv("entailment",   0.9),
        _cv("contradiction", 0.9),
        _cv("neutral",      1.0),
    ]
    result = compute_verdict(ensemble_score=0.5, claim_verdicts=claims)
    # net adjustment = +0.05×0.9 - 0.10×0.9 = -0.045 → 0.455 → uncertain
    assert result["verdict"] == "uncertain"
    assert result["needs_human_review"] is True


# ── Score clamping ─────────────────────────────────────────────────────────────

def test_score_never_exceeds_1():
    claims = [_cv("entailment", 1.0)] * 50
    result = compute_verdict(ensemble_score=1.0, claim_verdicts=claims)
    assert result["final_score"] <= 1.0


def test_score_never_below_0():
    claims = [_cv("contradiction", 1.0)] * 50
    result = compute_verdict(ensemble_score=0.0, claim_verdicts=claims)
    assert result["final_score"] >= 0.0


# ── Human review flag ──────────────────────────────────────────────────────────

def test_needs_human_review_only_when_uncertain():
    real_result = compute_verdict(ensemble_score=0.9, claim_verdicts=[])
    fake_result = compute_verdict(ensemble_score=0.1, claim_verdicts=[])
    unc_result  = compute_verdict(ensemble_score=0.5, claim_verdicts=[])

    assert real_result["needs_human_review"] is False
    assert fake_result["needs_human_review"] is False
    assert unc_result["needs_human_review"]  is True


# ── Custom thresholds ──────────────────────────────────────────────────────────

def test_custom_thresholds():
    # Widen the uncertainty band
    result = compute_verdict(
        ensemble_score=0.65,
        claim_verdicts=[],
        escalation_low=0.3,
        escalation_high=0.8,
    )
    assert result["verdict"] == "uncertain"

    # Score that would normally be "likely_real" (0.65 > 0.6) is now uncertain
    result2 = compute_verdict(
        ensemble_score=0.65,
        claim_verdicts=[],
        escalation_low=0.3,
        escalation_high=0.8,
    )
    assert result2["needs_human_review"] is True
