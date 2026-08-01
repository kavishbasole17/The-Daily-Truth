"""
Adjudicator agent node — pure computation, no external API calls.

Combines two signals into a final verdict:
  1. ensemble_score  — classical/hybrid model's P(real)
  2. evidence_score  — derived from evidence-grounded claim verdicts
                        (entailment/contradiction against trusted sources)

Why evidence gets more weight than the ensemble
-------------------------------------------------
The classical ensemble is a fast text-pattern classifier trained on a mostly
static, dataset-specific corpus (see model benchmark notes). Empirically it's
unreliable on well-written fabricated content: it keys on writing *style*
(formal wire-service tone vs. tabloid tone) rather than truthfulness, so a
fabricated article written in a formal register can score confidently "real"
from the ensemble alone. Evidence-grounded claim verification — checking
each claim against real trusted sources (see evidence_retriever.py) — is a
substantially stronger signal when it has something informative to say.

So: when evidence retrieval found at least one claim it could meaningfully
corroborate or contradict, evidence dominates the final score. When nothing
informative came back (e.g. all claims neutral — evidence too vague, or claim
extraction found nothing to check), we fall back to trusting the ensemble.

Output schema:
  {
    "verdict":            "likely_fake" | "likely_real" | "uncertain",
    "confidence":         float (0-1),
    "ensemble_score":     float,
    "claim_verdicts":     [...],
    "reasoning_trace":    str,
    "needs_human_review": bool,
  }
"""
from __future__ import annotations

import logging
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from config import settings
    _LOW  = settings.ESCALATION_LOW
    _HIGH = settings.ESCALATION_HIGH
except ImportError:
    _LOW, _HIGH = 0.4, 0.6

from paper_trail_core.agents.state import ClaimVerdict, PipelineState

log = logging.getLogger(__name__)

# Weight given to the classical ensemble when evidence retrieval found at
# least one informative (non-neutral) claim verdict — evidence dominates.
_W_ENSEMBLE_WITH_EVIDENCE = 0.15
# Weight given to the ensemble when no informative evidence was found —
# it's the only signal available, so it drives the verdict.
_W_ENSEMBLE_NO_EVIDENCE = 0.8


def _evidence_score(claim_verdicts: list[ClaimVerdict]) -> float:
    """
    Maps claim verdicts to a [0,1] "P(real) according to evidence" score.
    0.5 is neutral/no-signal. Confident entailments pull toward 1, confident
    contradictions pull toward 0. Neutral verdicts contribute nothing but
    still dilute the average — more neutral claims means less decisive
    evidence overall, which is the correct behavior.
    """
    if not claim_verdicts:
        return 0.5
    total = 0.0
    for cv in claim_verdicts:
        v, c = cv["verdict"], cv["confidence"]
        if v == "entailment":
            total += c
        elif v == "contradiction":
            total -= c
        # neutral -> no contribution
    avg = total / len(claim_verdicts)  # roughly in [-1, 1]
    return max(0.0, min(1.0, 0.5 + avg / 2.0))


def _ensemble_weight(claim_verdicts: list[ClaimVerdict]) -> float:
    n_informative = sum(1 for cv in claim_verdicts if cv["verdict"] != "neutral")
    return _W_ENSEMBLE_NO_EVIDENCE if n_informative == 0 else _W_ENSEMBLE_WITH_EVIDENCE


def _combine(ensemble_score: float, claim_verdicts: list[ClaimVerdict]) -> tuple[float, float, float]:
    """Returns (final_score, evidence_score, ensemble_weight_used)."""
    ev_score = _evidence_score(claim_verdicts)
    w = _ensemble_weight(claim_verdicts)
    final_score = max(0.0, min(1.0, w * ensemble_score + (1 - w) * ev_score))
    return final_score, ev_score, w


def _verdict_from_score(final_score: float, low: float, high: float) -> tuple[str, float]:
    if final_score >= high:
        return "likely_real", round(final_score, 4)
    if final_score <= low:
        return "likely_fake", round(1.0 - final_score, 4)
    return "uncertain", round(abs(final_score - 0.5) * 2 * 0.5, 4)


def _build_trace(
    ensemble_score: float,
    evidence_score: float,
    ensemble_weight: float,
    final_score: float,
    verdict: str,
    claim_verdicts: list[ClaimVerdict],
) -> str:
    n_ent = sum(1 for v in claim_verdicts if v["verdict"] == "entailment")
    n_con = sum(1 for v in claim_verdicts if v["verdict"] == "contradiction")
    n_neu = sum(1 for v in claim_verdicts if v["verdict"] == "neutral")

    lines = [
        f"Ensemble model probability of real news: {ensemble_score:.3f}",
        f"Evidence-based score (from {n_ent} supporting, {n_con} contradicting, "
        f"{n_neu} neutral claim verdicts): {evidence_score:.3f}",
        f"Blend weights: ensemble={ensemble_weight:.2f}, evidence={1 - ensemble_weight:.2f} "
        f"({'evidence found informative signal' if ensemble_weight < 0.5 else 'no informative evidence — trusting ensemble'})",
        f"Combined score: {final_score:.3f}",
        f"Final verdict: {verdict}",
    ]

    if claim_verdicts:
        lines.append("Claim-level breakdown:")
        for cv in claim_verdicts:
            lines.append(
                f"  [{cv['verdict'].upper()[:3]} {cv['confidence']:.2f}] {cv['claim'][:80]}"
            )

    return "\n".join(lines)


def adjudicate_node(state: PipelineState) -> dict:
    """LangGraph node: combine evidence-grounded claim verdicts + ensemble score."""
    ensemble_score = state.get("ensemble_score", 0.5)
    claim_verdicts = state.get("claim_verdicts", [])

    log.info(
        "adjudicator: ensemble=%.3f, %d claim verdicts",
        ensemble_score,
        len(claim_verdicts),
    )

    final_score, ev_score, w = _combine(ensemble_score, claim_verdicts)
    verdict, confidence = _verdict_from_score(final_score, _LOW, _HIGH)
    needs_human_review = verdict == "uncertain"

    reasoning_trace = _build_trace(
        ensemble_score, ev_score, w, final_score, verdict, claim_verdicts
    )

    log.info(
        "adjudicator: verdict=%s conf=%.4f human_review=%s",
        verdict,
        confidence,
        needs_human_review,
    )

    return {
        "final_verdict":      verdict,
        "final_confidence":   confidence,
        "reasoning_trace":    reasoning_trace,
        "needs_human_review": needs_human_review,
    }


# ── Public helper (used by tests) ──────────────────────────────────────────────

def compute_verdict(
    ensemble_score: float,
    claim_verdicts: list[ClaimVerdict],
    escalation_low: float = _LOW,
    escalation_high: float = _HIGH,
) -> dict:
    """Pure-function wrapper for unit tests — no LangGraph state required."""
    final_score, _ev_score, _w = _combine(ensemble_score, claim_verdicts)
    verdict, confidence = _verdict_from_score(final_score, escalation_low, escalation_high)

    return {
        "verdict":            verdict,
        "confidence":         confidence,
        "final_score":        round(final_score, 4),
        "needs_human_review": verdict == "uncertain",
    }
