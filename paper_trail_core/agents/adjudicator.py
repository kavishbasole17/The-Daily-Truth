"""
Adjudicator agent node — pure computation, no external calls.

Turns the feature vector into a final verdict, and decides when *not* to give
one.

The abstention band is the point of this stage. A detector that must commit to
every article ships confident errors; one that can say "uncertain" and route to
a human keeps its committed verdicts trustworthy. The band's edges are chosen
on held-out data to hold a target precision (config.TARGET_PRECISION), and the
coverage that costs is measured and reported rather than hidden.

Two hard gates force abstention regardless of score:
  * NLI ran degraded — the fallback verifier cannot detect contradiction and
    has historically scored refutations as support, so its output must never
    reach a committed verdict.
  * No claim could be checked at all — then the only remaining signal is the
    style classifier, which on novel text is roughly a coin flip weighted
    toward "real".
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Any, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from config import settings
    _LOW  = settings.ESCALATION_LOW
    _HIGH = settings.ESCALATION_HIGH
except ImportError:  # pragma: no cover
    _LOW, _HIGH = 0.4, 0.6

from paper_trail_core.agents import fusion
from paper_trail_core.agents.features import build_features
from paper_trail_core.agents.state import ClaimVerdict, PipelineState
from paper_trail_core.degradations import degrade, format_degradations

log = logging.getLogger(__name__)

_SENTINEL = fusion.UNSET


def _verdict_from_score(p_real: float, low: float, high: float) -> tuple[str, float]:
    """
    Map a calibrated P(real) to a verdict and a confidence.

    Confidence is the calibrated probability of the class actually reported —
    not a distance-from-0.5 rescaling. The previous formula returned
    `abs(score - 0.5) * 2 * 0.5`, capped at 0.1 inside the uncertain band, so
    the UI displayed "10% Confidence" on literally every uncertain verdict.
    Reporting "we're 58% leaning real, which isn't enough to commit" is both
    honest and useful.
    """
    if p_real >= high:
        return "likely_real", round(p_real, 4)
    if p_real <= low:
        return "likely_fake", round(1.0 - p_real, 4)
    return "uncertain", round(max(p_real, 1.0 - p_real), 4)


def _build_trace(
    features: dict[str, float],
    p_real: float,
    method: str,
    verdict: str,
    confidence: float,
    claim_verdicts: list[ClaimVerdict],
    low: float,
    high: float,
    degradations: list,
    forced_reason: Optional[str] = None,
) -> str:
    n = {v: sum(1 for c in claim_verdicts if c.get("verdict") == v)
         for v in ("entailment", "contradiction", "neutral", "unverifiable")}

    lines = [
        "EVIDENCE SUMMARY",
        f"  Claims checked            : {len(claim_verdicts)}",
        f"  Supported by evidence     : {n['entailment']}",
        f"  Contradicted by evidence  : {n['contradiction']}",
        f"  Evidence inconclusive     : {n['neutral']}",
        f"  Could not be checked      : {n['unverifiable']}",
        f"  Evidence coverage         : {features.get('coverage', 0.0):.0%} of claims",
        f"  Mean evidence relevance   : {features.get('mean_relevance', 0.0):.3f}",
        f"  Mean source trust         : {features.get('mean_trust', 0.0):.3f}",
        "",
        "SCORING",
        f"  Style-classifier P(real)  : {features.get('ensemble_score', 0.5):.3f}",
        f"  Mean claim support        : {features.get('mean_support', 0.0):+.3f}  (-1 refuted .. +1 supported)",
        f"  Strongest refutation      : {features.get('min_support', 0.0):+.3f}",
        f"  Fusion method             : {method}",
        f"  Combined P(real)          : {p_real:.3f}",
        f"  Abstention band           : {low:.3f} .. {high:.3f}",
        "",
        f"VERDICT: {verdict}  (confidence {confidence:.1%})",
    ]

    if forced_reason:
        lines.append(f"  Held for review: {forced_reason}")

    if claim_verdicts:
        lines += ["", "CLAIM-BY-CLAIM"]
        for cv in claim_verdicts:
            tag = cv.get("verdict", "?").upper()[:6].ljust(6)
            src = ", ".join(dict.fromkeys(s for s in cv.get("sources", []) if s)) or "no source"
            lines.append(
                f"  [{tag} {cv.get('confidence', 0.0):.2f}] {cv.get('claim', '')[:96]}"
            )
            lines.append(
                f"          support={cv.get('support', 0.0):+.2f} "
                f"relevance={cv.get('top_relevance', 0.0):.2f} via {src[:70]}"
            )

    trace = "\n".join(lines)
    return trace + format_degradations(degradations)


def compute_verdict(
    ensemble_score: float,
    claim_verdicts: list[ClaimVerdict],
    retrieval_stats: Optional[dict] = None,
    nli_degraded: bool = False,
    fusion_payload: Any = _SENTINEL,
    escalation_low: Optional[float] = None,
    escalation_high: Optional[float] = None,
) -> dict:
    """
    Pure function: signals in, verdict out. No LangGraph state, no I/O.

    Pass `fusion_payload=None` to force the deterministic heuristic path —
    that's what the unit tests do, so they don't depend on whether a fusion
    model happens to be trained on the machine running them.
    """
    payload = fusion.load_fusion() if fusion_payload is _SENTINEL else fusion_payload

    features = build_features(ensemble_score, claim_verdicts, retrieval_stats, nli_degraded)
    p_real, method = fusion.fused_probability(features, payload)

    t_low, t_high = fusion.thresholds(payload)
    low = t_low if escalation_low is None else escalation_low
    high = t_high if escalation_high is None else escalation_high

    verdict, confidence = _verdict_from_score(p_real, low, high)

    # Hard gates — abstain regardless of the score.
    forced_reason = None
    if nli_degraded:
        forced_reason = (
            "claim checking ran on the degraded word-overlap backend, which "
            "cannot reliably detect contradiction"
        )
    elif features["coverage"] == 0.0 and claim_verdicts:
        forced_reason = "no claim could be checked against any source"

    if forced_reason and verdict != "uncertain":
        verdict = "uncertain"
        confidence = round(max(p_real, 1.0 - p_real), 4)

    return {
        "verdict": verdict,
        "confidence": confidence,
        "final_score": round(p_real, 4),
        "fused_score": round(p_real, 4),
        "fusion_method": method,
        "features": features,
        "needs_human_review": verdict == "uncertain",
        "forced_reason": forced_reason,
        "thresholds": {"low": low, "high": high},
    }


def adjudicate_node(state: PipelineState) -> dict:
    """LangGraph node: fuse all signals into the final verdict."""
    ensemble_score = state.get("ensemble_score", 0.5)
    claim_verdicts = state.get("claim_verdicts", [])
    retrieval_stats = state.get("retrieval_stats", {})
    nli_degraded = bool(state.get("nli_degraded", False))
    prior_degradations = list(state.get("degradations", []))

    result = compute_verdict(
        ensemble_score, claim_verdicts, retrieval_stats, nli_degraded
    )

    new_degradations: list = []
    if result["fusion_method"] == "heuristic_fallback":
        new_degradations += degrade(
            "adjudicator", "fusion_untrained",
            "No trained fusion model was found, so the verdict used the "
            "conservative hand-set blend. Run scripts/train_fusion.py to enable "
            "the calibrated model.",
            severity="info",
        )
    if result["forced_reason"]:
        new_degradations += degrade(
            "adjudicator", "forced_abstention",
            f"Verdict held for human review: {result['forced_reason']}.",
        )

    trace = _build_trace(
        result["features"], result["fused_score"], result["fusion_method"],
        result["verdict"], result["confidence"], claim_verdicts,
        result["thresholds"]["low"], result["thresholds"]["high"],
        prior_degradations + new_degradations, result["forced_reason"],
    )

    log.info(
        "adjudicator: verdict=%s conf=%.4f p_real=%.4f method=%s review=%s",
        result["verdict"], result["confidence"], result["fused_score"],
        result["fusion_method"], result["needs_human_review"],
    )

    return {
        "final_verdict":      result["verdict"],
        "final_confidence":   result["confidence"],
        "fused_score":        result["fused_score"],
        "fusion_features":    result["features"],
        "fusion_method":      result["fusion_method"],
        "reasoning_trace":    trace,
        "needs_human_review": result["needs_human_review"],
        "degradations":       new_degradations,
    }
