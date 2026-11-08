"""
Fusion layer — combines the style classifier and the evidence signal into one
calibrated probability that the article is real.

The previous adjudicator hard-coded the blend: evidence got 85% of the vote
whenever any claim came back non-neutral, and 20% otherwise. Both numbers were
asserted, never validated, and they sat on top of an evidence signal that was
itself broken — so 85% of the decision came from a heuristic that scored
contradicting evidence as support.

Two changes. The weights are *learned* from labelled data, and the output is
*calibrated*, so a reported 0.9 means the article is real about 90% of the
time rather than being an arbitrary point on an uncalibrated scale.

When no trained model is present the heuristic fallback below runs instead. It
is deliberately conservative and its structure differs from the old one in a
way that matters: evidence weight scales continuously with coverage rather
than flipping on a binary "did any claim come back non-neutral" test. Low
coverage now yields a middling score — which lands in the abstention band —
instead of a confident verdict extrapolated from one lucky snippet.
"""
from __future__ import annotations

import logging
import math
import os
import sys
import threading
from typing import Any

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from config import settings
    _MODEL_PATH = settings.FUSION_MODEL_PATH
    _LOW_DEFAULT = settings.ESCALATION_LOW
    _HIGH_DEFAULT = settings.ESCALATION_HIGH
except ImportError:  # pragma: no cover
    _MODEL_PATH, _LOW_DEFAULT, _HIGH_DEFAULT = "models/fusion/adjudicator_v1.joblib", 0.4, 0.6

from paper_trail_core.agents.features import FEATURE_NAMES, to_vector

log = logging.getLogger(__name__)

_payload: dict | None = None
_loaded = False
_lock = threading.Lock()

# Distinguishes "no payload supplied — go load one" from "explicitly no payload
# — use the fallback". Overloading `None` for both meant callers could not
# force the deterministic path once a trained model existed on disk.
UNSET = object()

# Cap on how much the evidence channel can pull the score in the fallback path.
_MAX_EVIDENCE_WEIGHT = 0.85


def load_fusion(path: str | None = None, force: bool = False) -> dict | None:
    """
    Load the trained fusion payload, or None if it hasn't been trained yet.

    A payload whose feature order disagrees with FEATURE_NAMES is rejected
    outright rather than used: silently reading feature k as feature j produces
    plausible-looking probabilities that are wrong in a way no test would catch.
    """
    global _payload, _loaded
    if _loaded and not force:
        return _payload
    with _lock:
        if _loaded and not force:
            return _payload
        _loaded = True
        _payload = None

        resolved = path or _MODEL_PATH
        if not os.path.isabs(resolved):
            resolved = os.path.join(_ROOT, resolved)
        if not os.path.exists(resolved):
            log.info("fusion: no trained model at %s — using heuristic fallback", resolved)
            return None

        try:
            import joblib
            payload = joblib.load(resolved)
            names = list(payload.get("feature_names", []))
            if names != FEATURE_NAMES:
                log.error(
                    "fusion: feature order mismatch in %s — refusing to use it. "
                    "Retrain with scripts/train_fusion.py. (model=%s, code=%s)",
                    resolved, names, FEATURE_NAMES,
                )
                return None
            _payload = payload
            log.info(
                "fusion: loaded %s (thresholds low=%.3f high=%.3f, target_precision=%.2f)",
                resolved,
                payload.get("thresholds", {}).get("low", _LOW_DEFAULT),
                payload.get("thresholds", {}).get("high", _HIGH_DEFAULT),
                payload.get("target_precision", 0.0),
            )
        except Exception as exc:
            log.error("fusion: could not load %s (%s) — using heuristic fallback", resolved, exc)
            _payload = None
    return _payload


def heuristic_probability(features: dict[str, float]) -> float:
    """
    Hand-set fallback used when no fusion model has been trained.

    Evidence weight scales with how much of the article we could actually
    check, and collapses to zero when NLI was degraded — a verifier that
    cannot detect contradiction contributes confidently wrong signal, which is
    strictly worse than contributing none.
    """
    coverage = features.get("coverage", 0.0)
    degraded = features.get("nli_degraded", 0.0)
    ensemble = features.get("ensemble_score", 0.5)
    mean_support = features.get("mean_support", 0.0)
    min_support = features.get("min_support", 0.0)

    w_ev = _MAX_EVIDENCE_WEIGHT * coverage * (1.0 - degraded)

    # Refutation and corroboration are treated asymmetrically, and on purpose.
    # One claim in an article being directly contradicted by a trusted source
    # is strong evidence about the whole article; several claims being merely
    # corroborated is weak evidence, because true and false articles alike
    # contain checkable true statements. Averaging the two directions together
    # lets a fabricated article's handful of accurate background facts cancel
    # out the one sentence that gives it away.
    refutation = -min(0.0, min_support)          # 0 .. 1
    corroboration = max(0.0, mean_support)       # 0 .. 1
    evidence_p = 0.5 + 0.5 * corroboration - 0.9 * refutation

    p = w_ev * max(0.0, min(1.0, evidence_p)) + (1.0 - w_ev) * ensemble
    return max(0.0, min(1.0, p))


def fused_probability(
    features: dict[str, float],
    payload: Any = UNSET,
) -> tuple[float, str]:
    """
    Return (P(real), method) where method is "learned" or "heuristic_fallback".

    Pass `payload=None` to force the heuristic path; omit it to use the loaded
    model if one exists.
    """
    if payload is UNSET:
        payload = load_fusion()
    if payload is None:
        return _safe_probability(heuristic_probability(features)), "heuristic_fallback"

    try:
        model = payload["model"]
        vec = [to_vector(features)]
        p = float(model.predict_proba(vec)[0][1])
        if math.isnan(p):
            raise ValueError("model returned NaN")
        return _safe_probability(p), "learned"
    except Exception as exc:
        log.error("fusion: learned model failed at predict time (%s) — falling back", exc)
        return _safe_probability(heuristic_probability(features)), "heuristic_fallback"


def _safe_probability(p: float) -> float:
    """
    Clamp to [0, 1], mapping anything non-finite to 0.5.

    `max(0, min(1, nan))` returns 1.0 in Python, because every NaN comparison
    is False — so without this a NaN anywhere upstream becomes a *confident*
    "real" verdict rather than an error. 0.5 lands in the abstention band,
    which is the correct response to a score we cannot interpret.
    """
    try:
        p = float(p)
    except (TypeError, ValueError):
        return 0.5
    if not math.isfinite(p):   # NaN or ±inf
        return 0.5
    return max(0.0, min(1.0, p))


def thresholds(payload: Any = UNSET) -> tuple[float, float]:
    """Abstention band (low, high). Falls back to the configured defaults."""
    if payload is UNSET:
        payload = load_fusion()
    if payload:
        t = payload.get("thresholds") or {}
        return float(t.get("low", _LOW_DEFAULT)), float(t.get("high", _HIGH_DEFAULT))
    return float(_LOW_DEFAULT), float(_HIGH_DEFAULT)


def describe(payload: Any = UNSET) -> dict[str, Any]:
    """Metadata for the reasoning trace / API diagnostics."""
    if payload is UNSET:
        payload = load_fusion()
    if not payload:
        return {"method": "heuristic_fallback", "trained": False}
    return {
        "method": "learned",
        "trained": True,
        "trained_at": payload.get("trained_at"),
        "target_precision": payload.get("target_precision"),
        "thresholds": payload.get("thresholds"),
        "test_metrics": payload.get("test_metrics"),
    }
