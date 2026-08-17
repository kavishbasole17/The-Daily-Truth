"""
The article-level feature vector consumed by the fusion adjudicator.

There is exactly one definition of this vector, used by both training
(scripts/train_fusion.py) and serving (adjudicator.py). Feature-order drift
between the two is the classic way a fusion layer silently starts reading the
wrong column and nobody notices for months, so FEATURE_NAMES is frozen here,
persisted inside the trained model payload, and asserted on load.
"""
from __future__ import annotations

from typing import Any

from paper_trail_core.agents.state import ClaimVerdict

FEATURE_NAMES: list[str] = [
    "ensemble_score",       # style classifier's P(real)
    "mean_support",         # mean of per-claim (entail_mass - contradict_mass)
    "max_support",          # strongest corroboration
    "min_support",          # strongest refutation
    "entail_mass_max",
    "contradict_mass_max",
    "coverage",             # fraction of claims we found relevant evidence for
    "informative_ratio",    # fraction of claims whose evidence was decisive
    "n_claims_norm",        # n_claims / 8, clipped
    "mean_relevance",
    "mean_trust",
    "numeric_conflict_max",
    "nli_degraded",         # 1.0 when evidence came from the lexical fallback
]


def _clip(x: float, lo: float = 0.0, hi: float = 1.0, default: float = 0.0) -> float:
    """
    Clamp to [lo, hi], mapping non-finite values to `default`.

    The NaN handling is not defensive padding — it is load-bearing. Python's
    `min`/`max` treat NaN comparisons as False, so `max(0.0, min(1.0, nan))`
    returns `1.0`: a NaN model score would silently become maximum confidence
    in "real" rather than an error. A base model emitting NaN on degenerate
    input is entirely possible, and that failure would be invisible.
    """
    try:
        x = float(x)
    except (TypeError, ValueError):
        return default
    if x != x or x in (float("inf"), float("-inf")):   # NaN or ±inf
        return default
    return max(lo, min(hi, x))


def build_features(
    ensemble_score: float,
    claim_verdicts: list[ClaimVerdict],
    retrieval_stats: dict[str, Any] | None = None,
    nli_degraded: bool = False,
) -> dict[str, float]:
    """
    Assemble the feature dict. Always returns every key in FEATURE_NAMES.

    Claims with no usable evidence are excluded from the support statistics
    rather than counted as support 0.0. Averaging them in would drag every
    article toward neutral in proportion to how much evidence we *failed* to
    find, which is a property of our retrieval budget, not of the article.
    That failure is represented honestly by `coverage` instead.
    """
    from paper_trail_core.agents.verifier import is_informative

    stats = retrieval_stats or {}
    graded = [cv for cv in claim_verdicts if cv.get("status") == "ok"]
    supports = [float(cv.get("support", 0.0)) for cv in graded]

    n_claims = len(claim_verdicts)
    n_informative = sum(1 for cv in claim_verdicts if is_informative(cv))

    # An unusable ensemble score defaults to 0.5 (no opinion) rather than 0.0,
    # which would read as a confident "fake".
    feats = {
        "ensemble_score": _clip(ensemble_score, default=0.5),
        "mean_support": _clip(sum(supports) / len(supports) if supports else 0.0, -1.0, 1.0),
        "max_support": _clip(max(supports) if supports else 0.0, -1.0, 1.0),
        "min_support": _clip(min(supports) if supports else 0.0, -1.0, 1.0),
        "entail_mass_max": _clip(max((cv.get("entail_mass", 0.0) for cv in graded), default=0.0)),
        "contradict_mass_max": _clip(max((cv.get("contradict_mass", 0.0) for cv in graded), default=0.0)),
        "coverage": _clip(len(graded) / n_claims) if n_claims else 0.0,
        "informative_ratio": _clip(n_informative / n_claims) if n_claims else 0.0,
        "n_claims_norm": _clip(n_claims / 8.0),
        "mean_relevance": _clip(stats.get("mean_relevance", 0.0)),
        "mean_trust": _clip(stats.get("mean_trust", 0.0)),
        "numeric_conflict_max": _clip(
            max((cv.get("numeric_conflict", 0.0) for cv in claim_verdicts), default=0.0)
        ),
        "nli_degraded": 1.0 if nli_degraded else 0.0,
    }
    return {k: round(feats[k], 6) for k in FEATURE_NAMES}


def to_vector(features: dict[str, float]) -> list[float]:
    """Feature dict -> ordered vector. Missing keys default to 0.0."""
    return [float(features.get(name, 0.0)) for name in FEATURE_NAMES]
