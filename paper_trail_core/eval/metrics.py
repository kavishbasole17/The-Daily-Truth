"""
Evaluation metrics.

Beyond the usual accuracy/P/R/F1, two things here matter more than the
headline number for this project:

**Calibration.** The old ensemble returned 0.999 for wire copy and 0.961 for a
fabricated article. Those numbers were not probabilities, they were scores that
happened to lie in [0, 1] — and a system that reports 96% confidence while
being wrong is worse than one that reports 55%. Expected Calibration Error and
the reliability table measure whether a stated confidence means anything.

**Coverage vs. precision.** A detector allowed to abstain trades away coverage
for precision. Reporting one without the other is meaningless: 99% precision at
3% coverage is a useless system, and so is 60% precision at 100% coverage. The
curve shows the whole trade-off so an operating point can be chosen honestly.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


def sanitise(probs: Sequence[float], default: float = 0.5) -> list[float]:
    """
    Coerce a score sequence to finite values in [0, 1].

    A single NaN used to abort a whole evaluation run inside
    `expected_calibration_error` (`int(nan * n_bins)` raises). Losing an
    hour of retrieval to one bad row is not an acceptable failure mode, and
    silently treating NaN as a valid score is worse — mapping it to 0.5 keeps
    it in the metrics as the non-informative value it is.
    """
    out = []
    for p in probs:
        try:
            v = float(p)
        except (TypeError, ValueError):
            v = default
        if not math.isfinite(v):   # NaN or ±inf
            v = default
        out.append(max(0.0, min(1.0, v)))
    return out


def confusion(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, int]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return round(precision, 4), round(recall, 4), round(f1, 4)


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, Any]:
    """Accuracy plus per-class precision/recall/F1. Class 1 = real, 0 = fake."""
    if not y_true:
        return {"n": 0}
    c = confusion(y_true, y_pred)
    p_real, r_real, f_real = _prf(c["tp"], c["fp"], c["fn"])
    # For the "fake" class the roles of the off-diagonal cells swap.
    p_fake, r_fake, f_fake = _prf(c["tn"], c["fn"], c["fp"])
    n = len(y_true)
    return {
        "n": n,
        "accuracy": round((c["tp"] + c["tn"]) / n, 4),
        "confusion": c,
        "real": {"precision": p_real, "recall": r_real, "f1": f_real},
        "fake": {"precision": p_fake, "recall": r_fake, "f1": f_fake},
        "macro_f1": round((f_real + f_fake) / 2, 4),
    }


def roc_auc(y_true: Sequence[int], scores: Sequence[float]) -> float | None:
    """
    AUC via the rank-sum identity, with ties averaged. No sklearn dependency
    so this stays usable from anywhere in the codebase.
    """
    scores = sanitise(scores)
    pairs = sorted(zip(scores, y_true), key=lambda x: x[0])
    n_pos = sum(1 for _, y in pairs if y == 1)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    ranks: list[float] = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1

    rank_sum = sum(r for r, (_, y) in zip(ranks, pairs) if y == 1)
    return round((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg), 4)


def brier_score(y_true: Sequence[int], probs: Sequence[float]) -> float:
    if not y_true:
        return 0.0
    probs = sanitise(probs)
    return round(sum((p - y) ** 2 for y, p in zip(y_true, probs)) / len(y_true), 4)


def expected_calibration_error(
    y_true: Sequence[int],
    probs: Sequence[float],
    n_bins: int = 15,
) -> tuple[float, float, list[dict[str, Any]]]:
    """
    Returns (ECE, MCE, reliability_table).

    Each bin reports how often articles the model called X% likely to be real
    actually were. A well-calibrated model tracks the diagonal; the gap column
    is what ECE averages.
    """
    if not y_true:
        return 0.0, 0.0, []

    probs = sanitise(probs)
    bins: list[list[tuple[int, float]]] = [[] for _ in range(n_bins)]
    for y, p in zip(y_true, probs):
        idx = min(n_bins - 1, int(p * n_bins))
        bins[idx].append((y, p))

    table = []
    ece = 0.0
    mce = 0.0
    total = len(y_true)
    for i, bucket in enumerate(bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        if not bucket:
            table.append({"bin": f"{lo:.2f}-{hi:.2f}", "n": 0,
                          "mean_confidence": None, "observed_accuracy": None, "gap": None})
            continue
        mean_conf = sum(p for _, p in bucket) / len(bucket)
        observed = sum(y for y, _ in bucket) / len(bucket)
        gap = abs(mean_conf - observed)
        ece += (len(bucket) / total) * gap
        mce = max(mce, gap)
        table.append({
            "bin": f"{lo:.2f}-{hi:.2f}",
            "n": len(bucket),
            "mean_confidence": round(mean_conf, 4),
            "observed_accuracy": round(observed, 4),
            "gap": round(gap, 4),
        })
    return round(ece, 4), round(mce, 4), table


def coverage_precision_curve(
    y_true: Sequence[int],
    probs: Sequence[float],
    n_points: int = 40,
) -> list[dict[str, Any]]:
    """
    Sweep a symmetric abstention band and record what it buys.

    At each half-width h, articles with p in (0.5-h, 0.5+h) abstain; the rest
    are committed. Reports coverage and the precision of each committed class.
    """
    probs = sanitise(probs)
    out = []
    for i in range(n_points + 1):
        h = 0.5 * i / n_points
        low, high = 0.5 - h, 0.5 + h
        committed = [(y, p) for y, p in zip(y_true, probs) if p <= low or p >= high]
        if not committed:
            continue
        y_c = [y for y, _ in committed]
        p_c = [1 if p >= high else 0 for _, p in committed]
        m = classification_metrics(y_c, p_c)
        out.append({
            "band_half_width": round(h, 4),
            "low": round(low, 4),
            "high": round(high, 4),
            "coverage": round(len(committed) / len(y_true), 4),
            "accuracy": m["accuracy"],
            "precision_real": m["real"]["precision"],
            "precision_fake": m["fake"]["precision"],
            "min_precision": round(min(m["real"]["precision"], m["fake"]["precision"]), 4),
        })
    return out


def select_thresholds(
    y_true: Sequence[int],
    probs: Sequence[float],
    target_precision: float = 0.90,
    min_committed: int = 10,
) -> dict[str, Any]:
    """
    Choose the widest abstention band that still meets `target_precision` on
    both committed classes — equivalently, the highest coverage compatible
    with the precision promise.

    If no band reaches the target (which is a real possibility and must not be
    papered over), returns the best available with `target_met=False` so the
    caller reports the shortfall rather than pretending.
    """
    curve = coverage_precision_curve(y_true, probs, n_points=100)
    feasible = [
        pt for pt in curve
        if pt["min_precision"] >= target_precision
        and pt["coverage"] * len(y_true) >= min_committed
    ]
    if feasible:
        best = max(feasible, key=lambda pt: pt["coverage"])
        return {
            "low": best["low"], "high": best["high"],
            "coverage": best["coverage"], "min_precision": best["min_precision"],
            "target_precision": target_precision, "target_met": True,
        }

    best = max(curve, key=lambda pt: pt["min_precision"]) if curve else None
    if best is None:
        return {"low": 0.4, "high": 0.6, "coverage": 0.0, "min_precision": 0.0,
                "target_precision": target_precision, "target_met": False}
    return {
        "low": best["low"], "high": best["high"],
        "coverage": best["coverage"], "min_precision": best["min_precision"],
        "target_precision": target_precision, "target_met": False,
    }


def summarise_scores(
    y_true: Sequence[int],
    probs: Sequence[float],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Everything at once, for a single scoring channel (fused, ensemble, ...)."""
    probs = sanitise(probs)
    y_pred = [1 if p >= threshold else 0 for p in probs]
    ece, mce, table = expected_calibration_error(y_true, probs)
    return {
        **classification_metrics(y_true, y_pred),
        "roc_auc": roc_auc(y_true, probs),
        "brier": brier_score(y_true, probs),
        "ece": ece,
        "mce": mce,
        "reliability": table,
        "mean_prob": round(sum(probs) / len(probs), 4) if probs else None,
    }
