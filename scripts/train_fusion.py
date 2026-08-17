"""
Train the calibrated fusion adjudicator.

Input is the per-article feature CSV that scripts/evaluate_pipeline.py writes.
Output is models/fusion/adjudicator_v1.joblib — a calibrated logistic
regression over the frozen FEATURE_NAMES vector, plus the abstention
thresholds chosen to hold a target precision.

This is the step that replaces asserted weights with learned ones. The old
adjudicator gave evidence 85% of the vote and the ensemble 15% because someone
decided so; nothing measured whether that was right, and it sat on top of an
evidence signal that was inverted. Here the weights come from labelled data,
the probabilities are calibrated so a reported 0.9 means something, and the
thresholds are chosen against a precision promise whose coverage cost is
reported rather than hidden.

Note it does **not** touch models/ base models. It adds one new file.

Run:
    py scripts/evaluate_pipeline.py --slice holdout --n-per-class 150
    py scripts/train_fusion.py --input results/eval_per_article_holdout.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import settings  # noqa: E402
from paper_trail_core.agents.features import FEATURE_NAMES  # noqa: E402
from paper_trail_core.eval import metrics as M  # noqa: E402

log = logging.getLogger("train_fusion")

DEFAULT_OUT = os.path.join(BASE_DIR, "models", "fusion", "adjudicator_v1.joblib")

# Isotonic calibration needs a lot of data to beat Platt scaling and overfits
# badly below that — it produces a step function that looks excellent on the
# fitting split and falls apart on held-out data. Below this many samples,
# sigmoid (Platt) is used unconditionally.
_ISOTONIC_MIN_SAMPLES = 2000


def load_features(paths: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X, y, groups = [], [], []
    for path in paths:
        with open(path, "r", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                X.append([float(row.get(f"f_{k}", 0.0) or 0.0) for k in FEATURE_NAMES])
                y.append(int(row["label"]))
                groups.append(row.get("source", "unknown"))
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=int), groups


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=BASE_DIR, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", nargs="+", required=True,
                    help="per-article CSV(s) from evaluate_pipeline.py")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--target-precision", type=float, default=settings.TARGET_PRECISION)
    ap.add_argument("--test-size", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X, y, _groups = load_features(args.input)
    if len(X) < 40:
        log.error("only %d rows — too few to fit a fusion model. Run the "
                  "evaluation harness over a larger slice first.", len(X))
        return 1
    if len(set(y)) < 2:
        log.error("all %d rows share one label; cannot fit a classifier.", len(X))
        return 1

    # Dev fits the model and picks the thresholds; test reports the numbers.
    # Doing both on the same rows is precisely the mistake that produced this
    # project's original "100% accuracy".
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    log.info("dev=%d test=%d (%d features)", len(X_dev), len(X_test), X.shape[1])

    method = "isotonic" if len(X_dev) >= _ISOTONIC_MIN_SAMPLES else "sigmoid"
    log.info("calibration method: %s (dev n=%d)", method, len(X_dev))

    base = Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)),
    ])
    n_folds = min(5, int(np.bincount(y_dev).min()))
    if n_folds < 2:
        log.error("not enough samples in the minority class to cross-validate")
        return 1
    model = CalibratedClassifierCV(base, method=method, cv=n_folds).fit(X_dev, y_dev)

    p_dev = model.predict_proba(X_dev)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]

    thresholds = M.select_thresholds(y_dev.tolist(), p_dev.tolist(), args.target_precision)
    if not thresholds["target_met"]:
        log.warning(
            "target precision %.2f was NOT reachable on dev at any abstention band "
            "(best worst-class precision %.3f). Saving anyway with the best band — "
            "the shortfall is recorded in the payload and reported by the harness.",
            args.target_precision, thresholds["min_precision"],
        )

    dev_metrics = M.summarise_scores(y_dev.tolist(), p_dev.tolist())
    test_metrics = M.summarise_scores(y_test.tolist(), p_test.tolist())

    # Coefficients from a plain LR on the full dev set — the calibrated wrapper
    # holds an ensemble of fold models, so this is the interpretable summary of
    # what the fusion actually learned. Worth reading: it is the empirical
    # answer to "how much is the style classifier worth?"
    plain = Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)),
    ]).fit(X_dev, y_dev)
    coefs = dict(zip(FEATURE_NAMES, [round(float(c), 4) for c in plain["lr"].coef_[0]]))

    payload = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "thresholds": {"low": thresholds["low"], "high": thresholds["high"]},
        "target_precision": args.target_precision,
        "target_met": thresholds["target_met"],
        "expected_coverage": thresholds["coverage"],
        "calibration_method": method,
        "coefficients": coefs,
        "dev_metrics": dev_metrics,
        "test_metrics": test_metrics,
        "n_dev": int(len(X_dev)),
        "n_test": int(len(X_test)),
        "sources": args.input,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
    }

    out = args.out if os.path.isabs(args.out) else os.path.join(BASE_DIR, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    import joblib
    joblib.dump(payload, out)

    print(f"\n{'='*64}\n  Fusion adjudicator trained\n{'='*64}")
    print(f"  dev  n={len(X_dev):<5} acc={dev_metrics['accuracy']:.3f} "
          f"auc={dev_metrics['roc_auc']} ece={dev_metrics['ece']:.3f}")
    print(f"  test n={len(X_test):<5} acc={test_metrics['accuracy']:.3f} "
          f"auc={test_metrics['roc_auc']} ece={test_metrics['ece']:.3f}")
    print(f"\n  abstention band : {thresholds['low']:.3f} .. {thresholds['high']:.3f}")
    print(f"  target precision: {args.target_precision:.2f} "
          f"({'met' if thresholds['target_met'] else 'NOT met'})")
    print(f"  expected coverage: {thresholds['coverage']:.1%}")
    print("\n  learned weights (standardised, + favours 'real'):")
    for name, c in sorted(coefs.items(), key=lambda kv: -abs(kv[1])):
        print(f"    {name:<22} {c:+.4f}")
    print(f"\n  saved -> {os.path.relpath(out, BASE_DIR)}\n")

    summary_path = os.path.join(BASE_DIR, "results", "fusion_training.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump({k: v for k, v in payload.items() if k != "model"}, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
