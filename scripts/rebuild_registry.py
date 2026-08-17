"""
Add the missing `hybrid_stacking` entry to models/registry.json.

models/registry.json was written before the hybrid model was trained, so it
never listed it. The consequence was invisible but real: `registry.predict(
text, "hybrid_stacking")` raised KeyError and `predict_all` skipped the hybrid
entirely, which is why the ensemble agent reached past the public API into the
private `_predict_hybrid`. Private-API calls are how a refactor silently breaks
one caller, so the registry gets fixed instead.

This does **not** retrain anything. It reads the already-trained
models/hybrid/stacking_lr.joblib and writes one metadata entry describing it.
The model files are untouched.

Run:  py scripts/rebuild_registry.py
"""
from __future__ import annotations

import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

REGISTRY_PATH = os.path.join(BASE_DIR, "models", "registry.json")
HYBRID_PATH = os.path.join(BASE_DIR, "models", "hybrid", "stacking_lr.joblib")


def main() -> int:
    if not os.path.exists(REGISTRY_PATH):
        print(f"registry not found: {REGISTRY_PATH}")
        return 1
    if not os.path.exists(HYBRID_PATH):
        print(f"hybrid model not found: {HYBRID_PATH} — run export_models.py first")
        return 1

    import joblib

    with open(REGISTRY_PATH, "r", encoding="utf-8") as fh:
        registry = json.load(fh)

    payload = joblib.load(HYBRID_PATH)
    base_names = list(payload.get("base_model_names", []))

    # Preserve any metrics an earlier export recorded; we are not in a position
    # to invent accuracy numbers for a model we did not just evaluate.
    existing = registry.get("hybrid_stacking", {})

    registry["hybrid_stacking"] = {
        "display_name": "Hybrid Stacking (LR)",
        "category": "Hybrid",
        "feature_type": "hybrid",
        "path": "models/hybrid/stacking_lr.joblib",
        "train_time": existing.get("train_time", 0.0),
        "metrics": existing.get("metrics", {}),
        "n_base_models": len(base_names),
        "base_model_names": base_names,
    }

    tmp = f"{REGISTRY_PATH}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(registry, fh, indent=2)
    os.replace(tmp, REGISTRY_PATH)

    print(f"registry.json updated — hybrid_stacking now listed ({len(base_names)} base models)")
    print(f"entries: {len(registry)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
