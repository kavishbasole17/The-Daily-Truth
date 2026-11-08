"""
Compute the missing accuracy / f1 / precision / recall / auc for
`hybrid_stacking` and write them into models/registry.json.

Why this is needed: `export_models.py::train_and_save_hybrid` computes these
metrics at training time, but the `models/hybrid/stacking_lr.joblib` currently
on disk was trained before registry.json was updated to include a
`hybrid_stacking` entry at all. `scripts/rebuild_registry.py` later added the
entry so the model would be listed and predictable, but — correctly — refused
to invent numbers for a model it did not just evaluate, and stored an empty
`metrics: {}` instead. That empty dict is what the `/models` page renders as
"N/A" for every column.

This script does not retrain anything. It reproduces the exact held-out test
split `export_models.py` used (same `random_state=42`, and oversampling is
train-only per `preprocess.get_train_test_split`, so it never touches the test
set), replays the already-trained base models over that split in the base
model order the meta-learner expects, and scores the already-trained
`meta_learner` on the result — the same computation `train_and_save_hybrid`
would have produced if this run had reached that point originally.

Run:  py scripts/backfill_hybrid_metrics.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

CLASSICAL_DIR = os.path.join(BASE_DIR, "models", "classical")
DL_DIR = os.path.join(BASE_DIR, "models", "deep_learning")
HYBRID_PATH = os.path.join(BASE_DIR, "models", "hybrid", "stacking_lr.joblib")
REGISTRY_PATH = os.path.join(BASE_DIR, "models", "registry.json")


def _get_prob(model, X):
    """Same probability extraction used at inference time (api/model_loader.py)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        s = model.decision_function(X)
        s = s if s.ndim == 1 else s[:, 1]
        return 1.0 / (1.0 + np.exp(-s))
    return model.predict(X).astype(float)


def compute_metrics(model, X_te, y_te) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_pred = model.predict(X_te)
    try:
        y_prob = _get_prob(model, X_te)
        auc = float(roc_auc_score(y_te, y_prob))
    except Exception:
        auc = None
    return {
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "f1": float(f1_score(y_te, y_pred, average="weighted", zero_division=0)),
        "precision": float(precision_score(y_te, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_te, y_pred, average="weighted", zero_division=0)),
        "auc": auc,
    }


def main() -> int:
    import joblib

    for path in (HYBRID_PATH, REGISTRY_PATH):
        if not os.path.exists(path):
            print(f"not found: {path} — run export_models.py first")
            return 1

    with open(REGISTRY_PATH, "r", encoding="utf-8") as fh:
        registry = json.load(fh)

    payload = joblib.load(HYBRID_PATH)
    meta_learner = payload["meta_learner"]
    base_model_names = list(payload["base_model_names"])

    print(f"Reproducing the held-out test split for {len(base_model_names)} base models: "
          f"{base_model_names}")

    # ── Reproduce the exact test split (test-only; oversampling never touches it) ──
    from preprocess import get_train_test_split, load_all_datasets

    df = load_all_datasets()
    _, X_test, _, y_test = get_train_test_split(df)
    print(f"Test set: {len(X_test)} articles")

    # ── Feature pipelines ─────────────────────────────────────────────────────
    tfidf = joblib.load(os.path.join(CLASSICAL_DIR, "tfidf.joblib"))
    dense = joblib.load(os.path.join(CLASSICAL_DIR, "dense_pipeline.joblib"))
    X_test_tfidf = tfidf.transform(X_test)
    X_test_dense = dense["svd"].transform(dense["tfidf"].transform(X_test))

    dl_names = {"lstm", "gru", "cnn", "cnn_lstm"}
    needs_dl = any(name in dl_names for name in base_model_names)
    X_test_seq = None
    dl_models: dict[str, object] = {}
    if needs_dl:
        import tensorflow as tf
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.preprocessing.text import tokenizer_from_json

        with open(os.path.join(DL_DIR, "tokenizer.json")) as f:
            tokenizer = tokenizer_from_json(f.read())
        with open(os.path.join(DL_DIR, "config.json")) as f:
            dl_config = json.load(f)
        max_len = dl_config.get("max_len", 200)

        X_test_seq = pad_sequences(
            tokenizer.texts_to_sequences(X_test),
            maxlen=max_len, padding="post", truncating="post",
        )
        for name in base_model_names:
            if name in dl_names:
                dl_models[name] = tf.keras.models.load_model(
                    os.path.join(DL_DIR, f"{name}.keras")
                )
                print(f"  loaded DL base model: {name}")

    # ── Replay each base model over X_test, in the order the meta-learner expects ──
    meta_test_cols = []
    for name in base_model_names:
        if name in dl_models:
            meta_test_cols.append(dl_models[name].predict(X_test_seq, verbose=0).flatten())
            continue

        model_path = os.path.join(CLASSICAL_DIR, f"{name}.joblib")
        if not os.path.exists(model_path):
            print(f"  MISSING base model on disk: {name} — cannot reproduce this "
                  "meta-feature column, aborting.")
            return 1
        model = joblib.load(model_path)
        feature_type = registry.get(name, {}).get("feature_type", "tfidf")
        Xt = X_test_dense if feature_type == "dense" else X_test_tfidf
        meta_test_cols.append(_get_prob(model, Xt))
        print(f"  replayed classical base model: {name} ({feature_type})")

    X_test_stack = np.column_stack(meta_test_cols)
    print(f"Meta-feature test matrix: {X_test_stack.shape}")

    t0 = time.time()
    metrics = compute_metrics(meta_learner, X_test_stack, y_test)
    print(f"Computed in {time.time() - t0:.1f}s: {metrics}")

    registry.setdefault("hybrid_stacking", {})
    registry["hybrid_stacking"].update({
        "display_name": "Hybrid Stacking (LR)",
        "category": "Hybrid",
        "feature_type": "hybrid",
        "path": "models/hybrid/stacking_lr.joblib",
        "metrics": metrics,
        "n_base_models": len(base_model_names),
        "base_model_names": base_model_names,
    })

    tmp = f"{REGISTRY_PATH}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(registry, fh, indent=2)
    os.replace(tmp, REGISTRY_PATH)

    print(f"registry.json updated — hybrid_stacking.metrics = {metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
