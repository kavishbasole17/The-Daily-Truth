"""
Read-only queries against models/registry.json.

Person D or any other consumer can call these helpers to inspect the model registry
without needing to understand the JSON schema.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Optional

_ROOT       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_REGISTRY   = os.path.join(_ROOT, "models", "registry.json")
_RESULTS    = os.path.join(_ROOT, "results", "all_results.csv")


@dataclass
class ModelEntry:
    name:         str
    display_name: str
    category:     str          # Classical ML | Deep Learning | Hybrid
    feature_type: str          # tfidf | dense | sequence | hybrid
    loaded:       bool
    path:         str
    train_time:   Optional[float]
    accuracy:     Optional[float]
    f1:           Optional[float]
    precision:    Optional[float]
    recall:       Optional[float]
    auc:          Optional[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name":         self.name,
            "display_name": self.display_name,
            "category":     self.category,
            "feature_type": self.feature_type,
            "path":         self.path,
            "train_time":   self.train_time,
            "metrics": {
                "accuracy":  self.accuracy,
                "f1":        self.f1,
                "precision": self.precision,
                "recall":    self.recall,
                "auc":       self.auc,
            },
        }


def _load_registry() -> dict:
    if not os.path.exists(_REGISTRY):
        return {}
    with open(_REGISTRY, encoding="utf-8") as fh:
        return json.load(fh)


def _entry_from_raw(name: str, raw: dict) -> ModelEntry:
    m = raw.get("metrics") or {}
    path = raw.get("path", "")
    exists = os.path.exists(os.path.join(_ROOT, path)) if path else False
    return ModelEntry(
        name=name,
        display_name=raw.get("display_name", name),
        category=raw.get("category", "Unknown"),
        feature_type=raw.get("feature_type", ""),
        loaded=exists,
        path=path,
        train_time=raw.get("train_time"),
        accuracy=m.get("accuracy"),
        f1=m.get("f1"),
        precision=m.get("precision"),
        recall=m.get("recall"),
        auc=m.get("auc"),
    )


def get_all_models() -> list[ModelEntry]:
    """Return every model recorded in the registry, in insertion order."""
    raw = _load_registry()
    return [_entry_from_raw(name, info) for name, info in raw.items()]


def get_top_models(
    n: int = 5,
    metric: str = "f1",
    category: Optional[str] = None,
    feature_type: Optional[str] = None,
) -> list[ModelEntry]:
    """
    Return the top-n models sorted descending by *metric*.

    Args:
        n:            Number of models to return.
        metric:       Sort key — "accuracy" | "f1" | "precision" | "recall" | "auc".
        category:     Optional filter: "Classical ML" | "Deep Learning" | "Hybrid".
        feature_type: Optional filter: "tfidf" | "dense" | "sequence" | "hybrid".

    Example:
        get_top_models(5, "f1")                       # top 5 by F1
        get_top_models(3, "auc", category="Classical ML")
    """
    entries = get_all_models()
    if category:
        entries = [e for e in entries if e.category == category]
    if feature_type:
        entries = [e for e in entries if e.feature_type == feature_type]

    def _key(e: ModelEntry) -> float:
        return getattr(e, metric) or 0.0

    return sorted(entries, key=_key, reverse=True)[:n]
