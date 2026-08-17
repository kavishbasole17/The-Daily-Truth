"""
Stylistic / ensemble agent node.

Calls the stacking-ensemble (Hybrid model) trained by export_models.py to obtain
a model-based fake/real probability for the article text.

Mock mode  → returns 0.5 (uncertain — neutral baseline).
Real mode  → loads the ModelRegistry from api/model_loader and runs predict_hybrid.
             Falls back gracefully if models are not yet trained.
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
    _MOCK = settings.ENSEMBLE_BACKEND == "mock"
except ImportError:
    _MOCK = False

from paper_trail_core.agents.state import PipelineState
from paper_trail_core.degradations import degrade

log = logging.getLogger(__name__)

# Lazy-loaded registry singleton — loaded once and reused across pipeline calls
_registry = None


def _get_registry():
    global _registry
    if _registry is not None:
        return _registry
    try:
        from api.model_loader import ModelRegistry
        r = ModelRegistry()
        r.load_all()
        _registry = r
        log.info("ensemble_agent: ModelRegistry loaded (%d models)", r.total_loaded)
    except Exception as exc:
        log.warning("ensemble_agent: could not load ModelRegistry: %s", exc)
        _registry = None
    return _registry


def run_ensemble_node(state: PipelineState) -> dict:
    """LangGraph node: get a model-based fake/real probability."""
    text = state.get("article_text", "")
    log.info("ensemble_agent: running (mock=%s)", _MOCK)

    if _MOCK:
        # Neutral baseline — adjudicator will refine via the evidence signal.
        return {"ensemble_score": 0.5, "degradations": []}

    registry = _get_registry()
    if registry is None or registry.hybrid_count == 0:
        log.warning("ensemble_agent: hybrid model unavailable, using 0.5 fallback")
        return {
            "ensemble_score": 0.5,
            "degradations": degrade(
                "ensemble", "model_missing",
                "The stacking ensemble could not be loaded; its contribution was "
                "replaced with a neutral 0.5. Run export_models.py to train it.",
            ),
        }

    try:
        # `predict` applies clean_text itself (api/model_loader.py), so the text
        # is passed raw — pre-cleaning here would clean it twice.
        result = registry.predict(text, "hybrid_stacking", threshold=0.5)
        score = round(result["prob_real"], 4)
        log.info("ensemble_agent: ensemble_score=%.4f", score)
        return {"ensemble_score": score, "degradations": []}
    except Exception as exc:
        log.error("ensemble_agent: prediction failed (%s), falling back to 0.5", exc)
        return {
            "ensemble_score": 0.5,
            "degradations": degrade(
                "ensemble", "predict_failed",
                f"Ensemble prediction failed ({type(exc).__name__}); used a "
                "neutral 0.5 instead.",
            ),
        }
