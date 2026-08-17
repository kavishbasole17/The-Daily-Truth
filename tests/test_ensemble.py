"""
Unit tests for the stacking-ensemble agent.

These tests verify:
  1. Mock mode returns a valid probability.
  2. The stylistic ensemble agent node returns a float in [0, 1].
  3. The node handles a missing registry gracefully (fallback to 0.5).

Run with: pytest tests/test_ensemble.py -v
"""
import pytest
from unittest.mock import MagicMock, patch

# Force mock mode so no models need to be loaded from disk
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Import the node under test ─────────────────────────────────────────────────

from paper_trail_core.agents.state import PipelineState


def _make_state(**kwargs) -> PipelineState:
    defaults: PipelineState = {
        "article_text": "Scientists confirm new discovery in renewable energy sector.",
        "url": None,
        "claims": [],
        "evidence": [],
        "claim_verdicts": [],
        "ensemble_score": 0.5,
        "final_verdict": "uncertain",
        "final_confidence": 0.0,
        "reasoning_trace": "",
        "needs_human_review": False,
        "error": None,
    }
    defaults.update(kwargs)
    return defaults


# ── Mock-mode tests ────────────────────────────────────────────────────────────

def test_mock_mode_returns_valid_probability():
    with patch("paper_trail_core.agents.stylistic_ensemble_agent._MOCK", True):
        from paper_trail_core.agents.stylistic_ensemble_agent import run_ensemble_node
        state = _make_state()
        result = run_ensemble_node(state)
        score = result["ensemble_score"]
        assert isinstance(score, float), "ensemble_score must be a float"
        assert 0.0 <= score <= 1.0, f"ensemble_score out of range: {score}"


def test_mock_mode_returns_neutral_baseline():
    with patch("paper_trail_core.agents.stylistic_ensemble_agent._MOCK", True):
        from paper_trail_core.agents.stylistic_ensemble_agent import run_ensemble_node
        # Mock returns exactly 0.5 (neutral baseline)
        result = run_ensemble_node(_make_state())
        assert result["ensemble_score"] == 0.5


# ── Real-mode fallback tests ───────────────────────────────────────────────────

def test_real_mode_fallback_when_registry_unavailable():
    """When the registry can't be loaded (no trained models), should return 0.5."""
    with patch("paper_trail_core.agents.stylistic_ensemble_agent._MOCK", False), \
         patch("paper_trail_core.agents.stylistic_ensemble_agent._registry", None), \
         patch("paper_trail_core.agents.stylistic_ensemble_agent._get_registry",
               return_value=None):
        from paper_trail_core.agents.stylistic_ensemble_agent import run_ensemble_node
        result = run_ensemble_node(_make_state())
        assert result["ensemble_score"] == 0.5


def test_real_mode_uses_registry_when_available():
    mock_registry = MagicMock()
    mock_registry.hybrid_count = 1
    mock_registry.predict.return_value = {"prob_real": 0.73}

    with patch("paper_trail_core.agents.stylistic_ensemble_agent._MOCK", False), \
         patch("paper_trail_core.agents.stylistic_ensemble_agent._registry", mock_registry), \
         patch("paper_trail_core.agents.stylistic_ensemble_agent._get_registry",
               return_value=mock_registry):
        from paper_trail_core.agents.stylistic_ensemble_agent import run_ensemble_node
        result = run_ensemble_node(_make_state())
        assert result["ensemble_score"] == pytest.approx(0.73, abs=1e-4)


def test_uses_the_public_predict_api():
    """
    Guards against reaching back into `_predict_hybrid`.

    The agent used to call that private method because models/registry.json was
    missing its `hybrid_stacking` entry, so the public `predict()` raised
    KeyError. The registry is fixed; this keeps the workaround from creeping
    back and silently coupling the agent to ModelRegistry internals.
    """
    mock_registry = MagicMock()
    mock_registry.hybrid_count = 1
    mock_registry.predict.return_value = {"prob_real": 0.61}

    with patch("paper_trail_core.agents.stylistic_ensemble_agent._MOCK", False), \
         patch("paper_trail_core.agents.stylistic_ensemble_agent._registry", mock_registry), \
         patch("paper_trail_core.agents.stylistic_ensemble_agent._get_registry",
               return_value=mock_registry):
        from paper_trail_core.agents.stylistic_ensemble_agent import run_ensemble_node
        run_ensemble_node(_make_state())

    mock_registry.predict.assert_called_once()
    mock_registry._predict_hybrid.assert_not_called()
    # predict() cleans the text itself, so passing pre-cleaned text would
    # double-clean it. The raw article must go through untouched.
    assert mock_registry.predict.call_args.args[1] == "hybrid_stacking"


def test_real_mode_exception_falls_back_to_half():
    """Exception during prediction should not crash — returns 0.5 plus a degradation."""
    mock_registry = MagicMock()
    mock_registry.hybrid_count = 1
    mock_registry.predict.side_effect = RuntimeError("model corrupt")

    with patch("paper_trail_core.agents.stylistic_ensemble_agent._MOCK", False), \
         patch("paper_trail_core.agents.stylistic_ensemble_agent._registry", mock_registry), \
         patch("paper_trail_core.agents.stylistic_ensemble_agent._get_registry",
               return_value=mock_registry):
        from paper_trail_core.agents.stylistic_ensemble_agent import run_ensemble_node
        result = run_ensemble_node(_make_state())
        assert result["ensemble_score"] == 0.5
        assert any(d["kind"] == "predict_failed" for d in result["degradations"])
