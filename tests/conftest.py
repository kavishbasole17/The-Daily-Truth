"""
Shared pytest configuration.

Two jobs: make the repo root importable regardless of how pytest is invoked,
and pin the pipeline to its offline, deterministic backends so the suite never
depends on a GPU, a downloaded model, or a network connection.

Individual tests opt back in to real components by monkeypatching, which keeps
"what this test actually exercises" explicit at the test site rather than
implicit in the environment.
"""
from __future__ import annotations

import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


@pytest.fixture(autouse=True)
def _isolate_fusion_singleton():
    """
    Reset the fusion model singleton around every test.

    It is process-global and lazily loaded, so without this a test that trains
    or stubs a fusion model would leak into every test that ran after it — and
    the resulting failures would depend on test ordering.
    """
    from paper_trail_core.agents import fusion

    saved_payload, saved_loaded = fusion._payload, fusion._loaded
    yield
    fusion._payload, fusion._loaded = saved_payload, saved_loaded


@pytest.fixture
def offline_settings(monkeypatch):
    """Force every stage onto a backend that needs no network and no model."""
    from config import settings

    monkeypatch.setattr(settings, "EVIDENCE_BACKEND", "cache_only")
    monkeypatch.setattr(settings, "NLI_BACKEND", "lexical")
    monkeypatch.setattr(settings, "ENSEMBLE_BACKEND", "mock")
    monkeypatch.setattr(settings, "WARM_MODELS", False)
    return settings
