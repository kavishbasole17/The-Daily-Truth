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

import contextlib
import os
import shutil
import sys
import tempfile

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# The suite must never touch the real database. config.py reads DATABASE_URL
# when it is first imported (an environment variable beats .env), and
# api/database.py builds its engine from that at import, so this has to be set
# here, before any test module imports either. Patching os.environ later, in a
# fixture, is too late: the tests then ran against the developer's
# paper_trail.db, leaving hundreds of test accounts in it and clearing its
# guest-usage table on every run.
_TEST_DB_DIR = tempfile.mkdtemp(prefix="paper_trail_tests_")
os.environ["DATABASE_URL"] = f"sqlite:///{os.path.join(_TEST_DB_DIR, 'test.db')}"


def pytest_sessionfinish(session, exitstatus):
    with contextlib.suppress(Exception):   # nothing imported the API: no engine to close
        from api.database import engine
        engine.dispose()
    shutil.rmtree(_TEST_DB_DIR, ignore_errors=True)


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


@pytest.fixture(autouse=True)
def _no_live_geocoding(monkeypatch, tmp_path):
    """
    Keep the locator off the network: gazetteer only, and a throwaway geocode
    cache so a developer's real cache can never make a test pass or fail.
    Tests that exercise the geocoder patch it explicitly.
    """
    from config import settings

    monkeypatch.setattr(settings, "GEOCODER_BACKEND", "offline")
    monkeypatch.setattr(settings, "GEOCODER_CACHE_DIR", str(tmp_path / "geo"))


@pytest.fixture(autouse=True)
def _no_wider_origin_search(monkeypatch):
    """
    Keep the origin tracer's wider search (Google News) off the network. Tests
    that exercise it stub the search explicitly.
    """
    from config import settings

    monkeypatch.setattr(settings, "ORIGIN_WIDE_SEARCH", False, raising=False)


@pytest.fixture(autouse=True)
def _article_path_by_default(monkeypatch):
    """
    Most tests exercise the article path (style classifier + learned fusion), and
    their sample texts happen to be under the claim-mode length limit. Claim mode
    is switched off here and enabled explicitly by the tests that cover it.
    """
    from config import settings

    monkeypatch.setattr(settings, "CLAIM_MODE", False)


@pytest.fixture(autouse=True)
def _no_llm_judge(monkeypatch):
    """
    The independent LLM fact-check needs an API key and the network. It is off
    for the suite and switched on explicitly by tests/test_llm_judge.py, which
    stubs the call.
    """
    from config import settings

    monkeypatch.setattr(settings, "LLM_JUDGE", False, raising=False)
    # The judge tests stub the Anthropic call; a local .env with
    # LLM_PROVIDER=nvidia must not turn them into live network calls.
    monkeypatch.setattr(settings, "LLM_PROVIDER", "anthropic", raising=False)
    monkeypatch.setattr(settings, "LLM_JUDGE_PROVIDER", "", raising=False)


@pytest.fixture(autouse=True)
def _forget_rejected_keys():
    """
    The retriever remembers, for the life of the process, any provider that
    rejected its API key. Clear that around every test so one test's simulated
    401 cannot switch a provider off for the tests that follow.
    """
    from paper_trail_core.agents import evidence_retriever as er

    er._rejected_keys.clear()
    yield
    er._rejected_keys.clear()


@pytest.fixture
def offline_settings(monkeypatch):
    """Force every stage onto a backend that needs no network and no model."""
    from config import settings

    monkeypatch.setattr(settings, "EVIDENCE_BACKEND", "cache_only")
    monkeypatch.setattr(settings, "NLI_BACKEND", "lexical")
    monkeypatch.setattr(settings, "ENSEMBLE_BACKEND", "mock")
    monkeypatch.setattr(settings, "WARM_MODELS", False)
    return settings
