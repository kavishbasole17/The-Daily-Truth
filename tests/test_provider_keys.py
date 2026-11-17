"""
A search key that is set but wrong (found in a live run on 2026-11-17: both the
NewsAPI and the Tavily key were answered 401, every stage reported green, and
the NewsAPI key was written to the server log inside the failing URL).
"""
from __future__ import annotations

import types

import pytest

from paper_trail_core.agents import evidence_retriever as er
from paper_trail_core.agents import llm_judge


class _Unauthorized(Exception):
    def __init__(self):
        super().__init__("401 Client Error: Unauthorized for url: "
                         "https://newsapi.org/v2/everything?q=x&apiKey=SECRET123&pageSize=5")
        self.response = types.SimpleNamespace(status_code=401)


def test_keys_are_redacted_from_log_text():
    out = er.redact(_Unauthorized())
    assert "SECRET123" not in out and "apiKey=<redacted>" in out
    assert "SECRET" not in er.redact("https://serpapi.com/search?q=a&api_key=SECRET&num=5")


def test_newsapi_key_is_sent_in_a_header_not_the_url(monkeypatch):
    import requests

    from config import settings

    monkeypatch.setattr(settings, "NEWS_API_KEY", "SECRET123")
    seen = {}

    def fake_get(url, params=None, headers=None, timeout=None):
        seen.update(params=params, headers=headers)
        return types.SimpleNamespace(status_code=200, raise_for_status=lambda: None,
                                     json=lambda: {"status": "ok", "articles": []})

    monkeypatch.setattr(requests, "get", fake_get)
    er._search_newsapi("eiffel tower")
    assert "SECRET123" not in str(seen["params"])
    assert seen["headers"] == {"X-Api-Key": "SECRET123"}


def test_a_rejected_key_is_recorded_once_and_the_provider_skipped(monkeypatch, tmp_path):
    from paper_trail_core.agents import evidence_cache

    monkeypatch.setattr(er, "_BACKEND", "live")
    monkeypatch.setattr(evidence_cache, "get", lambda *a, **k: None)
    monkeypatch.setattr(evidence_cache, "put", lambda *a, **k: None)
    calls = []

    def rejecting(query):
        calls.append(query)
        raise _Unauthorized()

    monkeypatch.setitem(er._PROVIDER_MAP, "newsapi", rejecting)

    assert er._search_cached("newsapi", "first") == ([], "error", False)
    assert calls == ["first"]                      # no retry: the key will not get better
    assert er.rejected_providers() == {"newsapi": "NEWS_API_KEY"}
    assert er._search_cached("newsapi", "second") == ([], "error", False)
    assert calls == ["first"]                      # ...and no further round trips


def test_a_transient_failure_is_not_a_rejected_key(monkeypatch):
    from paper_trail_core.agents import evidence_cache

    monkeypatch.setattr(er, "_BACKEND", "live")
    monkeypatch.setattr(evidence_cache, "get", lambda *a, **k: None)
    monkeypatch.setattr(er.time, "sleep", lambda s: None)
    calls = []

    def flaky(query):
        calls.append(query)
        raise ConnectionError("connection reset")

    monkeypatch.setitem(er._PROVIDER_MAP, "newsapi", flaky)
    assert er._search_cached("newsapi", "q")[1] == "error"
    assert len(calls) == 2 and er.rejected_providers() == {}


def test_accuracy_report_names_a_rejected_key(monkeypatch):
    from api.accuracy import accuracy_report
    from config import settings

    monkeypatch.setattr(settings, "NEWS_API_KEY", "set-but-wrong")
    er._note_rejected("newsapi")
    stage = {s["stage"]: s for s in accuracy_report(deep=False)["stages"]}["evidence_retrieval"]
    assert stage["full_accuracy"] is False
    assert "KEY REJECTED" in stage["running"] and "NEWS_API_KEY" in stage["fix"]
    assert "set-but-wrong" not in str(stage)


def test_judge_search_falls_back_to_headlines_when_tavily_rejects_the_key(monkeypatch):
    import sys

    from config import settings

    monkeypatch.setattr(settings, "TAVILY_API_KEY", "tvly-wrong")
    tavily_calls = []

    class FakeTavily:
        def __init__(self, api_key):
            pass

        def search(self, q, max_results=4):
            tavily_calls.append(q)
            raise RuntimeError("Unauthorized: missing or invalid API key.")

    monkeypatch.setitem(sys.modules, "tavily", types.SimpleNamespace(TavilyClient=FakeTavily))
    monkeypatch.setattr(er, "_search_gnews_rss", lambda q, cc: [
        {"url": "https://www.reuters.com/a", "text": f"Headline about {q}", "domain": "reuters.com",
         "published_at": "2026-11-01"}])

    results = llm_judge._web_search(["eiffel tower height", "eiffel tower 1889"])
    assert [r["url"] for r in results] == ["https://www.reuters.com/a"]
    assert tavily_calls == ["eiffel tower height"]          # stopped at the first rejection
    assert "tavily" in er.rejected_providers()
    # later checks go straight to headlines
    llm_judge._web_search(["again"])
    assert tavily_calls == ["eiffel tower height"]


@pytest.mark.parametrize("pref,cuda,mps,expected", [
    ("auto", True, True, "cuda"), ("auto", False, True, "mps"), ("auto", False, False, "cpu"),
    ("cpu", True, True, "cpu"), ("mps", False, False, "mps"),
])
def test_nli_device_choice(monkeypatch, pref, cuda, mps, expected):
    import torch

    from paper_trail_core.nlp import nli

    monkeypatch.setattr(nli, "_DEVICE_PREF", pref)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert nli._resolve_device() == expected
