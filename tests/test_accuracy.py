"""
Full-accuracy guarantees: the LLM helper used by claim extraction and
translation, the no-key fallback, and the /health/accuracy self-check.
"""
import sys
import types

import pytest


class _Block:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Msg:
    def __init__(self, stop_reason, blocks):
        self.stop_reason, self.content, self.stop_details = stop_reason, blocks, None


def _fake_anthropic(monkeypatch, msg, seen=None):
    class _Stream:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get_final_message(self):
            return msg

    class _Messages:
        def stream(self, **kw):
            if seen is not None:
                seen.append(kw)
            return _Stream()

    class _Client:
        def __init__(self, **kw):
            self.beta = types.SimpleNamespace(messages=_Messages())
            self.messages = _Messages()

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=_Client))
    from config import settings
    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setattr(settings, "LLM_PROVIDER", "anthropic")


def test_helper_reads_text_after_a_thinking_block(monkeypatch):
    from paper_trail_core.nlp.llm import complete_json
    seen = []
    _fake_anthropic(monkeypatch, _Msg("end_turn", [
        _Block(type="thinking", thinking=""),
        _Block(type="text", text='Here you go:\n```json\n{"claims": ["A is B."]}\n```'),
    ]), seen)
    assert complete_json("prompt") == {"claims": ["A is B."]}
    assert seen[0]["max_tokens"] >= 16000          # room for thinking + answer


@pytest.mark.parametrize("stop", ["refusal", "max_tokens"])
def test_helper_rejects_refusals_and_truncation(monkeypatch, stop):
    from paper_trail_core.nlp.llm import LLMError, complete_json
    _fake_anthropic(monkeypatch, _Msg(stop, [_Block(type="text", text='{"claims": ["partial')]))
    with pytest.raises(LLMError):
        complete_json("prompt")


def test_llm_claim_extraction_uses_helper_and_sees_whole_article(monkeypatch):
    from paper_trail_core.agents import claim_extractor as ce
    seen = []
    long_tail = "Filler sentence about nothing in particular. " * 150
    article = long_tail + "The Reserve Bank of India raised the repo rate to 6.5 percent in February 2023."
    _fake_anthropic(monkeypatch, _Msg("end_turn", [_Block(type="text", text=
        '{"claims": ["The Reserve Bank of India raised the repo rate to 6.5 percent in February 2023."]}')]), seen)
    monkeypatch.setattr(ce, "_BACKEND", "llm")
    monkeypatch.setattr(ce, "_reject_hallucinations", lambda text, claims: claims)
    out = ce._extract_claims_impl({"article_text": article})
    assert out["claims"] == ["The Reserve Bank of India raised the repo rate to 6.5 percent in February 2023."]
    assert out["degradations"] == []
    # the claim sits past character 4000 — the old cap would never have shown it to the model
    assert len(article) > 4000 and "repo rate" in seen[0]["messages"][0]["content"]


def test_llm_backend_without_key_falls_back_and_says_so(monkeypatch):
    from config import settings
    from paper_trail_core.agents import claim_extractor as ce
    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", None)
    monkeypatch.setattr(ce, "_BACKEND", "llm")
    text = ("The Reserve Bank of India raised the repo rate to 6.5 percent in February 2023. "
            "Inflation in India fell to 4.25 percent in May 2023 according to official data.")
    out = ce._extract_claims_impl({"article_text": text})
    assert out["claims"]
    assert [d["kind"] for d in out["degradations"]] == ["llm_not_configured"]


def test_accuracy_report_flags_missing_key(monkeypatch):
    from api.accuracy import accuracy_report
    from config import settings
    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", None)
    monkeypatch.setattr(settings, "CLAIM_BACKEND", "llm")
    report = accuracy_report(deep=False)
    by = {s["stage"]: s for s in report["stages"]}
    assert report["full_accuracy"] is False
    assert by["claim_extraction"]["full_accuracy"] is False
    assert "ANTHROPIC_API_KEY" in by["claim_extraction"]["fix"]
    assert by["chat_replies"]["running"] == "rule-based"
    # never leaks secrets
    assert "sk-" not in str(report)


def test_accuracy_report_flags_degraded_backends(monkeypatch):
    from api.accuracy import accuracy_report
    from config import settings
    monkeypatch.setattr(settings, "NLI_BACKEND", "lexical")
    monkeypatch.setattr(settings, "EVIDENCE_BACKEND", "mock")
    by = {s["stage"]: s for s in accuracy_report()["stages"]}
    assert by["nli"]["full_accuracy"] is False and "NLI_BACKEND=transformer" in by["nli"]["fix"]
    assert by["evidence_retrieval"]["full_accuracy"] is False
