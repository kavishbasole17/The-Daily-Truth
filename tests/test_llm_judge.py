"""
Independent LLM fact-check + blend tests.

The judge itself needs an API key and the network, so `judge_text` is stubbed
here; what is asserted is the blend arithmetic and guards
(paper_trail_core/agents/blend.py), the reply parsing, and the pipeline
contract: the judge runs concurrently, its result is blended and persisted,
and every way it can go missing becomes a degradation rather than a crash.
"""
from __future__ import annotations

import threading
import time

import pytest

from paper_trail_core.agents import evidence_retriever as ER
from paper_trail_core.agents import llm_judge as J
from paper_trail_core.agents import pipeline as P
from paper_trail_core.agents import verifier as V
from paper_trail_core.agents.blend import blend_verdicts
from paper_trail_core.nlp.llm import LLMError
from paper_trail_core.nlp.nli import NLIResult

ARTICLE = (
    "The World Health Organization declared the COVID-19 outbreak a pandemic on "
    "11 March 2020. Director-General Tedros Adhanom Ghebreyesus said the agency "
    "was deeply concerned by the alarming levels of spread. At the time of the "
    "announcement more than 118,000 cases had been reported in 114 countries."
)


def judge(p_real=0.9, confidence=0.9, verdict=None, sources=1, web=True):
    verdict = verdict or ("likely_real" if p_real >= 0.6 else "likely_fake" if p_real <= 0.4 else "uncertain")
    return {"verdict": verdict, "p_real": p_real, "confidence": confidence, "summary": "s",
            "claims": [], "sources": [{"url": f"https://x{i}.com", "title": "t"} for i in range(sources)],
            "model": "m", "used_web_search": web, "n_searches": sources, "elapsed_s": 1.0}


# ── Blend rules ───────────────────────────────────────────────────────────────

def test_no_judge_leaves_agentic_verdict_alone():
    out = blend_verdicts(0.82, "likely_real", None)
    assert out["verdict"] == "likely_real" and out["rule"] == "agentic_only"
    assert out["p_final"] == pytest.approx(0.82)


def test_weights_default_to_60_40():
    out = blend_verdicts(0.8, "likely_real", judge(p_real=0.9, confidence=1.0))
    assert out["w_llm"] == pytest.approx(0.6) and out["w_agentic"] == pytest.approx(0.4)
    assert out["p_final"] == pytest.approx(0.6 * 0.9 + 0.4 * 0.8)
    assert out["rule"] == "agreement" and out["verdict"] == "likely_real"


def test_agreement_on_fake_commits_fake():
    out = blend_verdicts(0.2, "likely_fake", judge(p_real=0.1, confidence=0.9))
    assert out["verdict"] == "likely_fake" and out["rule"] == "agreement"
    assert out["confidence"] == pytest.approx(1 - out["p_final"])


def test_disagreement_is_uncertain_not_an_average():
    """0.6 x 0.9 + 0.4 x 0.2 = 0.62 would read 'likely real'. It must not."""
    out = blend_verdicts(0.2, "likely_fake", judge(p_real=0.9, confidence=0.95))
    assert out["verdict"] == "uncertain" and out["rule"] == "disagreement"
    assert out["confidence"] == 0.0


def test_llm_alone_commits_when_confident_and_grounded():
    out = blend_verdicts(0.5, "uncertain", judge(p_real=0.92, confidence=0.9))
    assert out["verdict"] == "likely_real" and out["rule"] == "llm_led"


def test_llm_alone_is_blocked_by_low_confidence():
    out = blend_verdicts(0.5, "uncertain", judge(p_real=0.92, confidence=0.3))
    assert out["verdict"] == "uncertain" and out["rule"] == "llm_low_confidence"


def test_llm_alone_is_blocked_without_sources():
    out = blend_verdicts(0.5, "uncertain", judge(p_real=0.95, confidence=0.95, sources=0))
    assert out["verdict"] == "uncertain" and out["rule"] == "llm_ungrounded"


def test_no_sources_halves_llm_weight():
    with_src = blend_verdicts(0.5, "uncertain", judge(p_real=0.9, confidence=1.0, sources=1))
    without = blend_verdicts(0.5, "uncertain", judge(p_real=0.9, confidence=1.0, sources=0))
    assert without["w_llm"] < with_src["w_llm"]


def test_low_judge_confidence_lowers_its_weight():
    sure = blend_verdicts(0.5, "uncertain", judge(p_real=0.9, confidence=1.0))
    unsure = blend_verdicts(0.5, "uncertain", judge(p_real=0.9, confidence=0.2))
    assert unsure["w_llm"] < sure["w_llm"]


def test_degraded_nli_removes_pipeline_weight():
    out = blend_verdicts(0.9, "uncertain", judge(p_real=0.2, confidence=0.9), nli_degraded=True)
    assert out["w_agentic"] == 0.0 and out["p_final"] == pytest.approx(0.2)


def test_weights_are_normalised():
    out = blend_verdicts(0.7, "likely_real", judge(p_real=0.7, confidence=0.5), coverage=0.0)
    assert out["w_llm"] + out["w_agentic"] == pytest.approx(1.0)


def test_no_usable_signal_is_uncertain():
    out = blend_verdicts(0.9, "likely_real", judge(p_real=0.9, confidence=0.9), weight_llm=0.0,
                         nli_degraded=True)
    assert out["verdict"] == "uncertain" and out["rule"] == "no_usable_signal"


# ── Reply parsing ─────────────────────────────────────────────────────────────

def test_extract_json_takes_the_last_verdict_object_despite_braces_in_prose():
    raw = ('Reasoning with {braces} and an earlier {"verdict": "uncertain", "p_real": 0.5}.\n'
           'Final: {"verdict": "likely_fake", "p_real": 0.1, "confidence": 0.8, "summary": "x"}')
    assert J.extract_json(raw)["verdict"] == "likely_fake"


def test_extract_json_raises_without_a_verdict():
    with pytest.raises(LLMError):
        J.extract_json("no json here {\"foo\": 1}")


def _nvidia_reply(content):
    from types import SimpleNamespace as NS
    msg = NS(content=content, reasoning_content=None)
    return NS(choices=[NS(message=msg, finish_reason="stop")])


@pytest.mark.parametrize("reply", [
    # Reasoning that mentions a brace before the object: the old greedy
    # first-{-to-last-} parse raised JSONDecodeError on exactly this.
    'I will search for {person} and {place}.\n{"queries": ["vijay oath telangana", "telangana chief minister"]}',
    # Two objects: an example, then the answer.
    'Example: {"queries": ["x"]}\nAnswer: {"queries": ["vijay oath telangana", "telangana chief minister"]}',
    '```json\n{"queries": ["vijay oath telangana", "telangana chief minister"]}\n```',
])
def test_query_plan_parses_despite_braces_and_fences(monkeypatch, reply):
    from paper_trail_core.nlp import llm

    class _Client:
        class chat:
            class completions:
                @staticmethod
                def create(**_kw):
                    return _nvidia_reply(reply)

    monkeypatch.setattr(llm, "nvidia_client", lambda: _Client)
    plan = J._nvidia_json("m", "p", key="queries", max_tokens=100)
    assert plan["queries"] == ["vijay oath telangana", "telangana chief minister"]


def test_query_plan_without_json_is_an_llm_error_not_a_decode_crash(monkeypatch):
    from paper_trail_core.nlp import llm

    class _Client:
        class chat:
            class completions:
                @staticmethod
                def create(**_kw):
                    return _nvidia_reply("Sorry, I can't plan searches {for this}.")

    monkeypatch.setattr(llm, "nvidia_client", lambda: _Client)
    with pytest.raises(LLMError):
        J._nvidia_json("m", "p", key="queries", max_tokens=100)


def test_judgement_from_clamps_and_derives_verdict():
    j = J.judgement_from({"p_real": 1.7, "confidence": "high", "claims": [
        {"claim": "c", "assessment": "SUPPORTED", "urls": ["https://a.com", "javascript:x"]}]})
    assert j.p_real == 1.0 and j.confidence == 0.0
    assert j.verdict == "likely_real"
    assert j.claims[0]["assessment"] == "supported" and j.claims[0]["urls"] == ["https://a.com"]


def test_judgement_from_rejects_unknown_verdict_label():
    assert J.judgement_from({"verdict": "maybe", "p_real": 0.5}).verdict == "uncertain"


# ── Pipeline integration ──────────────────────────────────────────────────────

@pytest.fixture
def pipeline_stubs(monkeypatch):
    """Offline pipeline with supporting evidence, deterministic fusion, judge enabled."""
    from config import settings
    from paper_trail_core.agents import fusion

    def _retrieve(claim, idx, **_kw):
        return {"claim": claim, "status": "ok", "cache_hits": 1, "queries": 1,
                "snippets": [{"text": ARTICLE, "url": "https://reuters.com/x", "domain": "reuters.com",
                              "trust": 0.9, "relevance": 0.75, "provider": "test"}]}
    monkeypatch.setattr(ER, "retrieve_for_claim", _retrieve)
    monkeypatch.setattr(V, "is_available", lambda: True)
    monkeypatch.setattr(V, "infer_batch", lambda p, h: [NLIResult(0.9, 0.05, 0.05) for _ in p])
    monkeypatch.setattr(settings, "ENSEMBLE_BACKEND", "mock")
    monkeypatch.setattr(settings, "LLM_JUDGE", True)
    monkeypatch.setattr(settings, "LLM_JUDGE_TIMEOUT_S", 5.0)
    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "sk-ant-test-key")
    fusion._payload, fusion._loaded = None, True
    yield settings
    fusion._loaded = False


def test_judge_runs_concurrently_and_is_blended(monkeypatch, pipeline_stubs):
    seen = {}

    def fake_judge(text):
        seen["thread"] = threading.current_thread().name
        seen["text"] = text
        time.sleep(0.2)
        return J.judgement_from(judge(p_real=0.95, confidence=0.9), model="m", used_web_search=True,
                                n_searches=2, sources=[{"url": "https://who.int/x", "title": "WHO"}])
    monkeypatch.setattr(J, "judge_text", fake_judge)

    vr = P.run_verification_pipeline(ARTICLE)

    assert seen["text"] == ARTICLE
    assert seen["thread"] != threading.current_thread().name, "the judge must run in the background"
    assert vr.llm_judge and vr.llm_judge["blend"]["rule"] in ("agreement", "llm_led", "weighted")
    assert vr.llm_judge["sources"][0]["url"] == "https://who.int/x"
    assert vr.fusion_method.endswith("+llm_judge")
    assert vr.fused_score == pytest.approx(vr.llm_judge["blend"]["p_final"])
    assert "INDEPENDENT LLM CHECK" in vr.reasoning_trace and "FINAL BLEND" in vr.reasoning_trace
    assert vr.to_dict()["llm_judge"] is vr.llm_judge
    assert not any(d["stage"] == "llm_judge" for d in vr.degradations)


def test_disagreement_forces_uncertain_end_to_end(monkeypatch, pipeline_stubs):
    monkeypatch.setattr(J, "judge_text", lambda text: J.judgement_from(
        judge(p_real=0.05, confidence=0.95), sources=[{"url": "https://a.com", "title": "a"}],
        used_web_search=True))
    vr = P.run_verification_pipeline(ARTICLE)
    if vr.llm_judge["blend"]["agentic_verdict"] == "likely_real":
        assert vr.verdict == "uncertain"
        assert any(d["kind"] == "judges_disagree" for d in vr.degradations)


def test_missing_key_is_a_degradation_not_a_crash(monkeypatch, pipeline_stubs):
    monkeypatch.setattr(pipeline_stubs, "ANTHROPIC_API_KEY", None)
    vr = P.run_verification_pipeline(ARTICLE)
    assert vr.llm_judge is None
    assert any(d["kind"] == "llm_judge_not_configured" for d in vr.degradations)
    assert "llm_judge_not_configured" in vr.reasoning_trace


def test_judge_failure_is_a_degradation(monkeypatch, pipeline_stubs):
    def boom(text):
        raise LLMError("declined")
    monkeypatch.setattr(J, "judge_text", boom)
    vr = P.run_verification_pipeline(ARTICLE)
    assert vr.llm_judge is None
    assert any(d["kind"] == "llm_judge_failed" for d in vr.degradations)
    assert vr.verdict in ("likely_real", "likely_fake", "uncertain")


def test_judge_timeout_is_a_degradation(monkeypatch, pipeline_stubs):
    monkeypatch.setattr(pipeline_stubs, "LLM_JUDGE_TIMEOUT_S", 0.3)
    release = threading.Event()

    def slow(text):
        release.wait(3.0)
        return J.judgement_from(judge())
    monkeypatch.setattr(J, "judge_text", slow)
    try:
        vr = P.run_verification_pipeline(ARTICLE)
    finally:
        release.set()
    assert vr.llm_judge is None
    assert any(d["kind"] == "llm_judge_timeout" for d in vr.degradations)


def test_judge_off_adds_nothing(monkeypatch, pipeline_stubs):
    monkeypatch.setattr(pipeline_stubs, "LLM_JUDGE", False)
    called = []
    monkeypatch.setattr(J, "judge_text", lambda text: called.append(1))
    vr = P.run_verification_pipeline(ARTICLE)
    assert not called and vr.llm_judge is None
    assert not any(d["stage"] == "llm_judge" for d in vr.degradations)


# ── API schema round-trip ─────────────────────────────────────────────────────

def test_llm_judge_out_accepts_the_pipeline_record():
    from api.routers.articles import LLMJudgeOut

    rec = {**judge(), "blend": blend_verdicts(0.8, "likely_real", judge())}
    out = LLMJudgeOut(**rec)
    assert out.verdict == "likely_real" and out.blend["rule"] == "agreement"
    assert out.sources[0].url.startswith("https://")
