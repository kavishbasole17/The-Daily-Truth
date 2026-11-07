"""
LangGraph verification pipeline.

Graph:
    (entry, routed on input_type)
      ├─ "image" → ocr ─────┐  ← extract text from an image, confidence-gated
      │                     ↓
      └─ "text"  ────→ translate  ← detect language, translate to English if needed
                            ↓
                       extract_claims   ← segment + rank check-worthy claims
                            ↓
                       locate           ← where is each claim about? (place + precision)
                            ↓
                       retrieve_evidence ← entity queries + regional news, relevance-gated, cached
                            ↓
                       verify_claims    ← NLI per (evidence sentence, claim) pair
                            ↓
                       trace_origin     ← fact-checks + dated trail for unsupported claims (report-only)
                            ↓
                       run_ensemble     ← stacking ensemble P(real)
                            ↓
                       adjudicate       ← calibrated fusion → verdict or abstention
                            ↓
                           END

Every node runs behind a stage guard: a timeout and a defined fallback state.
A stage that hangs or throws degrades that stage only, records why, and lets
the rest of the pipeline finish — so a search provider going down produces an
honest "couldn't verify" instead of a 500 or, worse, a
confident verdict computed from nothing. A failed/low-confidence OCR read or
an untranslatable/unsupported-language input is routed straight to a
dedicated `abstain` terminal node instead of into extract_claims — see
`_abstain_node`'s docstring for why routing it through the normal
claims/ensemble/fusion path (which looked like a natural reuse of the
existing "no claims" abstention) was wrong: with no usable article text at
all, that path let the *trained* fusion model score an all-zero evidence
vector it was never fit on, which is a different failure mode than "a short
real article had nothing checkable."

Public API:
    run_verification_pipeline(article_text, url, region_hint) -> VerdictResult
    run_verification_pipeline_image(image_path, url, region_hint) -> VerdictResult
"""
from __future__ import annotations

import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Callable, Optional

from langgraph.graph import END, StateGraph

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from config import settings
    _STAGE_TIMEOUT = settings.STAGE_TIMEOUT_S
except ImportError:  # pragma: no cover
    _STAGE_TIMEOUT = 45.0

from paper_trail_core.agents.adjudicator import adjudicate_node
from paper_trail_core.agents.claim_extractor import extract_claims_node
from paper_trail_core.agents.evidence_retriever import retrieve_evidence_node
from paper_trail_core.agents.locator import locate_node
from paper_trail_core.agents.ocr_agent import ocr_node
from paper_trail_core.agents.origin_tracer import trace_origin_node
from paper_trail_core.agents.state import ClaimVerdict, PipelineState
from paper_trail_core.agents.stylistic_ensemble_agent import run_ensemble_node
from paper_trail_core.agents.translation_agent import translate_node
from paper_trail_core.agents.verifier import verify_claims_node
from paper_trail_core.degradations import Degradation, degrade, summarise

log = logging.getLogger(__name__)

MIN_ARTICLE_CHARS = 40
MAX_ARTICLE_CHARS = 50_000


# ── Stage guard ────────────────────────────────────────────────────────────────

def _fallback_ocr(_state: PipelineState) -> dict:
    return {
        "ocr_text": "", "ocr_confidence": 0.0, "ocr_script": "unknown",
        "article_text": "",
    }


def _fallback_translate(_state: PipelineState) -> dict:
    return {
        "detected_language": "unknown", "language_confidence": 0.0,
        "original_text": "", "translated_text": "", "translation_confidence": 0.0,
        "article_text": "",
    }


def _abstain_node(state: PipelineState) -> dict:
    """
    Terminal node for input that never became usable English article text —
    a failed/low-confidence OCR read, or an untranslated/unsupported-language
    submission. Deliberately does NOT proceed to
    extract_claims -> ... -> adjudicate.

    Early testing of the OCR path showed why: with claim_verdicts=[]
    (nothing to check) and article_text="", the adjudicator's "no claims ->
    fall back to the style ensemble" path (a real, tested contract — see
    tests/test_adjudicator.py::test_no_claims_falls_back_to_ensemble, which
    exists for a short-but-real article the style model can still score)
    ran the *trained fusion model* over an all-zero evidence feature vector
    it was never fit on. On a smoke test that returned a confident
    "likely_fake" for a file that simply didn't exist — worse than a
    coin flip, since it looked like a real, checked verdict. There is no
    article here for the ensemble or fusion model to score; this node says
    so directly instead of asking either to extrapolate from nothing. The
    same reasoning applies to a translation failure: an untranslated Hindi
    article isn't "no claims", it's "no English text to check", and deserves
    the same explicit stop rather than an English-tuned verdict on foreign text.
    """
    relevant = [
        d for d in state.get("degradations", [])
        if d.get("stage") in ("ocr", "translation") and d.get("severity") == "critical"
    ]
    detail = relevant[-1]["detail"] if relevant else "Input did not produce usable article text."
    return {
        "ensemble_score": 0.5,
        "final_verdict": "uncertain",
        "final_confidence": 0.0,
        "fused_score": 0.5,
        "fusion_features": {},
        "fusion_method": "not_run",
        "reasoning_trace": f"Could not verify: {detail}",
    }


def _fallback_claims(_state: PipelineState) -> dict:
    return {"claims": []}


def _fallback_locate(state: PipelineState) -> dict:
    return {"claim_locations": [None] * len(state.get("claims", [])), "article_location": None}


def _fallback_evidence(state: PipelineState) -> dict:
    return {
        "evidence": [{"claim": c, "snippets": [], "status": "error"}
                     for c in state.get("claims", [])],
        "retrieval_stats": {},
    }


def _fallback_verify(state: PipelineState) -> dict:
    return {
        "claim_verdicts": [
            ClaimVerdict(claim=item.get("claim", ""), verdict="unverifiable",
                         confidence=0.0, evidence_snippets=[], support=0.0,
                         entail_mass=0.0, contradict_mass=0.0, top_relevance=0.0,
                         numeric_conflict=0.0, sources=[], status="error")
            for item in state.get("evidence", [])
        ],
        "nli_degraded": True,
    }


def _fallback_origin(_state: PipelineState) -> dict:
    # Leaves claim_verdicts as the verifier wrote them: no trail, same verdict.
    return {}


def _fallback_ensemble(_state: PipelineState) -> dict:
    return {"ensemble_score": 0.5}


def _fallback_adjudicate(_state: PipelineState) -> dict:
    return {
        "final_verdict": "uncertain",
        "final_confidence": 0.0,
        "fused_score": 0.5,
        "fusion_features": {},
        "fusion_method": "failed",
        "reasoning_trace": "Adjudication failed; no verdict could be reached.",
    }


def with_stage_guard(
    fn: Callable[[PipelineState], dict],
    stage: str,
    fallback: Callable[[PipelineState], dict],
    timeout_s: float = _STAGE_TIMEOUT,
) -> Callable[[PipelineState], dict]:
    """
    Wrap a node so it can never hang the request or propagate an exception.

    Note on the timeout: Python cannot kill a running thread, so a timed-out
    stage keeps executing in the background until it finishes on its own. That
    is acceptable here because every stage is network- or GPU-bound and
    self-terminating — nothing spins forever — and the alternative (a request
    that blocks indefinitely) is worse.
    """
    def _wrapped(state: PipelineState) -> dict:
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(fn, state).result(timeout=timeout_s)
        except FutureTimeout:
            log.error("pipeline: stage %s timed out after %.0fs", stage, timeout_s)
            out = dict(fallback(state))
            out["degradations"] = degrade(
                stage, "timeout",
                f"The {stage} stage exceeded its {timeout_s:.0f}s budget and was "
                "abandoned. Its contribution is missing from this verdict.",
                severity="critical",
            )
            return out
        except Exception as exc:
            # Full exception detail goes to the server log only. The
            # degradation record is persisted to the DB and served back by
            # the unauthenticated GET /articles/{id} endpoint, so it must
            # never carry raw exception text — that can include internal
            # paths, provider error bodies, or (for the LLM claim-extraction
            # backend) fragments of a misconfigured API key.
            log.error("pipeline: stage %s failed — %s", stage, exc, exc_info=True)
            out = dict(fallback(state))
            out["degradations"] = degrade(
                stage, "stage_error",
                f"The {stage} stage failed unexpectedly ({type(exc).__name__}). "
                "Its contribution is missing from this verdict.",
                severity="critical",
            )
            return out
    return _wrapped


# ── Graph construction ─────────────────────────────────────────────────────────

def _route_on_input_type(state: PipelineState) -> str:
    return "image" if state.get("input_type") == "image" else "text"


def _route_after_ocr(state: PipelineState) -> str:
    return "proceed" if (state.get("article_text") or "").strip() else "abstain"


def _route_after_translation(state: PipelineState) -> str:
    return "proceed" if (state.get("article_text") or "").strip() else "abstain"


def _build_graph() -> StateGraph:
    g = StateGraph(PipelineState)

    g.add_node("ocr",
               with_stage_guard(ocr_node, "ocr", _fallback_ocr))
    g.add_node("translate",
               with_stage_guard(translate_node, "translation", _fallback_translate))
    g.add_node("abstain", _abstain_node)
    g.add_node("extract_claims",
               with_stage_guard(extract_claims_node, "claim_extractor", _fallback_claims))
    g.add_node("locate",
               with_stage_guard(locate_node, "locator", _fallback_locate))
    g.add_node("retrieve_evidence",
               with_stage_guard(retrieve_evidence_node, "evidence_retriever", _fallback_evidence))
    g.add_node("verify_claims",
               with_stage_guard(verify_claims_node, "verifier", _fallback_verify))
    g.add_node("trace_origin",
               with_stage_guard(trace_origin_node, "origin_tracer", _fallback_origin))
    g.add_node("run_ensemble",
               with_stage_guard(run_ensemble_node, "ensemble", _fallback_ensemble))
    g.add_node("adjudicate",
               with_stage_guard(adjudicate_node, "adjudicator", _fallback_adjudicate))

    g.set_conditional_entry_point(
        _route_on_input_type, {"image": "ocr", "text": "translate"}
    )
    g.add_conditional_edges(
        "ocr", _route_after_ocr, {"proceed": "translate", "abstain": "abstain"}
    )
    g.add_conditional_edges(
        "translate", _route_after_translation, {"proceed": "extract_claims", "abstain": "abstain"}
    )
    g.add_edge("abstain",           END)
    g.add_edge("extract_claims",    "locate")
    g.add_edge("locate",            "retrieve_evidence")
    g.add_edge("retrieve_evidence", "verify_claims")
    g.add_edge("verify_claims",     "trace_origin")
    g.add_edge("trace_origin",      "run_ensemble")
    g.add_edge("run_ensemble",      "adjudicate")
    g.add_edge("adjudicate",        END)

    return g


_app = None


def _get_app():
    global _app
    if _app is None:
        _app = _build_graph().compile()
    return _app


# ── Public return type ─────────────────────────────────────────────────────────

@dataclass
class VerdictResult:
    verdict: str                        # likely_fake | likely_real | uncertain
    confidence: float
    claim_verdicts: list[ClaimVerdict]
    ensemble_score: float
    reasoning_trace: str
    fused_score: float = 0.5
    fusion_method: str = "heuristic_fallback"
    fusion_features: dict = field(default_factory=dict)
    degradations: list[Degradation] = field(default_factory=list)
    retrieval_stats: dict = field(default_factory=dict)
    # Final English article text the pipeline actually verified — for a text
    # submission this just echoes the input, but for an image submission it's
    # the only place the OCR (and, if applicable, translated) transcription
    # comes back to the caller. Empty when OCR/translation abstained.
    article_text: str = ""

    def to_dict(self) -> dict:
        return {
            "verdict":            self.verdict,
            "confidence":         self.confidence,
            "claim_verdicts":     list(self.claim_verdicts),
            "ensemble_score":     self.ensemble_score,
            "reasoning_trace":    self.reasoning_trace,
            "fused_score":        self.fused_score,
            "fusion_method":      self.fusion_method,
            "fusion_features":    self.fusion_features,
            "degradations":       list(self.degradations),
            "retrieval_stats":    self.retrieval_stats,
            "article_text":       self.article_text,
        }

    def claim_verdicts_json(self) -> str:
        return json.dumps(list(self.claim_verdicts))

    def degradations_json(self) -> str:
        return json.dumps(list(self.degradations))


# ── Entry point ────────────────────────────────────────────────────────────────

def _guarded_result(verdict: str, trace: str, degradations: list[Degradation]) -> VerdictResult:
    return VerdictResult(
        verdict=verdict, confidence=0.0, claim_verdicts=[], ensemble_score=0.5,
        reasoning_trace=trace, fused_score=0.5,
        fusion_method="not_run", degradations=degradations,
    )


def _base_state(url: Optional[str], region_hint: Optional[str] = None) -> dict:
    """Fields common to every entry path, before input-type-specific ones are set."""
    return {
        "url":               url,
        "input_type":        "text",
        "raw_image_ref":     None,
        "ocr_text":          None,
        "ocr_confidence":    None,
        "ocr_script":        None,
        "detected_language": None,
        "language_confidence": None,
        "original_text":     None,
        "translated_text":   None,
        "translation_confidence": None,
        "region_hint":       region_hint,
        "claims":            [],
        "claim_locations":   [],
        "article_location":  None,
        "evidence":          [],
        "retrieval_stats":   {},
        "claim_verdicts":    [],
        "nli_degraded":      False,
        "ensemble_score":    0.5,
        "final_verdict":     "uncertain",
        "final_confidence":  0.0,
        "fused_score":       0.5,
        "fusion_features":   {},
        "fusion_method":     "heuristic_fallback",
        "reasoning_trace":   "",
        "degradations":      [],
        "error":             None,
    }


def _run_graph(initial_state: PipelineState) -> VerdictResult:
    """Shared invoke + result-building for every entry path."""
    try:
        result_state = _get_app().invoke(initial_state)
    except Exception as exc:
        # Stage guards mean this should be unreachable; if the graph machinery
        # itself fails we still owe the caller a well-formed answer. As above,
        # raw exception text never goes into a field the public, unauthenticated
        # GET /articles/{id} endpoint returns — only the server log gets it.
        log.error("pipeline: unexpected error — %s", exc, exc_info=True)
        return _guarded_result(
            "uncertain",
            "Verification could not be completed due to an internal error.",
            degrade("pipeline", "pipeline_error",
                    f"Unexpected error ({type(exc).__name__}).", severity="critical"),
        )

    degradations = result_state.get("degradations", [])
    if degradations:
        log.info("pipeline: completed with degradations %s", summarise(degradations))

    return VerdictResult(
        verdict=result_state.get("final_verdict", "uncertain"),
        confidence=result_state.get("final_confidence", 0.0),
        claim_verdicts=result_state.get("claim_verdicts", []),
        ensemble_score=result_state.get("ensemble_score", 0.5),
        reasoning_trace=result_state.get("reasoning_trace", ""),
        fused_score=result_state.get("fused_score", 0.5),
        fusion_method=result_state.get("fusion_method", "heuristic_fallback"),
        fusion_features=result_state.get("fusion_features", {}),
        degradations=degradations,
        retrieval_stats=result_state.get("retrieval_stats", {}),
        article_text=result_state.get("article_text", "") or "",
    )


def run_verification_pipeline(
    article_text: str,
    url: Optional[str] = None,
    region_hint: Optional[str] = None,
) -> VerdictResult:
    """
    Run the full multi-agent verification pipeline over plain text.

    Never raises and never returns a confident verdict it cannot justify: any
    failure path yields `uncertain` with a
    populated degradation list explaining what went wrong.
    """
    text = (article_text or "").strip()
    log.info("pipeline: starting (text_len=%d)", len(text))

    # Input guards — reject before spending model time, with a real explanation.
    min_chars = MIN_ARTICLE_CHARS
    try:
        if settings.CLAIM_MODE:
            min_chars = min(min_chars, int(settings.CLAIM_MODE_MIN_CHARS))
    except Exception:  # pragma: no cover
        pass
    if len(text) < min_chars:
        return _guarded_result(
            "uncertain",
            f"Article is too short to verify ({len(text)} characters; "
            f"{min_chars} required).",
            degrade("pipeline", "text_too_short",
                    f"Submitted text was {len(text)} characters.", severity="critical"),
        )
    if len(text) > MAX_ARTICLE_CHARS:
        log.warning("pipeline: truncating %d chars to %d", len(text), MAX_ARTICLE_CHARS)
        text = text[:MAX_ARTICLE_CHARS]

    initial_state: PipelineState = {**_base_state(url, region_hint), "article_text": text}
    return _run_graph(initial_state)


def run_verification_pipeline_image(
    image_path: str,
    url: Optional[str] = None,
    region_hint: Optional[str] = None,
) -> VerdictResult:
    """
    Run the full multi-agent verification pipeline over an image.

    article_text starts empty by design — the ocr node populates it (or
    leaves it empty and abstains, on a failed read or low-confidence
    transcription; see ocr_agent.py). Unlike the text entry point, there is
    no text-length guard here: a length check on an empty starting string
    would reject every image submission before OCR ever ran.
    """
    log.info("pipeline: starting (input_type=image)")

    if not image_path:
        return _guarded_result(
            "uncertain",
            "No image was provided.",
            degrade("pipeline", "no_image", "Image submission had no file.", severity="critical"),
        )

    initial_state: PipelineState = {
        **_base_state(url, region_hint),
        "article_text":   "",
        "input_type":     "image",
        "raw_image_ref":  image_path,
    }
    return _run_graph(initial_state)
