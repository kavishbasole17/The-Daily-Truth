"""
LangGraph verification pipeline.

Graph:
    extract_claims
         ↓
    retrieve_evidence
         ↓
    verify_claims      ← NLI per claim
         ↓
    run_ensemble       ← stacking ensemble P(real)
         ↓
    adjudicate         ← combine → final verdict
         ↓
       END

Public API:
    run_verification_pipeline(article_text, url) -> VerdictResult
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional

from langgraph.graph import END, StateGraph

from paper_trail_core.agents.claim_extractor import extract_claims_node
from paper_trail_core.agents.evidence_retriever import retrieve_evidence_node
from paper_trail_core.agents.verifier import verify_claims_node
from paper_trail_core.agents.stylistic_ensemble_agent import run_ensemble_node
from paper_trail_core.agents.adjudicator import adjudicate_node
from paper_trail_core.agents.state import ClaimVerdict, PipelineState

log = logging.getLogger(__name__)


# ── Graph construction ─────────────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    g = StateGraph(PipelineState)

    g.add_node("extract_claims",    extract_claims_node)
    g.add_node("retrieve_evidence", retrieve_evidence_node)
    g.add_node("verify_claims",     verify_claims_node)
    g.add_node("run_ensemble",      run_ensemble_node)
    g.add_node("adjudicate",        adjudicate_node)

    g.set_entry_point("extract_claims")
    g.add_edge("extract_claims",    "retrieve_evidence")
    g.add_edge("retrieve_evidence", "verify_claims")
    g.add_edge("verify_claims",     "run_ensemble")
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
    needs_human_review: bool

    def to_dict(self) -> dict:
        return {
            "verdict":            self.verdict,
            "confidence":         self.confidence,
            "claim_verdicts":     list(self.claim_verdicts),
            "ensemble_score":     self.ensemble_score,
            "reasoning_trace":    self.reasoning_trace,
            "needs_human_review": self.needs_human_review,
        }

    def claim_verdicts_json(self) -> str:
        return json.dumps(list(self.claim_verdicts))


# ── Entry point ────────────────────────────────────────────────────────────────

def run_verification_pipeline(
    article_text: str,
    url: Optional[str] = None,
) -> VerdictResult:
    """
    Run the full multi-agent verification pipeline.

    Args:
        article_text: Raw news article text.
        url:          Optional source URL (stored but not currently fetched).

    Returns:
        VerdictResult with verdict, confidence, claim-level breakdown, and reasoning.
    """
    log.info("pipeline: starting (text_len=%d)", len(article_text))

    initial_state: PipelineState = {
        "article_text":      article_text,
        "url":               url,
        "claims":            [],
        "evidence":          [],
        "claim_verdicts":    [],
        "ensemble_score":    0.5,
        "final_verdict":     "uncertain",
        "final_confidence":  0.0,
        "reasoning_trace":   "",
        "needs_human_review": False,
        "error":             None,
    }

    try:
        result_state = _get_app().invoke(initial_state)
    except Exception as exc:
        log.error("pipeline: unexpected error — %s", exc, exc_info=True)
        return VerdictResult(
            verdict="uncertain",
            confidence=0.0,
            claim_verdicts=[],
            ensemble_score=0.5,
            reasoning_trace=f"Pipeline error: {exc}",
            needs_human_review=True,
        )

    if result_state.get("error"):
        log.warning("pipeline: completed with error: %s", result_state["error"])

    return VerdictResult(
        verdict=result_state.get("final_verdict", "uncertain"),
        confidence=result_state.get("final_confidence", 0.0),
        claim_verdicts=result_state.get("claim_verdicts", []),
        ensemble_score=result_state.get("ensemble_score", 0.5),
        reasoning_trace=result_state.get("reasoning_trace", ""),
        needs_human_review=result_state.get("needs_human_review", True),
    )
