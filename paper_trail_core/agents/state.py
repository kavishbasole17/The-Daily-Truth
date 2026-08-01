"""LangGraph shared state TypedDict for the verification pipeline."""
from __future__ import annotations

from typing import Any, Optional

from typing_extensions import TypedDict


class ClaimVerdict(TypedDict):
    claim: str
    verdict: str            # entailment | contradiction | neutral
    confidence: float
    evidence_snippets: list[str]


class PipelineState(TypedDict):
    article_text: str
    url: Optional[str]

    # Populated by claim_extractor
    claims: list[str]

    # Populated by evidence_retriever: [{claim, snippets}]
    evidence: list[dict[str, Any]]

    # Populated by verifier
    claim_verdicts: list[ClaimVerdict]

    # Populated by ensemble_agent
    ensemble_score: float

    # Populated by adjudicator
    final_verdict: str              # likely_fake | likely_real | uncertain
    final_confidence: float         # confidence in the verdict (0-1)
    reasoning_trace: str
    needs_human_review: bool

    error: Optional[str]
