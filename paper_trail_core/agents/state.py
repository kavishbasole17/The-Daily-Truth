"""LangGraph shared state TypedDict for the verification pipeline."""
from __future__ import annotations

import operator
from typing import Annotated, Any, Optional

from typing_extensions import TypedDict

from paper_trail_core.degradations import Degradation


class ClaimVerdict(TypedDict, total=False):
    """
    Per-claim result.

    `claim`, `verdict`, `confidence` and `evidence_snippets` are the original
    four fields and are always populated — `frontend/src/app/verify/page.jsx`
    reads them directly, and `evidence_snippets` must stay `list[str]` because
    the UI renders each element as a text node.

    The remaining fields are additive diagnostics, declared optional
    (`total=False`) so verdict rows written by the previous version still
    deserialise.
    """
    claim: str
    verdict: str                    # entailment | contradiction | neutral | unverifiable
    confidence: float
    evidence_snippets: list[str]

    # Diagnostics — also the raw material for the fusion feature vector.
    support: float                  # entail_mass - contradict_mass, in [-1, 1]
    entail_mass: float              # in [0, 1]
    contradict_mass: float          # in [0, 1]
    top_relevance: float            # best claim<->snippet cosine after gating
    numeric_conflict: float         # in [0, 1]; see paper_trail_core/nlp/numeric.py
    sources: list[str]              # domains, positionally parallel to evidence_snippets
    status: str                     # ok | no_hits | no_relevant | rate_limited | error


class PipelineState(TypedDict):
    article_text: str
    url: Optional[str]

    # Input routing. "text" (default, today's only path) skips the OCR node
    # entirely; "image" enters through it. raw_image_ref is a filesystem path
    # to the uploaded image — kept as a path rather than raw bytes so it
    # doesn't get copied through every state dict update.
    input_type: str                     # "text" | "image"
    raw_image_ref: Optional[str]

    # Populated by ocr_agent (image input only; None/unset for text input)
    ocr_text: Optional[str]
    ocr_confidence: Optional[float]     # 0-1, mean confidence across detected blocks
    ocr_script: Optional[str]           # detected script, e.g. "Devanagari", "Latin"

    # Populated by claim_extractor
    claims: list[str]

    # Populated by evidence_retriever: [{claim, snippets, status}]
    evidence: list[dict[str, Any]]
    retrieval_stats: dict[str, Any]

    # Populated by verifier
    claim_verdicts: list[ClaimVerdict]
    nli_degraded: bool

    # Populated by ensemble_agent
    ensemble_score: float

    # Populated by adjudicator
    final_verdict: str              # likely_fake | likely_real | uncertain
    final_confidence: float         # confidence in the reported verdict (0-1)
    fused_score: float              # calibrated P(real) before thresholding
    fusion_features: dict[str, float]
    fusion_method: str              # learned | heuristic_fallback
    reasoning_trace: str
    needs_human_review: bool

    # Accumulated across every node. The reducer matters: LangGraph *overwrites*
    # state keys returned by a node unless one is declared, so without
    # `operator.add` each node would clobber the previous node's degradations
    # and only the last stage's would survive.
    degradations: Annotated[list[Degradation], operator.add]

    error: Optional[str]
