"""
Claim-verification agent node — Natural Language Inference (NLI).

For each (claim, evidence snippet) pair, classifies the relationship as:
  entailment   — the evidence supports the claim
  contradiction — the evidence contradicts the claim
  neutral      — the evidence is unrelated / inconclusive

Mock mode  → heuristic keyword-based classification (no model download).
Real mode  → facebook/bart-large-mnli via HuggingFace Transformers.
             Model is lazy-loaded and cached as a module-level singleton (~1.6 GB).
"""
from __future__ import annotations

import logging
import os
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from config import settings
    _MOCK = settings.MOCK_MODE
except ImportError:
    _MOCK = True

from paper_trail_core.agents.evidence_retriever import UNCORROBORATED_MARKER
from paper_trail_core.agents.state import ClaimVerdict, PipelineState

log = logging.getLogger(__name__)

# A claim with zero coverage from any trusted outlet or reference source is
# treated as suspicious rather than neutral — mirrors how a human fact-checker
# would react to being unable to corroborate a claim anywhere credible.
_UNCORROBORATED_VERDICT = ("contradiction", 0.6)

_NLI_MODEL_NAME = "facebook/bart-large-mnli"
_nli_tokenizer = None
_nli_model     = None


def _load_nli():
    global _nli_tokenizer, _nli_model
    if _nli_tokenizer is not None:
        return
    log.info("verifier: loading NLI model %s (first call — may take a moment)", _NLI_MODEL_NAME)
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    _nli_tokenizer = AutoTokenizer.from_pretrained(_NLI_MODEL_NAME)
    _nli_model = AutoModelForSequenceClassification.from_pretrained(_NLI_MODEL_NAME)
    _nli_model.eval()
    log.info("verifier: NLI model loaded")


def _nli_infer(premise: str, hypothesis: str) -> tuple[str, float]:
    """Returns (label, confidence) where label ∈ {entailment, contradiction, neutral}."""
    import torch
    import torch.nn.functional as F

    _load_nli()
    inputs = _nli_tokenizer(
        premise, hypothesis,
        truncation=True, max_length=512, return_tensors="pt",
    )
    with torch.no_grad():
        logits = _nli_model(**inputs).logits
    probs = F.softmax(logits, dim=-1)[0]
    # BART-MNLI label order: 0=contradiction, 1=neutral, 2=entailment
    label_map = {0: "contradiction", 1: "neutral", 2: "entailment"}
    idx = int(probs.argmax())
    return label_map[idx], float(probs[idx])


# ── Lexical-overlap heuristic (lightweight NLI baseline, no GPU/model needed) ──
#
# Real retrieved evidence (Wikipedia/NewsAPI prose) doesn't contain scripted
# trigger words, so classifying a snippet in isolation from generic keyword
# lists is unreliable. Instead we measure how much of the *claim's* content
# words actually appear in the snippet (a standard lexical-overlap baseline
# for textual entailment — see McCoy et al., HANS) and only then check for
# negation/contradiction cues, so "confirmed" in an unrelated snippet can't
# masquerade as support for a claim it never mentions.

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "and", "or", "but", "if", "of", "at", "by", "for", "with", "about",
    "against", "between", "into", "through", "during", "to", "from", "in",
    "on", "off", "over", "under", "again", "further", "then", "once",
    "this", "that", "these", "those", "has", "have", "had", "having",
    "do", "does", "did", "doing", "will", "would", "could", "should",
    "it", "its", "as", "than", "so", "not", "no",
}

_NEGATION_MARKERS = (
    "false", "incorrect", "misleading", "no evidence", "debunked", "denies",
    "denied", "doubt", "disputed", "inaccurate", "untrue", "fabricated",
    "hoax", "does not", "did not", "never happened", "no credible",
    "not supported", "contradicts", "refuted",
)


def _content_words(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def _mock_infer(claim: str, snippet: str) -> tuple[str, float]:
    claim_words = _content_words(claim)
    snippet_lower = snippet.lower()
    snippet_words = _content_words(snippet)

    overlap = len(claim_words & snippet_words) / max(len(claim_words), 1)
    has_negation = any(marker in snippet_lower for marker in _NEGATION_MARKERS)

    if overlap < 0.25:
        # Evidence doesn't substantially address this specific claim's content
        return "neutral", round(0.5 + 0.1 * (1 - overlap), 4)
    if has_negation:
        return "contradiction", round(min(0.55 + 0.35 * overlap, 0.9), 4)
    return "entailment", round(min(0.55 + 0.35 * overlap, 0.9), 4)


# ── Node ──────────────────────────────────────────────────────────────────────

def verify_claims_node(state: PipelineState) -> dict:
    """LangGraph node: run NLI on each (claim, evidence) pair."""
    evidence_list = state.get("evidence", [])
    log.info("verifier: %d evidence items, mock=%s", len(evidence_list), _MOCK)

    claim_verdicts: list[ClaimVerdict] = []

    for item in evidence_list:
        claim    = item.get("claim", "")
        snippets = item.get("snippets", [])

        if not snippets:
            claim_verdicts.append(
                ClaimVerdict(
                    claim=claim,
                    verdict="neutral",
                    confidence=0.5,
                    evidence_snippets=[],
                )
            )
            continue

        # Aggregate NLI across snippets — take the verdict with the highest confidence
        best_label, best_conf = "neutral", 0.0
        for snippet in snippets[:3]:
            if not snippet.strip():
                continue
            try:
                if snippet.startswith(UNCORROBORATED_MARKER):
                    label, conf = _UNCORROBORATED_VERDICT
                elif _MOCK:
                    label, conf = _mock_infer(claim, snippet)
                else:
                    label, conf = _nli_infer(snippet, claim)
                if conf > best_conf:
                    best_label, best_conf = label, conf
            except Exception as exc:
                log.error("verifier: NLI failed for snippet: %s", exc)
                label, conf = _mock_infer(claim, snippet)
                if conf > best_conf:
                    best_label, best_conf = label, conf

        claim_verdicts.append(
            ClaimVerdict(
                claim=claim,
                verdict=best_label,
                confidence=round(best_conf, 4),
                evidence_snippets=snippets[:3],
            )
        )

    log.info(
        "verifier: %d verdicts — entail=%d contra=%d neutral=%d",
        len(claim_verdicts),
        sum(1 for v in claim_verdicts if v["verdict"] == "entailment"),
        sum(1 for v in claim_verdicts if v["verdict"] == "contradiction"),
        sum(1 for v in claim_verdicts if v["verdict"] == "neutral"),
    )
    return {"claim_verdicts": claim_verdicts}
