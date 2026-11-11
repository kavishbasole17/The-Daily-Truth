"""
Verifier tests — aggregation, NLI label handling, and numeric conflict.

Two defects are pinned here.

1. The old aggregation compared confidences *across labels*, keeping whichever
   snippet had the highest one. A confident `neutral` therefore beat a moderate
   `entailment`, and an article whose evidence was mostly off-topic came back
   neutral even when one snippet flatly refuted it.

2. Absence of evidence was converted into `contradiction` at 0.6 confidence,
   so anything the search providers missed looked fabricated.

Everything here runs offline — the NLI model is never loaded; results are
constructed directly.
"""
from __future__ import annotations

import pytest

from paper_trail_core.agents.verifier import (
    DECISION_FLOOR,
    aggregate_claim,
    is_informative,
    lexical_infer,
    scale_relevance,
)
from paper_trail_core.nlp.nli import NLIResult, build_label_index
from paper_trail_core.nlp.numeric import numeric_conflict


def snip(text="evidence text", relevance=0.75, trust=0.9, domain="reuters.com"):
    return {"text": text, "relevance": relevance, "trust": trust,
            "domain": domain, "url": f"https://{domain}/x"}


def ent(p=0.95):
    return NLIResult(entail=p, neutral=1 - p - 0.01, contradict=0.01)


def con(p=0.95):
    return NLIResult(entail=0.01, neutral=1 - p - 0.01, contradict=p)


def neu(p=0.98):
    return NLIResult(entail=(1 - p) / 2, neutral=p, contradict=(1 - p) / 2)


# ── NLI label mapping ─────────────────────────────────────────────────────────

def test_label_index_follows_config_not_position():
    """
    The two NLI checkpoints this project has used order their labels
    *inversely*. Reading positions instead of names would invert every verdict
    while the pipeline ran without error.
    """
    bart = build_label_index({0: "contradiction", 1: "neutral", 2: "entailment"})
    deberta = build_label_index({0: "entailment", 1: "neutral", 2: "contradiction"})

    assert bart["entailment"] == 2 and bart["contradiction"] == 0
    assert deberta["entailment"] == 0 and deberta["contradiction"] == 2
    assert bart["entailment"] != deberta["entailment"]


def test_label_index_rejects_unknown_labels():
    with pytest.raises(ValueError, match="missing label"):
        build_label_index({0: "yes", 1: "maybe", 2: "no"})


# ── Aggregation ───────────────────────────────────────────────────────────────

def test_confident_neutral_does_not_beat_entailment():
    """The exact shape of the old argmax-across-labels bug."""
    cv = aggregate_claim(
        "A claim.",
        [snip("off-topic but confident"), snip("clearly supporting")],
        [neu(0.99), ent(0.80)],
    )
    assert cv["verdict"] == "entailment"
    assert cv["support"] > 0


def test_single_strong_contradiction_outweighs_weak_support():
    cv = aggregate_claim(
        "A claim.",
        [snip("weak support", relevance=0.40, trust=0.5),
         snip("strong refutation", relevance=0.80, trust=1.0)],
        [ent(0.70), con(0.95)],
    )
    assert cv["verdict"] == "contradiction"
    assert cv["support"] < 0


def test_neutral_contributes_nothing():
    only_neutral = aggregate_claim("A claim.", [snip()], [neu()])
    assert only_neutral["verdict"] == "neutral"
    assert only_neutral["entail_mass"] == pytest.approx(0.0, abs=0.05)
    assert only_neutral["support"] == pytest.approx(0.0, abs=0.05)


def test_irrelevant_evidence_cannot_produce_a_confident_verdict():
    """Relevance gates weight, so a barely-relevant snippet can't decide a claim."""
    cv = aggregate_claim("A claim.", [snip(relevance=0.36, trust=0.5)], [con(0.99)])
    assert max(cv["entail_mass"], cv["contradict_mass"]) < DECISION_FLOOR
    assert cv["verdict"] == "neutral"


def test_support_is_bounded():
    cv = aggregate_claim("A claim.", [snip(relevance=1.0, trust=1.0)], [con(1.0)])
    assert -1.0 <= cv["support"] <= 1.0


# ── Provenance ────────────────────────────────────────────────────────────────

def test_evidence_meta_is_parallel_and_carries_stance():
    snippets = [snip("repeats it", domain="viralnews.in"),
                snip("debunks it", domain="altnews.in"),
                snip("unrelated", domain="reuters.com")]
    snippets[0]["published_at"] = "2026-10-28T09:00:00Z"
    snippets[0]["provider"] = "newsapi"
    cv = aggregate_claim("A claim.", snippets, [ent(0.9), con(0.9), neu()])

    meta = cv["evidence_meta"]
    assert len(meta) == len(cv["evidence_snippets"])
    assert [m["domain"] for m in meta] == cv["sources"]
    assert [m["stance"] for m in meta] == ["repeats", "debunks", "neutral"]
    assert meta[0]["url"] == "https://viralnews.in/x"
    assert meta[0]["published_at"] == "2026-10-28T09:00:00Z"
    assert meta[0]["provider"] == "newsapi"


def test_stance_ignores_trust():
    """A low-trust site repeating the claim is still recorded as repeating it."""
    cv = aggregate_claim("A claim.", [snip(trust=0.1, relevance=0.4)], [ent(0.9)])
    assert cv["verdict"] == "neutral"
    assert cv["evidence_meta"][0]["stance"] == "repeats"
    assert cv["evidence_meta"][0]["trust"] == pytest.approx(0.1)


def test_unverifiable_has_empty_evidence_meta():
    cv = aggregate_claim("A claim.", [], [], status="no_hits")
    assert cv["evidence_meta"] == []


# ── Absence of evidence ───────────────────────────────────────────────────────

@pytest.mark.parametrize("status", ["no_hits", "no_relevant", "rate_limited", "error"])
def test_missing_evidence_is_unverifiable_never_contradiction(status):
    """
    Regression: no evidence must never read as evidence against the article.

    Our retrieval is English-only, capped at ~100 news requests a day and
    roughly a one-month window — missing a claim is routine and says nothing
    about whether it is true.
    """
    cv = aggregate_claim("An unfindable claim.", [], [], status=status)
    assert cv["verdict"] == "unverifiable"
    assert cv["support"] == 0.0
    assert cv["contradict_mass"] == 0.0
    assert cv["confidence"] == 0.0
    assert not is_informative(cv)


def test_unverifiable_is_not_informative():
    assert not is_informative(aggregate_claim("c", [], [], status="no_hits"))


def test_strong_evidence_is_informative():
    cv = aggregate_claim("c", [snip(relevance=0.8, trust=1.0)], [con(0.95)])
    assert is_informative(cv)


# ── Relevance scaling ─────────────────────────────────────────────────────────

def test_scale_relevance_monotonic_and_bounded():
    assert scale_relevance(0.30) == 0.0            # below the gate
    assert scale_relevance(0.90) == 1.0            # at/above the ceiling
    assert 0.0 < scale_relevance(0.55) < 1.0
    assert scale_relevance(0.65) > scale_relevance(0.50)


def test_strong_match_carries_near_full_weight():
    """
    Scaling is against a realistic ceiling, not 1.0.

    Cosine 1.0 means "identical string", which real evidence never is. Scaling
    against it discounted every genuine match toward zero — a 0.7-cosine
    paraphrase carried under half the weight it should, which pushed correct
    contradictions below the decision floor and reported them as neutral.
    """
    assert scale_relevance(0.70) > 0.75
    assert scale_relevance(0.75) == 1.0


# ── Numeric conflict ──────────────────────────────────────────────────────────

def test_numeric_conflict_detects_rate_mismatch():
    assert numeric_conflict(
        "India raised its repo rate to 6.75 percent",
        "The Reserve Bank of India held the repo rate at 6.5 percent",
    ) > 0.15


def test_numeric_conflict_zero_when_figures_agree():
    assert numeric_conflict(
        "India raised its repo rate to 6.75 percent",
        "The Reserve Bank of India raised the repo rate to 6.75 percent",
    ) == 0.0


def test_year_mismatch_is_total_conflict():
    assert numeric_conflict(
        "Apollo 11 landed on the Moon in 1969",
        "Apollo 11 landed on the Moon in 1972",
    ) == 1.0


def test_no_conflict_without_a_shared_subject():
    """Two unrelated sentences with different percentages are not in conflict."""
    assert numeric_conflict(
        "Tesla revenue grew 5 percent",
        "Microsoft cloud spending rose 40 percent",
    ) == 0.0


# ── Lexical fallback ──────────────────────────────────────────────────────────

def test_lexical_fallback_is_tagged_as_degraded():
    """It may run, but it must always announce itself."""
    assert lexical_infer("a claim about rates", "some evidence about rates").backend == "lexical"
