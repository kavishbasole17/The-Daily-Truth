"""
Claim-extraction tests.

The headline test here is `test_decimal_is_not_split`. The original extractor
split on "." with no protection, so "6.75 percent" became two claims — "…to 6."
and "75 percent on Thursday…" — and every article containing a figure was
shredded into fragments before retrieval ever saw it. That bug is the reason
this suite exists, so it is asserted directly, in both directions: the current
backend must not do it, and the preserved `naive` backend must still do it, so
the test cannot pass vacuously if someone rewires the default.
"""
from __future__ import annotations

import pytest

from paper_trail_core.agents.claim_extractor import (
    extract_claims_heuristic,
    extract_claims_naive,
)
from paper_trail_core.nlp.checkworthiness import checkworthiness, deduplicate
from paper_trail_core.nlp.segmentation import segment_sentences

RATE_ARTICLE = (
    "NEW DELHI - India's central bank raised its benchmark repo rate by 25 basis "
    "points to 6.75 percent on Thursday, citing persistent food inflation. "
    "Governor Sanjay Malhotra said the monetary policy committee voted 4-2 in "
    "favour of the hike. Retail inflation stood at 5.8 percent in July, above "
    "the bank's 4 percent target."
)


# ── Segmentation ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("India raised its repo rate by 25 basis points to 6.75 percent on Thursday.", 1),
    ("Dr. Smith of the U.S. Dept. of Energy approved the plan on Monday.", 1),
    ("He left the building quickly. She stayed behind to finish the report.", 2),
    ("J. R. R. Tolkien wrote the novel in stages. It sold millions of copies.", 2),
    ("WASHINGTON (Reuters) - The head of the agency resigned on Tuesday.", 1),
    ("Inflation stood at 5.8 percent in July. Economists expected a hold.", 2),
])
def test_segmentation_counts(text, expected):
    assert len(segment_sentences(text)) == expected


def test_decimal_survives_segmentation():
    sentences = segment_sentences("The rate rose to 6.75 percent from 6.5 percent this week.")
    assert len(sentences) == 1
    assert "6.75" in sentences[0]
    assert "6.5" in sentences[0]


# ── The regression that motivated the rewrite ─────────────────────────────────

def test_decimal_is_not_split():
    """No extracted claim may be a fragment of a decimal number."""
    claims = extract_claims_heuristic(RATE_ARTICLE)
    assert claims, "expected at least one claim"

    joined = " ".join(claims)
    assert "6.75 percent" in joined
    # The signature of the old bug: a claim ending in a bare digit-plus-period,
    # or one starting mid-number.
    for c in claims:
        assert not c.rstrip().endswith(" 6."), f"decimal split survived: {c!r}"
        assert not c.lstrip().startswith("75 percent"), f"decimal split survived: {c!r}"


def test_naive_backend_still_reproduces_the_bug():
    """
    Guards the test above from becoming vacuous.

    If this ever stops failing to preserve decimals, the fixture no longer
    represents the original defect and `test_decimal_is_not_split` proves
    nothing.
    """
    claims = extract_claims_naive(RATE_ARTICLE, 5)
    joined = " ".join(claims)
    assert "6.75 percent" not in joined
    assert any(c.rstrip().endswith("6.") for c in claims)


def test_claims_are_complete_sentences():
    for claim in extract_claims_heuristic(RATE_ARTICLE):
        assert claim[0].isupper() or claim[0].isdigit(), f"fragment: {claim!r}"
        assert len(claim.split()) >= 6


# ── Check-worthiness ──────────────────────────────────────────────────────────

def test_factual_scores_above_opinion():
    factual = "Retail inflation stood at 5.8 percent in July, above the 4 percent target."
    opinion = "I think this was a terrible decision that the government should reverse."
    assert checkworthiness(factual) > checkworthiness(opinion)


def test_opinion_is_dropped():
    article = (
        "The Reserve Bank of India raised the repo rate to 6.75 percent on Thursday. "
        "I think this was a terrible decision that the government should reverse. "
        "Retail inflation stood at 5.8 percent in July, according to official data."
    )
    claims = extract_claims_heuristic(article)
    assert not any("terrible decision" in c for c in claims)


def test_questions_score_low():
    assert checkworthiness("Is the central bank going to raise rates again?") < 0.35


def test_claims_returned_in_document_order():
    claims = extract_claims_heuristic(RATE_ARTICLE)
    positions = [RATE_ARTICLE.index(c[:40]) for c in claims]
    assert positions == sorted(positions)


def test_respects_max_claims():
    long_article = " ".join(
        f"Official figures show output in sector {i} rose {i}.5 percent in March, "
        f"according to the National Statistics Agency."
        for i in range(1, 20)
    )
    assert len(extract_claims_heuristic(long_article, max_claims=3)) <= 3


def test_deduplicate_keeps_higher_scoring():
    a = (0, "The bank raised the repo rate to 6.75 percent on Thursday.", 0.9)
    b = (1, "The bank raised the repo rate to 6.75 percent on Thursday!", 0.4)
    c = (2, "Unrelated: the rupee weakened against the dollar in August.", 0.8)
    kept = deduplicate([a, b, c])
    texts = [t for _, t, _ in kept]
    assert a[1] in texts
    assert c[1] in texts
    assert b[1] not in texts


def test_short_text_yields_no_claims():
    assert extract_claims_heuristic("Too short.") == []
