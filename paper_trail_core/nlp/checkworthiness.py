"""
Check-worthiness scoring.

Not every sentence in an article is a factual claim you can go and verify.
"The government should do more about this" cannot be checked against any
source; "Retail inflation stood at 5.8 percent in July" can. Feeding the
former to a retrieval-and-NLI pipeline produces confident noise.

The previous extractor took the first N sentences regardless, so opinion,
transitions and bylines all became "claims" and consumed the evidence budget.

This is a lexical approximation of ClaimBuster's check-worthiness classifier.
ClaimBuster proper is an SVM over part-of-speech, entity and sentiment
features; without a POS tagger installed we approximate with surface cues.
The scoring is additive and every term is interpretable, which matters more
here than squeezing out the last few points of ranking quality — the score is
used to *rank* sentences, and ranking is forgiving.
"""
from __future__ import annotations

import re

from paper_trail_core.nlp.numeric import extract_entities, extract_numbers

_RE_ATTRIBUTION = re.compile(
    r"\b(said|says|announced|reported|confirmed|stated|told|claimed|warned|"
    r"found|ruled|voted|signed|filed|testified|declared|denied|according to|"
    r"disclosed|revealed|estimated|projected|published)\b",
    re.IGNORECASE,
)

_RE_TEMPORAL = re.compile(
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"january|february|march|april|may|june|july|august|september|october|"
    r"november|december|yesterday|today|tomorrow|last (?:week|month|year)|"
    r"this (?:week|month|year)|(?:19|20)\d{2})\b",
    re.IGNORECASE,
)

_RE_QUANTIFIER = re.compile(
    r"\b(percent|per cent|billion|million|trillion|crore|lakh|"
    r"more than|less than|up from|down from|record|rose|fell|increased|"
    r"decreased|doubled|halved|majority|median|average)\b|%",
    re.IGNORECASE,
)

_RE_OPINION = re.compile(
    r"\b(should|must|ought|believe|think|feel|arguably|obviously|clearly|"
    r"surely|amazing|shocking|disgusting|outrageous|terrible|wonderful|"
    r"i|we|you|my|our)\b",
    re.IGNORECASE,
)

_RE_PRONOUN_START = re.compile(r"^\s*(he|she|it|they|this|that|these|those)\b", re.IGNORECASE)

_RE_WEAK_START = re.compile(
    r"^\s*(and|but|however|meanwhile|moreover|furthermore|also|then|so)\b",
    re.IGNORECASE,
)

_RE_TOKEN = re.compile(r"[A-Za-z0-9']+")

MIN_TOKENS = 6
# The NLI hypothesis must survive premise-only truncation intact; capping the
# claim length here guarantees that regardless of how long the evidence is.
MAX_TOKENS = 60


def tokens(text: str) -> list[str]:
    return _RE_TOKEN.findall(text)


def checkworthiness(sentence: str) -> float:
    """
    Score a sentence's suitability for external fact-checking, in [0, 1].

    Positive signals are the fingerprints of reported fact — a figure, a named
    actor, an attribution, a date. Negative signals mark sentences that cannot
    be checked even in principle: opinions, questions, and sentences whose
    subject is an unresolved pronoun ("He said the rate would rise" is not
    verifiable once lifted out of its article).
    """
    s = sentence.strip()
    if not s:
        return 0.0

    n_tokens = len(tokens(s))
    score = 0.0

    if extract_numbers(s) or re.search(r"\d", s):
        score += 0.30
    if extract_entities(s):
        score += 0.20
    if _RE_ATTRIBUTION.search(s):
        score += 0.20
    if _RE_TEMPORAL.search(s):
        score += 0.15
    if _RE_QUANTIFIER.search(s):
        score += 0.10

    if _RE_OPINION.search(s):
        score -= 0.25
    if s.rstrip().endswith("?"):
        score -= 0.20
    # Unresolved subject: the cheap stand-in for coreference resolution.
    # Dropping these beats trying to resolve them with no NLP stack available.
    if _RE_PRONOUN_START.match(s) and not extract_entities(s):
        score -= 0.20
    if _RE_WEAK_START.match(s) and not extract_entities(s):
        score -= 0.10
    if n_tokens < 8 or len(s) > 320:
        score -= 0.15

    return round(max(0.0, min(1.0, score)), 4)


def is_eligible(sentence: str) -> bool:
    """Hard filters applied before scoring — length bounds only."""
    n = len(tokens(sentence))
    return MIN_TOKENS <= n <= MAX_TOKENS


def _content_words(text: str) -> set[str]:
    return {w.lower() for w in tokens(text) if len(w) > 3}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def deduplicate(
    candidates: list[tuple[int, str, float]],
    cosine_threshold: float = 0.92,
) -> list[tuple[int, str, float]]:
    """
    Drop near-duplicate claims, keeping the higher-scoring member.

    Two-stage on purpose: a cheap Jaccard prefilter over content words decides
    which pairs are even worth embedding, then MiniLM cosine makes the final
    call on the survivors. Articles routinely restate their lead in the body,
    and duplicate claims would spend the retrieval budget twice on the same
    question while double-counting the resulting evidence.

    `candidates` is [(doc_position, sentence, score)]; returns the same shape.
    """
    if len(candidates) < 2:
        return candidates

    ranked = sorted(candidates, key=lambda c: -c[2])
    word_sets = {i: _content_words(s) for i, (_, s, _) in enumerate(ranked)}

    suspect_pairs: list[tuple[int, int]] = []
    for i in range(len(ranked)):
        for j in range(i + 1, len(ranked)):
            if _jaccard(word_sets[i], word_sets[j]) >= 0.6:
                suspect_pairs.append((i, j))

    duplicates: set[int] = set()
    if suspect_pairs:
        try:
            from paper_trail_core.nlp.embeddings import pairwise_cosine
            sims = pairwise_cosine([s for _, s, _ in ranked])
            for i, j in suspect_pairs:
                if j not in duplicates and sims[i][j] >= cosine_threshold:
                    duplicates.add(j)   # ranked is score-sorted, so j is the weaker one
        except Exception:
            # Embeddings unavailable: fall back to the lexical prefilter alone.
            for i, j in suspect_pairs:
                if j not in duplicates and _jaccard(word_sets[i], word_sets[j]) >= 0.85:
                    duplicates.add(j)

    return [c for i, c in enumerate(ranked) if i not in duplicates]
