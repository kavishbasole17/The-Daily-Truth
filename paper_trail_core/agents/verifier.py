"""
Claim-verification agent node.

For each claim, decides whether the retrieved evidence supports it,
contradicts it, or says nothing useful — then aggregates across snippets into
one per-claim support score.

Two things here were broken badly enough to invert results.

**The old backend could not detect contradiction.** With MOCK_MODE on (the
shipped default) verification was lexical word overlap plus a list of canned
negation words. Evidence saying the rate was *held at 6.5* scored 0.83
`entailment` for a claim that it was *raised to 6.75*, because the sentences
share nearly every word. Real NLI now handles this (0.998 contradiction on
that exact pair); the lexical path survives only as an explicitly-flagged
emergency fallback.

**The old aggregation compared confidences across labels.** It looped over
snippets keeping whichever verdict had the highest confidence — so a `neutral`
at 0.9 beat an `entailment` at 0.7, and an article whose evidence was mostly
off-topic came back "neutral" even when one snippet flatly refuted it. Labels
are not points on a scale; you cannot argmax across them. Entailment and
contradiction now accumulate in separate masses and only those two are ever
compared. Neutral contributes nothing at all, because that is what it means:
absence of signal, not a third opinion.
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
    _BACKEND  = settings.NLI_BACKEND
    _MAX_PAIRS = settings.NLI_MAX_PAIRS
    _REL_MIN  = settings.RELEVANCE_MIN
except ImportError:  # pragma: no cover
    _BACKEND, _MAX_PAIRS, _REL_MIN = "transformer", 200, 0.35

from paper_trail_core.agents.state import ClaimVerdict, PipelineState
from paper_trail_core.degradations import degrade
from paper_trail_core.nlp.nli import NLIResult, infer_batch, is_available
from paper_trail_core.nlp.numeric import numeric_conflict
from paper_trail_core.nlp.segmentation import segment_sentences

log = logging.getLogger(__name__)

# Cosine at which a snippet is considered a fully on-topic match. Measured, not
# guessed: MiniLM puts genuinely relevant claim/evidence news pairs in the
# 0.45-0.70 band, and near-paraphrases around 0.8. Treating 1.0 as the target
# made a 0.53-cosine match — a good one — carry only ~0.28 weight, which pushed
# correct contradictions below the decision floor and reported them as neutral.
RELEVANCE_CEILING = 0.75

# Below this, neither direction has enough mass to call — the claim is neutral.
DECISION_FLOOR = 0.20

# Contradiction is held to a higher bar than entailment, because the two error
# modes are not symmetric in practice.
#
# A FEVER-trained NLI model expects its premise to be the *specific* evidence
# sentence for the claim. Feed it a topically adjacent paragraph — a
# Chandrayaan-2 page for a Chandrayaan-3 claim, an article about a different
# year's Nobel Prize — and it frequently returns confident `contradiction`
# purely because the details don't line up. Measured on the golden set, that
# spurious-refutation mode turned several plainly true articles into
# "likely_fake", while the reverse (spurious entailment) barely occurred.
#
# So a refutation must clear a higher mass, and must beat corroboration by a
# margin rather than merely edging it out.
CONTRADICTION_FLOOR = 0.35
CONTRADICTION_MARGIN = 0.10
# A claim only counts as "informative" (i.e. gets to move the article verdict)
# if its best evidence was reasonably on-topic.
INFORMATIVE_WEIGHT_FLOOR = 0.25

# In claim mode a page is "anchored" when its title appears in the claim. Only
# the pages about the first two things the claim names (the subject, and usually
# where it is) may refute it. A page for something mentioned in passing ("... on
# the Arabian Sea") says nothing about the subject, and an NLI model reads that
# silence as contradiction.
SUBJECT_RANK_LIMIT = 2

# A near-paraphrase of the claim may refute it whatever page it is on.
PARAPHRASE_RELEVANCE = 0.80


def _may_refute(snip: dict) -> bool:
    """
    In claim mode a snippet may only *refute* if it is about the claim's own
    subject: a page named after one of the first things the claim mentions, or a
    near-paraphrase of the claim. A neighbouring page that merely scored well
    (the Blue Nile for a claim about the Nile, Agra Fort for one about the Red
    Fort) holds true facts about something else, which an NLI model reads as
    contradiction. Such pages can still support a claim.
    """
    if snip.get("anchored") and snip.get("subject_rank", 99) < SUBJECT_RANK_LIMIT:
        return True
    return float(snip.get("relevance", 0.0)) >= PARAPHRASE_RELEVANCE

# A claim about a happening ("shut 57 stations from 9pm on Friday", "live
# updates") is true or false at a moment. A static page cannot refute it, and a
# news item about the same story is usually a different day's update with
# different figures (17 stations closed on day 3, not 57 on day 1).
_TIME_BOUND_RE = re.compile(
    r"\b(?:mon|tues?|wednes|thurs?|fri|satur|sun)day\b"
    r"|\b\d{1,2}(?::\d{2})?\s?(?:am|pm)\b"
    r"|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}\b"
    r"|\b\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b"
    r"|\b(?:live updates?|breaking|today|tonight|tomorrow|yesterday|this week|ahead of)\b",
    re.IGNORECASE,
)


def _is_time_bound(claim: str) -> bool:
    return bool(_TIME_BOUND_RE.search(claim or ""))


def _is_news_source(snip: dict) -> bool:
    """A regional news item: an update on a story, not a reference page."""
    return bool(snip.get("in_region")) and not str(
        snip.get("domain", "")).endswith("wikipedia.org")


# How much a detected numeric conflict may add to contradiction probability.
# Deliberately partial: the NLI model already catches most numeric
# contradictions on its own, and a figure that differs may be a revision or a
# differently-defined metric rather than a refutation.
NUMERIC_BOOST = 0.5


# ── Lexical fallback (degraded — never allowed to drive a committed verdict) ───

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
    return {w for w in re.findall(r"[a-z0-9]+", text.lower())
            if w not in _STOPWORDS and len(w) > 2}


def lexical_infer(claim: str, snippet: str) -> NLIResult:
    """
    Word-overlap stand-in for NLI. Retained for the case where the transformer
    genuinely cannot load, and marked `backend="lexical"` so callers can refuse
    to act on it.
    """
    claim_words = _content_words(claim)
    snippet_words = _content_words(snippet)
    overlap = len(claim_words & snippet_words) / max(len(claim_words), 1)
    has_negation = any(m in snippet.lower() for m in _NEGATION_MARKERS)

    if overlap < 0.25:
        return NLIResult(entail=0.1, neutral=0.8, contradict=0.1, backend="lexical")
    conf = min(0.55 + 0.35 * overlap, 0.9)
    if has_negation:
        return NLIResult(entail=0.05, neutral=1 - conf - 0.05, contradict=conf, backend="lexical")
    return NLIResult(entail=conf, neutral=1 - conf - 0.05, contradict=0.05, backend="lexical")


# ── Aggregation ───────────────────────────────────────────────────────────────

def scale_relevance(cosine: float, floor: float = _REL_MIN,
                    ceiling: float = RELEVANCE_CEILING) -> float:
    """
    Rescale a raw cosine to [0, 1] over the band that actually discriminates.

    A snippet that only just clears the gate should carry little weight, and
    one at or above `ceiling` should carry full weight. Scaling against 1.0
    instead would be wrong in practice: cosine 1.0 means "identical string",
    which real evidence never is, so every genuine match would be discounted
    toward zero.
    """
    if cosine <= floor:
        return 0.0
    return round(min(1.0, (cosine - floor) / max(1e-6, ceiling - floor)), 4)


def aggregate_claim(
    claim: str,
    snippets: list[dict],
    results: list[NLIResult],
    status: str = "ok",
) -> ClaimVerdict:
    """
    Fold per-snippet NLI results into one verdict for the claim.

        wᵢ            = scaled_relevanceᵢ × trustᵢ
        entail_mass   = maxᵢ (wᵢ · P(entailment)ᵢ)
        contra_mass   = maxᵢ (wᵢ · P(contradiction)ᵢ)
        support       = entail_mass − contra_mass          ∈ [−1, 1]

    Max rather than mean, deliberately. Fact-checking is asymmetric: one solid
    refutation from a wire service settles the question, while ten vague
    corroborations do not. The relevance×trust weighting is what stops a single
    junk snippet from exploiting that — an off-topic or low-trust snippet
    cannot reach a high mass no matter how confident the model is about it.

    The entailment/contradiction asymmetry is *not* hand-tuned here. Both
    masses are emitted as separate features so the fusion layer can learn the
    relative weight from data instead of us asserting one.
    """
    if status != "ok" or not snippets or not results:
        return ClaimVerdict(
            claim=claim,
            verdict="unverifiable",
            confidence=0.0,
            evidence_snippets=[],
            support=0.0,
            entail_mass=0.0,
            contradict_mass=0.0,
            top_relevance=0.0,
            numeric_conflict=0.0,
            sources=[],
            status=status,
        )

    entail_mass = 0.0
    contradict_mass = 0.0
    top_relevance = 0.0
    max_numeric = 0.0

    for snip, res in zip(snippets, results):
        rel = scale_relevance(float(snip.get("relevance", 0.0)))
        trust = float(snip.get("trust", 0.5))
        w = rel * trust
        top_relevance = max(top_relevance, float(snip.get("relevance", 0.0)))

        p_ent, p_con = res.entail, res.contradict
        cannot_refute = False
        if snip.get("claim_mode"):
            cannot_refute = (
                not _may_refute(snip)
                or _is_news_source(snip)
                or _is_time_bound(claim)
            )
        if cannot_refute:
            p_con = 0.0

        # Numeric disagreement the NLI head may have read as merely "unrelated".
        nc = 0.0 if cannot_refute else numeric_conflict(claim, snip.get("text", ""))
        if nc > 0:
            max_numeric = max(max_numeric, nc)
            p_con = min(1.0, p_con + NUMERIC_BOOST * nc)
            p_ent = p_ent * (1.0 - nc)

        entail_mass = max(entail_mass, w * p_ent)
        contradict_mass = max(contradict_mass, w * p_con)

    # A refutation only counts once it clears the higher contradiction bar; a
    # weak one is discounted rather than allowed to drive the article's score.
    refutes = (
        contradict_mass >= CONTRADICTION_FLOOR
        and contradict_mass >= entail_mass + CONTRADICTION_MARGIN
    )
    effective_contradiction = contradict_mass if refutes else contradict_mass * 0.25
    support = entail_mass - effective_contradiction

    if refutes:
        verdict = "contradiction"
        confidence = round(contradict_mass, 4)
    elif entail_mass >= DECISION_FLOOR and entail_mass > contradict_mass:
        verdict = "entailment"
        confidence = round(entail_mass, 4)
    else:
        verdict = "neutral"
        confidence = round(1.0 - max(entail_mass, contradict_mass), 4)

    return ClaimVerdict(
        claim=claim,
        verdict=verdict,
        confidence=confidence,
        evidence_snippets=[s.get("text", "") for s in snippets],
        support=round(support, 4),
        entail_mass=round(entail_mass, 4),
        contradict_mass=round(contradict_mass, 4),
        top_relevance=round(top_relevance, 4),
        numeric_conflict=round(max_numeric, 4),
        sources=[s.get("domain", "") for s in snippets],
        status=status,
        in_region=[bool(s.get("in_region")) for s in snippets],
    )


def _attach_event_corroboration(cv: ClaimVerdict, claim: str, snippets: list[dict]) -> None:
    """
    For a time-bound short claim that nothing refutes, record whether recent
    independent news items report the same event. It does not change the NLI
    verdict; the claim-mode adjudicator reads these fields.
    """
    if cv.get("verdict") == "contradiction" or not _is_time_bound(claim):
        return
    if not any(s.get("claim_mode") for s in snippets):
        return
    try:
        from config import settings
        if not settings.EVENT_CORROBORATION:
            return
        from paper_trail_core.nlp.event_match import corroboration
        c = corroboration(
            claim, snippets,
            min_snippets=settings.EVENT_MIN_SNIPPETS,
            min_sources=settings.EVENT_MIN_SOURCES,
            max_age_days=settings.EVENT_MAX_AGE_DAYS,
        )
    except Exception:   # noqa: BLE001 — corroboration is optional context
        log.exception("verifier: event corroboration failed")
        return
    cv["event_corroborated"] = bool(c["corroborated"])
    cv["corroborating_sources"] = c["sources"] if c["corroborated"] else []
    cv["detail_mismatch"] = bool(c["detail_mismatch"]) if c["corroborated"] else False


def is_informative(cv: ClaimVerdict) -> bool:
    """Whether a claim verdict earned the right to move the article-level score."""
    return (
        cv.get("status") == "ok"
        and cv.get("verdict") in ("entailment", "contradiction")
        and max(cv.get("entail_mass", 0.0), cv.get("contradict_mass", 0.0)) >= DECISION_FLOOR
        and scale_relevance(cv.get("top_relevance", 0.0)) >= INFORMATIVE_WEIGHT_FLOOR
    )


# ── Node ──────────────────────────────────────────────────────────────────────

def _premise_sentences(snippet_text: str, limit: int = 4) -> list[str]:
    """
    Split a snippet into sentences for NLI.

    FEVER premises are single sentences, and that is what the model is good at.
    A four-sentence premise dilutes the signal: the one sentence that refutes
    the claim gets averaged against three that merely mention the topic, and
    contradiction gets suppressed into neutral.
    """
    sents = segment_sentences(snippet_text, min_chars=20)
    if not sents:
        return [snippet_text.strip()] if snippet_text.strip() else []
    return sents[:limit]


def verify_claims_node(state: PipelineState) -> dict:
    """LangGraph node: run NLI over each (evidence, claim) pair and aggregate."""
    evidence_list = state.get("evidence", [])
    log.info("verifier: %d evidence items, backend=%s", len(evidence_list), _BACKEND)

    if not evidence_list:
        return {"claim_verdicts": [], "nli_degraded": False, "degradations": []}

    degradations: list = []
    use_transformer = _BACKEND == "transformer" and is_available()
    nli_degraded = not use_transformer

    if _BACKEND == "transformer" and not use_transformer:
        degradations += degrade(
            "verifier", "nli_fallback_lexical",
            "The NLI model could not be loaded, so claim checking fell back to "
            "word-overlap matching. That heuristic cannot detect contradiction "
            "and has scored directly contradictory evidence as support, so this "
            "run's evidence is not trusted and the verdict is left uncertain.",
            severity="critical",
        )
    elif _BACKEND == "lexical":
        degradations += degrade(
            "verifier", "nli_backend_lexical",
            "NLI_BACKEND=lexical — word-overlap matching cannot detect "
            "contradiction. Evidence will not be trusted for a committed verdict.",
            severity="critical",
        )
    elif _BACKEND == "off":
        log.info("verifier: NLI disabled by config")
        return {
            "claim_verdicts": [
                ClaimVerdict(
                    claim=item.get("claim", ""), verdict="unverifiable", confidence=0.0,
                    evidence_snippets=[], support=0.0, entail_mass=0.0,
                    contradict_mass=0.0, top_relevance=0.0, numeric_conflict=0.0,
                    sources=[], status="disabled",
                )
                for item in evidence_list
            ],
            "nli_degraded": True,
            "degradations": degrade(
                "verifier", "nli_disabled",
                "NLI_BACKEND=off — no claim was checked against its evidence.",
                severity="critical",
            ),
        }

    # Flatten every (premise sentence, claim) pair so the whole article goes
    # through the model in one batched call instead of one forward pass per
    # snippet.
    plan: list[tuple[int, int, str, str, float]] = []  # (claim_i, snip_i, premise, claim, weight)
    for ci, item in enumerate(evidence_list):
        claim = item.get("claim", "")
        if item.get("status") != "ok":
            continue
        for si, snip in enumerate(item.get("snippets", [])):
            weight = scale_relevance(float(snip.get("relevance", 0.0))) * float(snip.get("trust", 0.5))
            premises = _premise_sentences(snip.get("text", ""))
            if snip.get("claim_mode") and snip.get("anchored"):
                # Later lead sentences refer to the subject as "it" / "the fort";
                # naming the page keeps them about that subject.
                title = (snip.get("text") or "").split(":", 1)[0].strip()
                premises = [p if i == 0 else f"{title}: {p}" for i, p in enumerate(premises)]
            for premise in premises:
                plan.append((ci, si, premise, claim, weight))

    # Budget cap: keep the highest-weight pairs so a pathological article can't
    # stall the request. Sorted by weight then position, so it stays deterministic.
    if len(plan) > _MAX_PAIRS:
        plan.sort(key=lambda p: (-p[4], p[0], p[1]))
        plan = plan[:_MAX_PAIRS]
        plan.sort(key=lambda p: (p[0], p[1]))
        degradations += degrade(
            "verifier", "pair_cap",
            f"Article produced more than {_MAX_PAIRS} evidence pairs; kept the "
            "most relevant ones.",
            severity="info",
        )

    # Run inference.
    if plan:
        premises = [p[2] for p in plan]
        hypotheses = [p[3] for p in plan]
        try:
            results = (infer_batch(premises, hypotheses) if use_transformer
                       else [lexical_infer(h, p) for p, h in zip(premises, hypotheses)])
        except Exception as exc:
            log.error("verifier: NLI inference failed (%s), falling back to lexical", exc)
            results = [lexical_infer(h, p) for p, h in zip(premises, hypotheses)]
            nli_degraded = True
            degradations += degrade(
                "verifier", "nli_fallback_lexical",
                f"NLI inference failed mid-run ({type(exc).__name__}); fell back "
                "to word-overlap matching, which cannot detect contradiction.",
                severity="critical",
            )
    else:
        results = []

    # Best result per (claim, snippet): the strongest sentence in a snippet
    # represents that snippet, per the max-aggregation rationale above.
    best: dict[tuple[int, int], NLIResult] = {}
    for (ci, si, _p, _h, _w), res in zip(plan, results):
        prev = best.get((ci, si))
        if prev is None or max(res.entail, res.contradict) > max(prev.entail, prev.contradict):
            best[(ci, si)] = res

    claim_verdicts: list[ClaimVerdict] = []
    for ci, item in enumerate(evidence_list):
        claim = item.get("claim", "")
        status = item.get("status", "ok")
        snippets = item.get("snippets", []) if status == "ok" else []
        ordered = [best.get((ci, si)) for si in range(len(snippets))]
        pairs = [(s, r) for s, r in zip(snippets, ordered) if r is not None]
        cv = aggregate_claim(
            claim,
            [s for s, _ in pairs],
            [r for _, r in pairs],
            status=status if pairs or status != "ok" else "no_relevant",
        )
        _attach_event_corroboration(cv, claim, snippets)
        # Region metadata is reported alongside the verdict; it plays no part in
        # computing it, so verdicts and fusion features are identical with or
        # without it.
        location = item.get("location")
        if location:
            region = item.get("region") or {}
            cv["location"] = location
            cv["search_scope"] = region.get("scope_used") or (
                (region.get("scopes_tried") or [None])[-1]
            )
            cv["regional_evidence"] = sum(1 for flag in cv.get("in_region", []) if flag)
        claim_verdicts.append(cv)

    counts = {v: sum(1 for c in claim_verdicts if c["verdict"] == v)
              for v in ("entailment", "contradiction", "neutral", "unverifiable")}
    log.info("verifier: %d verdicts — %s", len(claim_verdicts), counts)

    return {
        "claim_verdicts": claim_verdicts,
        "nli_degraded": nli_degraded,
        "degradations": degradations,
    }
