"""
Structured degradation records.

The old pipeline caught exceptions and quietly carried on with a worse
algorithm — a lexical stand-in for NLI, canned snippets instead of retrieval,
0.5 instead of a model score. The output looked identical to a healthy run, so
a systematically degraded verdict was indistinguishable from a good one.

Every fallback now emits a Degradation. They accumulate across nodes (see the
`operator.add` reducer on PipelineState.degradations), ride along on
VerdictResult, get persisted, and are printed in the reasoning trace. If the
system did something other than its best, it says so.
"""
from __future__ import annotations

from typing import Any, Literal

from typing_extensions import TypedDict

Severity = Literal["info", "warn", "critical"]


class Degradation(TypedDict):
    stage: str      # claim_extractor | evidence_retriever | verifier | ensemble | adjudicator
    kind: str       # short machine-readable slug, e.g. nli_fallback_lexical
    detail: str     # human-readable explanation
    severity: str   # info | warn | critical


def degrade(stage: str, kind: str, detail: str, severity: Severity = "warn") -> list[Degradation]:
    """
    Build a single-element degradation list.

    Returns a list rather than a bare dict so node return values can say
    `{"degradations": degrade(...)}` and have LangGraph's reducer concatenate
    them without every caller remembering to wrap.
    """
    return [Degradation(stage=stage, kind=kind, detail=detail, severity=severity)]


def worst_severity(degradations: list[Degradation]) -> str | None:
    """Highest severity present, or None if the run was clean."""
    if not degradations:
        return None
    order = {"info": 0, "warn": 1, "critical": 2}
    return max(degradations, key=lambda d: order.get(d["severity"], 0))["severity"]


def has_kind(degradations: list[Degradation], kind: str) -> bool:
    return any(d["kind"] == kind for d in degradations)


def format_degradations(degradations: list[Degradation]) -> str:
    """Plain-text block for the reasoning trace (rendered raw in a <pre>)."""
    if not degradations:
        return ""
    lines = ["", "Degradations (the system did not run at full capability):"]
    for d in degradations:
        lines.append(f"  [{d['severity'].upper()}] {d['stage']}/{d['kind']}: {d['detail']}")
    return "\n".join(lines)


def summarise(degradations: list[Degradation]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for d in degradations:
        counts[d["kind"]] = counts.get(d["kind"], 0) + 1
    return {"n": len(degradations), "worst": worst_severity(degradations), "kinds": counts}
