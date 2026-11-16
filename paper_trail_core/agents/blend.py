"""
Blend the agentic pipeline's verdict with the independent LLM fact-check.

Pure function, no I/O. The rules, in order:

  1. **Weights.** The LLM check starts at LLM_JUDGE_WEIGHT (default 0.6) and the
     pipeline at the remainder (0.4). Each is then discounted for the things
     that make its own number less trustworthy:
       * the LLM found no web sources (knowledge only)       -> its weight x 0.5
       * the LLM reports low confidence in its own number    -> x (0.5 + 0.5 x confidence)
       * NLI ran on the degraded word-overlap backend        -> pipeline weight 0
       * no claim could be checked against any evidence      -> pipeline weight x 0.5
     The final P(real) is the weight-normalised average.

  2. **Agreement.** When both checks commit to the same side the verdict is
     committed; this is the case that is far more reliable than either alone.

  3. **Disagreement guard.** When they commit to *opposite* sides the verdict
     is "uncertain", whatever the weighted average says. 0.6 x 0.9 + 0.4 x 0.2
     is 0.62, which would read as "likely real at 62%" and hide the fact that
     one of two careful checks thinks the text is false.

  4. **Lone commitment.** When only the LLM commits (the pipeline abstained or
     was discounted to nothing), it must clear LLM_JUDGE_MIN_CONFIDENCE and
     have found at least one source if search was on; otherwise "uncertain".

Everything the blend used is returned so the trace and the UI can show it.
"""
from __future__ import annotations

from typing import Any


def _side(p: float, low: float, high: float) -> str | None:
    if p >= high:
        return "real"
    if p <= low:
        return "fake"
    return None


def _verdict_side(verdict: str) -> str | None:
    return {"likely_real": "real", "likely_fake": "fake"}.get(verdict)


def blend_verdicts(
    p_agentic: float,
    agentic_verdict: str,
    judge: dict[str, Any] | None,
    *,
    weight_llm: float = 0.6,
    low: float = 0.4,
    high: float = 0.6,
    nli_degraded: bool = False,
    coverage: float = 1.0,
    min_llm_confidence: float = 0.6,
) -> dict[str, Any]:
    """
    Returns {verdict, confidence, p_final, p_agentic, p_llm, w_agentic, w_llm,
    rule, note, agentic_verdict}. With `judge=None` the agentic verdict is
    returned unchanged (rule "agentic_only").
    """
    out: dict[str, Any] = {
        "agentic_verdict": agentic_verdict, "p_agentic": round(float(p_agentic), 4),
        "p_llm": None, "w_agentic": 1.0, "w_llm": 0.0,
        "p_final": round(float(p_agentic), 4), "verdict": agentic_verdict,
        "confidence": None, "rule": "agentic_only", "note": "",
    }
    if judge is None:
        return out

    p_llm = max(0.0, min(1.0, float(judge.get("p_real", 0.5))))
    c_llm = max(0.0, min(1.0, float(judge.get("confidence", 0.0))))
    has_sources = bool(judge.get("sources"))
    notes: list[str] = []

    w_llm = max(0.0, min(1.0, float(weight_llm)))
    if not has_sources:
        w_llm *= 0.5
        notes.append("LLM check found no web sources, so it counts half")
    w_llm *= 0.5 + 0.5 * c_llm

    w_agentic = 1.0 - max(0.0, min(1.0, float(weight_llm)))
    if nli_degraded:
        w_agentic = 0.0
        notes.append("pipeline NLI was degraded, so the pipeline score is not used")
    elif coverage <= 0.0:
        w_agentic *= 0.5
        notes.append("pipeline found no evidence for any claim, so it counts half")

    total = w_llm + w_agentic
    if total <= 0.0:
        out.update(p_llm=round(p_llm, 4), w_agentic=0.0, w_llm=0.0, p_final=0.5,
                   verdict="uncertain", confidence=0.0, rule="no_usable_signal",
                   note="; ".join(notes))
        return out

    p_final = (w_llm * p_llm + w_agentic * p_agentic) / total
    p_final = max(0.0, min(1.0, p_final))

    llm_side = _side(p_llm, low, high)
    agentic_side = _verdict_side(agentic_verdict)
    blended_side = _side(p_final, low, high)

    if llm_side and agentic_side and llm_side != agentic_side:
        rule, verdict, confidence = "disagreement", "uncertain", 0.0
        notes.append(f"the pipeline said {agentic_verdict} and the LLM check said "
                     f"{judge.get('verdict', 'the opposite')}; abstaining")
    elif llm_side and agentic_side and llm_side == agentic_side:
        rule = "agreement"
        verdict = "likely_real" if llm_side == "real" else "likely_fake"
        confidence = p_final if verdict == "likely_real" else 1.0 - p_final
    elif blended_side and not agentic_side:
        # The LLM carries the commitment on its own (the pipeline abstained or
        # was discounted). Demand that it be sure, and grounded.
        if c_llm < min_llm_confidence:
            rule, verdict, confidence = "llm_low_confidence", "uncertain", 0.0
            notes.append(f"only the LLM check committed, with confidence {c_llm:.2f} "
                         f"below the {min_llm_confidence:.2f} bar")
        elif judge.get("used_web_search") and not has_sources:
            rule, verdict, confidence = "llm_ungrounded", "uncertain", 0.0
            notes.append("only the LLM check committed, and it found no web sources")
        else:
            rule = "llm_led"
            verdict = "likely_real" if blended_side == "real" else "likely_fake"
            confidence = p_final if verdict == "likely_real" else 1.0 - p_final
    elif blended_side:
        rule = "weighted"
        verdict = "likely_real" if blended_side == "real" else "likely_fake"
        confidence = p_final if verdict == "likely_real" else 1.0 - p_final
    else:
        rule, verdict, confidence = "weighted", "uncertain", 0.0

    out.update(
        p_llm=round(p_llm, 4), w_agentic=round(w_agentic / total, 4), w_llm=round(w_llm / total, 4),
        p_final=round(p_final, 4), verdict=verdict, confidence=round(confidence, 4),
        rule=rule, note="; ".join(notes),
    )
    return out
