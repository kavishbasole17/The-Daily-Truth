"""
Answer follow-up questions about a conversation from its own evidence.

Every check in a chat leaves behind claims, per-claim verdicts and the evidence
snippets (with outlet and link) they were judged on. A follow-up question is
answered from *that* material only — numbered so the answer can cite [n] — so
the assistant can explain its own verdicts without inventing new facts.

Two modes:
  llm         — Claude (or the configured provider) writes the answer, citing [n]
  extractive  — no LLM available (no key, error, refusal): return the most
                relevant evidence for the question, ranked lexically, and say so
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

try:
    from config import settings
except ImportError:  # pragma: no cover
    settings = None

MAX_SOURCES = 40          # evidence items placed in the prompt
MAX_SNIPPET_CHARS = 600

VERDICT_WORDS = {
    "entailment": "supported",
    "contradiction": "contradicted",
    "neutral": "inconclusive",
    "unverifiable": "could not be verified",
}
OVERALL_WORDS = {
    "likely_real": "likely real",
    "likely_fake": "likely misinformation",
    "uncertain": "uncertain",
}

SYSTEM_PROMPT = (
    "You are Paper Trail, the assistant inside a fact-checking app. The user has "
    "already run one or more checks in this chat; you are given those checks: the "
    "text checked, the overall verdict, each extracted claim with its verdict, and "
    "the numbered evidence snippets each claim was judged against.\n\n"
    "Answer the user's follow-up question using only that material. Cite evidence "
    "with its number in square brackets, like [3], right after the statement it "
    "supports. If the material doesn't answer the question, say so plainly and "
    "suggest what the user could check next (for example, pasting a specific "
    "article). Never present outside knowledge as something the evidence shows. "
    "Explain verdicts in plain language: 'supported' means the evidence agrees, "
    "'contradicted' means it disagrees, 'inconclusive' means evidence was found "
    "but didn't settle it, 'could not be verified' means no relevant evidence was "
    "found, which is not evidence that the claim is false. Keep answers short: a "
    "few sentences, or a brief list when comparing claims. Answer in the language "
    "the question is written in."
)


@dataclass
class Citation:
    n: int
    claim: str
    source: str
    url: str
    snippet: str

    def to_dict(self) -> dict[str, Any]:
        return {"n": self.n, "claim": self.claim, "source": self.source,
                "url": self.url, "snippet": self.snippet}


@dataclass
class QAContext:
    text: str                                   # the prompt block
    citations: list[Citation] = field(default_factory=list)


@dataclass
class QAResult:
    answer: str
    citations: list[dict[str, Any]]
    mode: str                                   # llm | extractive


# ── Context ─────────────────────────────────────────────────────────────────────

def build_context(checks: list[dict[str, Any]]) -> QAContext:
    """
    `checks` are ArticleOut-shaped dicts (with `full_text`/`text`, `verdict`).
    Returns the prompt text plus the numbered citation table.
    """
    lines: list[str] = []
    citations: list[Citation] = []
    n = 0
    for i, check in enumerate(checks, start=1):
        v = check.get("verdict") or {}
        text = (check.get("text") or check.get("full_text") or check.get("text_preview") or "").strip()
        title = check.get("source_title")
        header = f"## Check {i}"
        if check.get("url"):
            header += f" (link: {check['url']}{' | ' + title if title else ''})"
        lines.append(header)
        lines.append(f"Text checked: {text[:1500]}{'…' if len(text) > 1500 else ''}")
        if not v:
            lines.append(f"Status: {check.get('status', 'unknown')} (no verdict)")
            continue
        overall = OVERALL_WORDS.get(v.get("verdict"), v.get("verdict"))
        lines.append(f"Overall verdict: {overall} ({round((v.get('confidence') or 0) * 100)}% confidence)")
        for j, cv in enumerate(v.get("claim_verdicts") or [], start=1):
            lines.append(f"- Claim {i}.{j}: \"{cv.get('claim', '')}\" -> "
                         f"{VERDICT_WORDS.get(cv.get('verdict'), cv.get('verdict'))}")
            snippets = cv.get("evidence_snippets") or []
            sources = cv.get("sources") or []
            urls = cv.get("urls") or []
            for k, snip in enumerate(snippets):
                if n >= MAX_SOURCES:
                    break
                n += 1
                src = sources[k] if k < len(sources) else ""
                url = urls[k] if k < len(urls) else ""
                snip = " ".join(str(snip).split())[:MAX_SNIPPET_CHARS]
                citations.append(Citation(n, cv.get("claim", ""), src, url, snip))
                lines.append(f"    [{n}] ({src or 'unknown source'}) {snip}")
        lines.append("")
    return QAContext(text="\n".join(lines).strip(), citations=citations)


# ── Extractive fallback ─────────────────────────────────────────────────────────

_WORD = re.compile(r"[\w']+", re.UNICODE)
_STOP = {"a", "an", "the", "is", "are", "was", "were", "be", "been", "of", "to", "in", "on", "for", "and", "or", "but", "with", "what", "why", "how", "who", "whom", "which", "that", "this", "these", "those", "it", "its", "do", "does", "did", "can", "could", "should", "would", "about", "from", "by", "as", "at", "not", "no", "yes", "i", "you", "me", "my", "your", "we", "our", "they", "their", "there", "here", "so", "than", "then"}


def _tokens(s: str) -> set[str]:
    return {w for w in (t.lower() for t in _WORD.findall(s or "")) if w not in _STOP and len(w) > 2}


def extractive_answer(question: str, ctx: QAContext, reason: str) -> QAResult:
    q = _tokens(question)
    scored = []
    for c in ctx.citations:
        overlap = len(q & (_tokens(c.snippet) | _tokens(c.claim)))
        scored.append((overlap, -c.n, c))
    scored.sort(reverse=True)
    top = [c for overlap, _, c in scored if overlap > 0][:3] or [c for *_, c in scored][:3]
    if not top:
        return QAResult(
            answer=(f"{reason} This chat also has no evidence to quote yet: none of its "
                    "checks found sources for their claims. Try checking another article "
                    "or a more specific claim."),
            citations=[], mode="extractive")
    bullets = "\n".join(f"- [{c.n}] {c.source or 'source'}: {c.snippet[:240]}" for c in top)
    answer = (f"{reason} Here is the evidence from this chat that looks most relevant "
              f"to your question:\n{bullets}")
    return QAResult(answer=answer, citations=[c.to_dict() for c in top], mode="extractive")


# ── LLM ─────────────────────────────────────────────────────────────────────────

def _cited(answer: str, ctx: QAContext) -> list[dict[str, Any]]:
    used = {int(m) for m in re.findall(r"\[(\d{1,3})\]", answer)}
    return [c.to_dict() for c in ctx.citations if c.n in used]


def _history_messages(history: list[tuple[str, str]]) -> list[dict[str, str]]:
    msgs: list[dict[str, str]] = []
    for q, a in history[-6:]:
        msgs.append({"role": "user", "content": q})
        msgs.append({"role": "assistant", "content": a})
    return msgs


def _ask_anthropic(question: str, ctx: QAContext, history: list[tuple[str, str]]) -> str | None:
    import anthropic

    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY or None)
    model = getattr(settings, "QA_MODEL", "claude-opus-5-5")
    messages = _history_messages(history) + [{
        "role": "user",
        "content": f"<checks>\n{ctx.text}\n</checks>\n\nQuestion: {question}",
    }]
    kwargs = {
        "model": model,
        "max_tokens": 16000,
        "system": [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        "messages": messages,
    }
    try:
        # Server-side refusal fallback: if the model declines (e.g. a
        # misclassified claim about a sensitive topic), the API re-runs the
        # request on a fallback model in the same call.
        resp = client.beta.messages.create(
            betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
    except TypeError:
        # Older SDK without the `fallbacks` parameter.
        resp = client.messages.create(**kwargs)

    if resp.stop_reason == "refusal":
        log.info("qa: model declined (%s)", getattr(resp, "stop_details", None))
        return None
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
    return text or None


def _ask_openai(question: str, ctx: QAContext, history: list[tuple[str, str]]) -> str | None:
    from openai import OpenAI

    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + _history_messages(history) + [{
        "role": "user", "content": f"<checks>\n{ctx.text}\n</checks>\n\nQuestion: {question}"}]
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    return (resp.choices[0].message.content or "").strip() or None


def answer_question(question: str, checks: list[dict[str, Any]],
                    history: list[tuple[str, str]] | None = None) -> QAResult:
    ctx = build_context(checks)
    history = history or []
    provider = getattr(settings, "LLM_PROVIDER", "anthropic") if settings else "anthropic"
    has_key = bool(settings and (
        (provider == "anthropic" and settings.ANTHROPIC_API_KEY
         and not settings.ANTHROPIC_API_KEY.startswith("sk-ant-...")) or
        (provider == "openai" and settings.OPENAI_API_KEY
         and not settings.OPENAI_API_KEY.startswith("sk-..."))
    ))
    if not getattr(settings, "QA_ENABLED", True) or not has_key:
        return extractive_answer(question, ctx, "Follow-up answers need an AI model, which isn't set up on this server.")

    try:
        if provider == "openai":
            answer = _ask_openai(question, ctx, history)
        else:
            answer = _ask_anthropic(question, ctx, history)
    except Exception as exc:
        log.warning("qa: LLM call failed: %s", exc)
        return extractive_answer(question, ctx, "I couldn't reach the AI model just now.")

    if not answer:
        return extractive_answer(question, ctx, "I couldn't write an answer to that.")
    return QAResult(answer=answer, citations=_cited(answer, ctx), mode="llm")


def dumps_citations(citations: list[dict[str, Any]]) -> str:
    return json.dumps(citations, ensure_ascii=False)
