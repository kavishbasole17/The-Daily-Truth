"""
Conversational follow-ups with continuity.

A chat message is answered with everything the conversation already holds —
every check (claims, verdicts, numbered evidence) and every earlier exchange —
and, when the user states or asks about something the evidence doesn't cover,
or pushes back on a finding with a new specific fact, by *running a new check*
on that statement through the real verification pipeline.

The user can also focus a message on one finding ("Discuss" on a claim or on
a whole verdict); the focus is put in front of the model so the reply stays on
that finding, and "what's your verdict?" gets one clear call.

Two engines:
  llm        — Claude with a `verify_claim` tool (manual tool loop, max 2 checks
               per reply). Claude decides when to check, then writes the reply.
  heuristic  — no LLM configured: a rule-based router decides between running
               a check (statements, "is it true that…", "verify …") and quoting
               the most relevant evidence, and replies from a template.

`verify` is injected by the caller (api/routers/conversations.py) so this
module never touches the database.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from api.qa import (
    Citation, OVERALL_WORDS, QAContext, VERDICT_WORDS, build_context, extractive_answer,
)

log = logging.getLogger(__name__)

try:
    from config import settings
except ImportError:  # pragma: no cover
    settings = None

MAX_CHECKS_PER_REPLY = 2
MAX_TOOL_ROUNDS = 4

# verify(statement) -> ArticleOut-shaped dict of the finished check (with "text")
VerifyFn = Callable[[str], dict[str, Any]]


@dataclass
class ChatReply:
    answer: str
    citations: list[dict[str, Any]]
    mode: str                                         # llm | heuristic
    article_ids: list[int] = field(default_factory=list)


SYSTEM_PROMPT = """You are Paper Trail, the assistant inside The Daily Truth, a fact-checking app. You are in an ongoing conversation with a user about news they have checked. You can see every check in this chat: the text checked, the overall verdict, each extracted claim with its verdict, and numbered evidence snippets. You can also see the earlier conversation.

How to respond:
- Ground everything in the evidence. Cite snippets by number in square brackets, like [3], right after the statement they support. Never present outside knowledge as something the evidence shows; if you add general context, say it's general context.
- Verdict words mean: supported = the evidence agrees; contradicted = the evidence disagrees; inconclusive = evidence was found but doesn't settle it; could not be verified = no relevant evidence was found, which is not evidence that a claim is false.
- You have a verify_claim tool that runs the full verification pipeline (evidence search across news and reference sources, then NLI) on one factual statement. Use it when the user asserts or asks about a specific factual claim that the existing evidence doesn't already cover, or pushes back on a finding with a new specific fact, a different number, date or source. Pass a single, self-contained, checkable statement (resolve pronouns; include who, what, where, when). Don't use it for questions the existing evidence answers, for opinions or predictions, or to re-check a claim that was already checked. At most two calls per reply.
- When the user asks for your verdict, or asks what you think about a finding, give one clear call (Supported, Contradicted, or Unclear) for that specific finding, say which evidence it rests on, and say what new evidence would change it. If the user offers a counter-argument, take it seriously: check it if it is checkable, then say whether it changes the verdict and why.
- If a message is focused on a specific finding, keep the reply about that finding.
- Be conversational and brief: a few sentences, or a short list when comparing. Answer in the language the user writes in."""

VERIFY_TOOL = {
    "name": "verify_claim",
    "description": (
        "Run the full fact-checking pipeline on one factual statement: searches trusted news "
        "and reference sources for evidence and judges whether each piece supports or "
        "contradicts it. Returns the verdict, confidence, and numbered evidence you can cite. "
        "Takes several seconds, so only call it for a specific, checkable claim the existing "
        "evidence in the chat does not already cover."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "statement": {
                "type": "string",
                "description": "A single self-contained factual statement, e.g. "
                               "'The Eiffel Tower was completed in 1889.'",
            },
        },
        "required": ["statement"],
        "additionalProperties": False,
    },
    "strict": True,
}


# ── Focus ───────────────────────────────────────────────────────────────────────

def describe_focus(focus: Optional[dict], checks: list[dict[str, Any]]) -> str:
    """A plain-language pointer to the finding the user is discussing."""
    if not focus:
        return ""
    aid = focus.get("article_id")
    for i, check in enumerate(checks, start=1):
        if check.get("id") != aid:
            continue
        v = check.get("verdict") or {}
        idx = focus.get("claim_index")
        cvs = v.get("claim_verdicts") or []
        if idx is not None and 0 <= idx < len(cvs):
            cv = cvs[idx]
            return (f"The user is asking specifically about Check {i}, Claim {i}.{idx + 1}: "
                    f"\"{cv.get('claim', '')}\" (found: "
                    f"{VERDICT_WORDS.get(cv.get('verdict'), cv.get('verdict'))}).")
        return (f"The user is asking specifically about the overall verdict of Check {i} "
                f"({OVERALL_WORDS.get(v.get('verdict'), v.get('verdict'))}).")
    return ""


def focused_claim(focus: Optional[dict], checks: list[dict[str, Any]]) -> Optional[dict]:
    if not focus or focus.get("claim_index") is None:
        return None
    for check in checks:
        if check.get("id") == focus.get("article_id"):
            cvs = (check.get("verdict") or {}).get("claim_verdicts") or []
            idx = focus["claim_index"]
            return cvs[idx] if 0 <= idx < len(cvs) else None
    return None


# ── New checks feed the citation table ──────────────────────────────────────────

def add_check_to_context(ctx: QAContext, check: dict[str, Any]) -> str:
    """Append a fresh check's evidence to the citation table; return a tool-result summary."""
    v = check.get("verdict") or {}
    if check.get("status") != "completed" or not v:
        return json.dumps({"status": "failed",
                           "reason": check.get("error") or "The check could not be completed."})
    n = max((c.n for c in ctx.citations), default=0)
    claims_out = []
    for cv in v.get("claim_verdicts") or []:
        evidence = []
        sources, urls = cv.get("sources") or [], cv.get("urls") or []
        for k, snip in enumerate(cv.get("evidence_snippets") or []):
            n += 1
            src = sources[k] if k < len(sources) else ""
            url = urls[k] if k < len(urls) else ""
            snip = " ".join(str(snip).split())[:600]
            ctx.citations.append(Citation(n, cv.get("claim", ""), src, url, snip))
            evidence.append({"n": n, "source": src, "snippet": snip})
        claims_out.append({
            "claim": cv.get("claim"),
            "verdict": VERDICT_WORDS.get(cv.get("verdict"), cv.get("verdict")),
            "confidence": round(cv.get("confidence") or 0, 2),
            "evidence": evidence,
        })
    return json.dumps({
        "status": "completed",
        "overall_verdict": OVERALL_WORDS.get(v.get("verdict"), v.get("verdict")),
        "confidence": round(v.get("confidence") or 0, 2),
        "claims": claims_out,
        "limitations": [d.get("detail") for d in v.get("degradations") or []][:3],
    }, ensure_ascii=False)


def _cited(answer: str, ctx: QAContext) -> list[dict[str, Any]]:
    used = {int(m) for m in re.findall(r"\[(\d{1,3})\]", answer)}
    return [c.to_dict() for c in ctx.citations if c.n in used]


# ── LLM engine ──────────────────────────────────────────────────────────────────

def _llm_available() -> bool:
    if not settings or not getattr(settings, "QA_ENABLED", True):
        return False
    if getattr(settings, "LLM_PROVIDER", "anthropic") != "anthropic":
        return False
    key = settings.ANTHROPIC_API_KEY or ""
    return bool(key) and not key.startswith("sk-ant-...")


def _history_messages(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = []
    for h in history[-8:]:
        msgs.append({"role": "user", "content": h["question"]})
        msgs.append({"role": "assistant", "content": h["answer"] or "(no reply)"})
    return msgs


def _create(client, **kwargs):
    try:
        return client.beta.messages.create(
            betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
    except TypeError:  # older SDK without `fallbacks`
        return client.messages.create(**kwargs)


def _llm_reply(message: str, focus_text: str, ctx: QAContext,
               history: list[dict[str, Any]], verify: VerifyFn,
               on_stage: Callable[[str], None]) -> Optional[ChatReply]:
    import anthropic

    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    model = getattr(settings, "QA_MODEL", "claude-opus-5-5")

    user_turn = f"<checks>\n{ctx.text or '(no finished checks yet)'}\n</checks>\n\n"
    if focus_text:
        user_turn += f"<focus>{focus_text}</focus>\n\n"
    user_turn += f"User: {message}"

    messages: list[dict[str, Any]] = _history_messages(history) + [
        {"role": "user", "content": user_turn}]
    system = [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]
    article_ids: list[int] = []
    checks_run = 0
    resp = None

    for _ in range(MAX_TOOL_ROUNDS):
        resp = _create(client, model=model, max_tokens=16000, system=system,
                       tools=[VERIFY_TOOL], messages=messages)
        if resp.stop_reason == "refusal":
            log.info("chat: model declined (%s)", getattr(resp, "stop_details", None))
            return None
        if resp.stop_reason == "pause_turn":
            messages.append({"role": "assistant", "content": resp.content})
            continue
        if resp.stop_reason != "tool_use":
            break

        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for block in resp.content:
            if getattr(block, "type", "") != "tool_use":
                continue
            statement = str((block.input or {}).get("statement", "")).strip()
            if block.name != "verify_claim" or not statement:
                content, is_error = "Unknown tool or empty statement.", True
            elif checks_run >= MAX_CHECKS_PER_REPLY:
                content, is_error = ("Check limit for this reply reached; answer with the "
                                     "evidence you have."), True
            else:
                checks_run += 1
                on_stage("verifying")
                check = verify(statement[:1000])
                if check.get("id"):
                    article_ids.append(check["id"])
                content, is_error = add_check_to_context(ctx, check), False
                on_stage("writing")
            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": content, "is_error": is_error})
        messages.append({"role": "user", "content": results})

    if resp is None:
        return None
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
    if not text:
        return None
    return ChatReply(answer=text, citations=_cited(text, ctx), mode="llm", article_ids=article_ids)


# ── Heuristic engine ────────────────────────────────────────────────────────────

_ASK_TO_CHECK = re.compile(
    r"^\s*(?:(?:can you |could you |please )?(?:verify|check|fact[- ]?check)(?: that| whether| if)?"
    r"|is it (?:true|correct|accurate|right) (?:that )?|what about(?: the claim that)?|but )\s*[:,]?\s*",
    re.I)
_ABOUT_EXISTING = re.compile(
    r"\b(claim|claims|source|sources|evidence|verdict|why|how come|explain|which|confidence|that one)\b",
    re.I)
_ASKS_FOR_VERDICT = re.compile(
    r"\b(your verdict|what do you think|so is it|bottom line|true or false|real or fake|final answer)\b",
    re.I)


def route_message(message: str) -> tuple[str, str]:
    """
    -> ("verify", statement) | ("verdict", "") | ("answer", "")
    Statements and explicit "is it true that…" requests get checked; questions
    about the existing findings get answered from the evidence.
    """
    m = message.strip()
    if _ASKS_FOR_VERDICT.search(m):
        return "verdict", ""
    trigger = _ASK_TO_CHECK.match(m)
    if trigger:
        rest = m[trigger.end():].strip().rstrip("?").strip()
        if len(rest.split()) >= 3:
            return "verify", rest[0].upper() + rest[1:] + ("" if rest.endswith(".") else ".")
    is_question = m.endswith("?") or re.match(r"^(who|what|when|where|why|how|which|is|are|was|were|do|does|did|can|could|should)\b", m, re.I)
    if not is_question and len(m.split()) >= 5 and not _ABOUT_EXISTING.search(m):
        return "verify", m  # a counter-claim / new statement
    return "answer", ""


_LEAD_IN = re.compile(
    r"^(?:(?:but|well|no|hmm|ok|okay|actually)[,]?\s+)*"
    r"(?:(?:i\s+(?:read|heard|saw|think|believe|thought)|someone\s+(?:said|told me)|they\s+say|"
    r"apparently|i'?m\s+(?:pretty\s+)?sure)(?:\s+that)?\s+)?",
    re.I)
_FILLER = re.compile(r"\b(actually|really|definitely|in fact)\s+", re.I)
_PRONOUN = re.compile(r"^(it|this|that|they|he|she)\b", re.I)
_SUBJECT = re.compile(r"^(.{3,80}?)\s+(?:is|was|were|are|has|have|had|will|would|did|does|do)\b", re.I)


def _reference_text(focus: Optional[dict], checks: list[dict[str, Any]]) -> str:
    """The claim (or checked text) a pronoun in the user's message most likely means."""
    cv = focused_claim(focus, checks)
    if cv:
        return cv.get("claim", "")
    target = next((c for c in checks if focus and c.get("id") == focus.get("article_id")),
                  checks[-1] if checks else None)
    if not target:
        return ""
    cvs = (target.get("verdict") or {}).get("claim_verdicts") or []
    if cvs:
        return cvs[0].get("claim", "")
    return re.split(r"(?<=[.!?])\s", (target.get("text") or "").strip(), maxsplit=1)[0]


def resolve_statement(statement: str, focus: Optional[dict], checks: list[dict[str, Any]]) -> str:
    """
    Turn a conversational pushback into a self-contained, checkable claim:
    "I read it was actually completed in 1887" + finding "The Eiffel Tower is
    in Paris." -> "The Eiffel Tower was completed in 1887."
    """
    s = _LEAD_IN.sub("", statement.strip(), count=1)
    s = _FILLER.sub("", s).strip()
    m = _PRONOUN.match(s)
    if m:
        subj = _SUBJECT.match(_reference_text(focus, checks).strip())
        if subj:
            s = subj.group(1) + s[m.end():]
    s = s.rstrip("?!. ").strip()
    return (s[0].upper() + s[1:] + ".") if s else statement


def _verdict_sentence(cv: dict[str, Any]) -> str:
    word = VERDICT_WORDS.get(cv.get("verdict"), cv.get("verdict"))
    call = {"supported": "Supported", "contradicted": "Contradicted"}.get(word, "Unclear")
    return f"**{call}** — \"{cv.get('claim')}\" was {word} ({round((cv.get('confidence') or 0) * 100)}% confidence)."


def _heuristic_reply(message: str, focus: Optional[dict], ctx: QAContext,
                     checks: list[dict[str, Any]], verify: VerifyFn,
                     on_stage: Callable[[str], None]) -> ChatReply:
    route, statement = route_message(message)

    if route == "verify":
        statement = resolve_statement(statement, focus, checks)
        on_stage("verifying")
        check = verify(statement[:1000])
        add_check_to_context(ctx, check)
        ids = [check["id"]] if check.get("id") else []
        v = check.get("verdict") or {}
        if check.get("status") != "completed" or not v:
            return ChatReply(f"I tried to check \"{statement}\" but couldn't finish: "
                             f"{check.get('error') or 'the check failed'}.", [], "heuristic", ids)
        overall = OVERALL_WORDS.get(v.get("verdict"), v.get("verdict"))
        lines = [f"I checked \"{statement}\". Overall: **{overall}** "
                 f"({round((v.get('confidence') or 0) * 100)}% confidence)."]
        cites = []
        new = [c for c in ctx.citations if c.claim in {cv.get('claim') for cv in v.get('claim_verdicts') or []}]
        for cv in v.get("claim_verdicts") or []:
            nums = [c.n for c in new if c.claim == cv.get("claim")][:3]
            lines.append("- " + _verdict_sentence(cv) + "".join(f" [{n}]" for n in nums))
        if not v.get("claim_verdicts"):
            lines.append("No checkable claim could be extracted from that statement.")
        answer = "\n".join(lines)
        cites = _cited(answer, ctx)
        return ChatReply(answer, cites, "heuristic", ids)

    if route == "verdict":
        cv = focused_claim(focus, checks)
        if cv:
            nums = [c.n for c in ctx.citations if c.claim == cv.get("claim")][:3]
            answer = ("My verdict on this finding: " + _verdict_sentence(cv)
                      + "".join(f" [{n}]" for n in nums)
                      + "\nNew, specific evidence (a dated report or an official source) could change it — "
                        "paste it here and I'll check it.")
            return ChatReply(answer, _cited(answer, ctx), "heuristic")
        finished = [c for c in checks if c.get("verdict")]
        if finished:
            v = finished[-1]["verdict"]
            lines = [f"My verdict on the latest check: **{OVERALL_WORDS.get(v['verdict'], v['verdict'])}** "
                     f"({round((v.get('confidence') or 0) * 100)}% confidence)."]
            lines += ["- " + _verdict_sentence(cv) for cv in (v.get("claim_verdicts") or [])[:5]]
            return ChatReply("\n".join(lines), [], "heuristic")

    focus_cv = focused_claim(focus, checks)
    query = f"{message} {focus_cv.get('claim', '')}" if focus_cv else message
    ex = extractive_answer(query, ctx, "Here's what this chat's evidence says.")
    return ChatReply(ex.answer, ex.citations, "heuristic")


# ── Entry point ─────────────────────────────────────────────────────────────────

def chat_reply(message: str, focus: Optional[dict], checks: list[dict[str, Any]],
               history: list[dict[str, Any]], verify: VerifyFn,
               on_stage: Callable[[str], None] = lambda s: None) -> ChatReply:
    ctx = build_context(checks)
    focus_text = describe_focus(focus, checks)
    if _llm_available():
        try:
            reply = _llm_reply(message, focus_text, ctx, history, verify, on_stage)
            if reply:
                return reply
        except Exception as exc:
            log.warning("chat: LLM reply failed, using heuristic: %s", exc)
    return _heuristic_reply(message, focus, ctx, checks, verify, on_stage)
