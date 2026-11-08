"""
One place for the pipeline's LLM calls (claim extraction, translation).

Why a helper: current Claude models think before answering, so a response's
first content block can be a thinking block, and thinking counts toward
`max_tokens`. Reading `content[0].text` with a small token cap (the old
pattern) returns nothing or a truncated reply — and the caller's except-branch
then silently falls back to the weaker local path. This helper:

  * reads only the text blocks,
  * leaves room for thinking + the answer (max_tokens 16000+, streamed),
  * treats a refusal or a truncated reply as an error (the caller records a
    degradation instead of trusting partial output),
  * enables server-side refusal fallback where the SDK supports it,
  * extracts the JSON object even if the model wraps it in prose or fences.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

log = logging.getLogger(__name__)

try:
    from config import settings
except ImportError:  # pragma: no cover
    settings = None


class LLMError(RuntimeError):
    pass


def llm_configured() -> bool:
    if settings is None:
        return False
    provider = getattr(settings, "LLM_PROVIDER", "anthropic")
    key = (settings.ANTHROPIC_API_KEY if provider == "anthropic" else settings.OPENAI_API_KEY) or ""
    return bool(key) and not key.endswith("...")


def _parse_json(raw: str) -> dict[str, Any]:
    raw = (raw or "").strip()
    try:
        return json.loads(raw)
    except ValueError:
        pass
    m = re.search(r"\{.*\}", raw, re.DOTALL)   # tolerate ```json fences or a lead-in sentence
    if not m:
        raise LLMError("LLM reply contained no JSON object")
    return json.loads(m.group(0))


def complete_json(prompt: str, *, max_tokens: int = 16000) -> dict[str, Any]:
    """Send one user prompt; return the JSON object in the reply."""
    provider = getattr(settings, "LLM_PROVIDER", "anthropic")
    model = getattr(settings, "LLM_MODEL", "claude-opus-5-5")

    if provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        kwargs = {"model": model, "max_tokens": max_tokens,
                      "messages": [{"role": "user", "content": prompt}]}
        # Streamed: long translations need a large max_tokens, which the SDK
        # rejects for non-streaming requests (HTTP timeout risk).
        try:
            with client.beta.messages.stream(
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs) as stream:
                msg = stream.get_final_message()
        except TypeError:  # older SDK without `fallbacks`
            with client.messages.stream(**kwargs) as stream:
                msg = stream.get_final_message()

        if msg.stop_reason == "refusal":
            raise LLMError(f"model declined ({getattr(msg, 'stop_details', None)})")
        if msg.stop_reason == "max_tokens":
            raise LLMError("reply truncated at max_tokens")
        raw = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        return _parse_json(raw)

    if provider == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        return _parse_json(resp.choices[0].message.content)

    raise LLMError(f"Unknown LLM provider: {provider}")
