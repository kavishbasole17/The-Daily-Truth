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
    key = {"anthropic": settings.ANTHROPIC_API_KEY,
           "nvidia": getattr(settings, "NVIDIA_API_KEY", None)}.get(provider, settings.OPENAI_API_KEY) or ""
    return bool(key) and not key.endswith("...")


def active_model() -> str:
    """The model claim extraction and translation actually call."""
    if getattr(settings, "LLM_PROVIDER", "anthropic") == "nvidia":
        return settings.NVIDIA_LLM_MODEL
    return getattr(settings, "LLM_MODEL", "claude-opus-5-5")


def nvidia_client():
    from openai import OpenAI

    return OpenAI(api_key=settings.NVIDIA_API_KEY, base_url=settings.NVIDIA_BASE_URL, timeout=240)


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def nvidia_text(message) -> str:
    """Reply text with any inline reasoning removed (some catalog models emit <think> blocks)."""
    return _THINK.sub("", message.content or "").strip()


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


def _stream_anthropic(client, kwargs: dict[str, Any]):
    """One streamed request; drops the newer parameters on an SDK that predates them."""
    try:
        with client.beta.messages.stream(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs) as stream:
            return stream.get_final_message()
    except TypeError:  # older SDK without `fallbacks` / `output_config`
        legacy = {k: v for k, v in kwargs.items() if k != "output_config"}
        with client.messages.stream(**legacy) as stream:
            return stream.get_final_message()


def complete_json(prompt: str, *, max_tokens: int = 16000, effort: str = "high") -> dict[str, Any]:
    """
    Send one user prompt; return the JSON object in the reply.

    `effort` sets the model's reasoning depth. Current models default to
    `medium`; claim extraction and translation are accuracy-critical, so the
    pipeline asks for `high` unless a caller says otherwise.
    """
    provider = getattr(settings, "LLM_PROVIDER", "anthropic")
    model = getattr(settings, "LLM_MODEL", "claude-opus-5-5")

    if provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        kwargs: dict[str, Any] = {"model": model, "max_tokens": max_tokens,
                                  "messages": [{"role": "user", "content": prompt}]}
        if effort:
            kwargs["output_config"] = {"effort": effort}
        # Streamed: long translations need a large max_tokens, which the SDK
        # rejects for non-streaming requests (HTTP timeout risk).
        msg = _stream_anthropic(client, kwargs)

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

    if provider == "nvidia":
        # No response_format: not every catalog model supports JSON mode, and
        # _parse_json already tolerates fences and lead-in prose.
        resp = nvidia_client().chat.completions.create(
            model=settings.NVIDIA_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.1,
        )
        choice = resp.choices[0]
        if choice.finish_reason == "length":
            raise LLMError("reply truncated at max_tokens")
        return _parse_json(nvidia_text(choice.message))

    raise LLMError(f"Unknown LLM provider: {provider}")
