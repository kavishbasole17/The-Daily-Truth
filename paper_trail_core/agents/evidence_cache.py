"""
On-disk cache for search results.

This is small but load-bearing. It gives:

  * **Determinism** — the same article yields the same evidence yields the same
    verdict. Without it, re-running an evaluation gives different numbers
    because a news index moved underneath you, and no result is reproducible.
  * **A usable evaluation budget** — NewsAPI's free tier allows 100 requests a
    day. Evaluating a few hundred articles is impossible without replay.
  * **Offline tests** — the end-to-end test ships a committed cache directory
    and needs no network at all.

Entries are plain JSON so they can be inspected, diffed and committed.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
import time
from typing import Any

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from config import settings
    _DIR = settings.EVIDENCE_CACHE_DIR
    _TTL_DAYS = settings.EVIDENCE_CACHE_TTL_DAYS
except ImportError:  # pragma: no cover
    _DIR, _TTL_DAYS = "cache/evidence", 7

log = logging.getLogger(__name__)

_RE_WS = re.compile(r"\s+")


def _normalise(query: str) -> str:
    """Case- and whitespace-insensitive key, so trivial variants share an entry."""
    return _RE_WS.sub(" ", query.strip().lower())


def cache_key(provider: str, query: str) -> str:
    return hashlib.sha256(f"{provider}|{_normalise(query)}".encode()).hexdigest()


def _path_for(key: str, base_dir: str | None = None) -> str:
    # Two-level fan-out: a flat directory with tens of thousands of files gets
    # painfully slow to enumerate on Windows.
    root = base_dir or _DIR
    if not os.path.isabs(root):
        root = os.path.join(_ROOT, root)
    return os.path.join(root, key[:2], f"{key}.json")


def get(
    provider: str,
    query: str,
    ttl_days: int | None = None,
    base_dir: str | None = None,
) -> list[dict[str, Any]] | None:
    """
    Cached results, or None on miss/expiry.

    A corrupt or unreadable entry is treated as a miss rather than an error —
    a damaged cache file should cost one refetch, not fail the request.
    """
    ttl = _TTL_DAYS if ttl_days is None else ttl_days
    path = _path_for(cache_key(provider, query), base_dir)
    if not os.path.exists(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("evidence_cache: unreadable entry %s (%s) — treating as miss", path, exc)
        return None

    if ttl > 0:
        age_days = (time.time() - payload.get("fetched_at", 0)) / 86400.0
        if age_days > ttl:
            return None

    return payload.get("results", [])


def put(
    provider: str,
    query: str,
    results: list[dict[str, Any]],
    base_dir: str | None = None,
) -> None:
    """Store results. Written atomically so a crash cannot leave a partial entry."""
    path = _path_for(cache_key(provider, query), base_dir)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "provider": provider,
            "query": query,
            "fetched_at": time.time(),
            "results": results,
        }
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
    except OSError as exc:
        log.warning("evidence_cache: could not write %s (%s)", path, exc)


def stats(base_dir: str | None = None) -> dict[str, Any]:
    root = base_dir or _DIR
    if not os.path.isabs(root):
        root = os.path.join(_ROOT, root)
    if not os.path.isdir(root):
        return {"entries": 0, "dir": root}
    n = sum(len([f for f in files if f.endswith(".json")]) for _, _, files in os.walk(root))
    return {"entries": n, "dir": root}
