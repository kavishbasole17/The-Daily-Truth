"""
Geocoding via OpenStreetMap Nominatim, with an on-disk cache.

Free and keyless, which is why it is the default — but it comes with a usage
policy this module enforces rather than hopes for: at most one request per
second, an identifying User-Agent, and no repeat queries (every answer is
cached, including "no result", so a place is looked up once ever).

The geocoder is *only* ever asked "what could this name refer to?". Deciding
which candidate is right is the locator's job (it has the surrounding text),
and is deliberately not done here.

Backends (config.GEOCODER_BACKEND):
  nominatim — live lookups through the cache.
  offline   — never touches the network; cache hits only.
  off       — returns nothing.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import threading
import time
from typing import Any

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

log = logging.getLogger(__name__)

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
HTTP_TIMEOUT = 8
MIN_INTERVAL_S = 1.1            # Nominatim policy is 1 req/s; leave a margin
CACHE_TTL_DAYS = 180            # places do not move

_lock = threading.Lock()
_last_call = 0.0

# Nominatim `addresstype` -> our level, most specific first.
_LEVEL_BY_ADDRESSTYPE = {
    "country": "country",
    "state": "state", "province": "state", "region": "state", "territory": "state",
    "state_district": "district", "county": "district", "district": "district",
    "city": "city", "town": "city", "village": "city", "municipality": "city",
    "suburb": "locality", "neighbourhood": "locality", "quarter": "locality",
    "residential": "locality", "locality": "locality", "hamlet": "locality",
    "city_district": "locality", "borough": "locality", "isolated_dwelling": "locality",
    "road": "street", "street": "street", "highway": "street", "footway": "street",
}
LEVEL_ORDER = ["site", "street", "locality", "city", "district", "state", "country"]


class GeocoderUnavailable(RuntimeError):
    """The live geocoder could not be reached (as opposed to 'no such place')."""


def _settings():
    try:
        from config import settings
        return settings
    except ImportError:  # pragma: no cover
        return None


def _backend() -> str:
    s = _settings()
    return getattr(s, "GEOCODER_BACKEND", "nominatim") if s else "nominatim"


def _cache_dir() -> str:
    s = _settings()
    d = getattr(s, "GEOCODER_CACHE_DIR", "cache/geo") if s else "cache/geo"
    return d if os.path.isabs(d) else os.path.join(_ROOT, d)


def _key(query: str, countrycodes: str | None) -> str:
    # "v2": results now carry alternative names (namedetails), so v1 entries are stale.
    raw = f"v2|{' '.join(query.lower().split())}|{(countrycodes or '').lower()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _path(key: str) -> str:
    return os.path.join(_cache_dir(), key[:2], f"{key}.json")


def _cache_get(key: str) -> list[dict[str, Any]] | None:
    path = _path(key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    if (time.time() - payload.get("fetched_at", 0)) / 86400.0 > CACHE_TTL_DAYS:
        return None
    return payload.get("results", [])


def _cache_put(key: str, query: str, results: list[dict[str, Any]]) -> None:
    path = _path(key)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"query": query, "fetched_at": time.time(), "results": results},
                      fh, ensure_ascii=False)
        os.replace(tmp, path)
    except OSError as exc:
        log.warning("geocoder: could not write cache (%s)", exc)


def _throttle() -> None:
    global _last_call
    with _lock:
        wait = MIN_INTERVAL_S - (time.time() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.time()


def level_of(raw: dict[str, Any]) -> str:
    """Our precision level for one raw Nominatim result."""
    addresstype = (raw.get("addresstype") or "").lower()
    if addresstype in _LEVEL_BY_ADDRESSTYPE:
        return _LEVEL_BY_ADDRESSTYPE[addresstype]
    rank = int(raw.get("place_rank") or 0)
    if rank <= 4:
        return "country"
    if rank <= 10:
        return "state"
    if rank <= 14:
        return "district"
    if rank <= 18:
        return "city"
    if rank <= 24:
        return "locality"
    if rank <= 27:
        return "street"
    return "site"


def _normalise(raw: dict[str, Any]) -> dict[str, Any]:
    addr = raw.get("address") or {}
    city = (addr.get("city") or addr.get("town") or addr.get("village")
            or addr.get("municipality"))
    locality = (addr.get("locality") or addr.get("suburb") or addr.get("neighbourhood")
                or addr.get("quarter") or addr.get("residential") or addr.get("city_district"))
    details = raw.get("namedetails") or {}
    names = sorted({str(v).strip() for v in details.values() if v and str(v).strip()})
    return {
        "name": raw.get("name") or (raw.get("display_name") or "").split(",")[0].strip(),
        # Every name the place is known by ("Aurangabad" for Chhatrapati Sambhajinagar).
        "names": names,
        "display_name": raw.get("display_name", ""),
        "level": level_of(raw),
        "lat": float(raw["lat"]) if raw.get("lat") not in (None, "") else None,
        "lon": float(raw["lon"]) if raw.get("lon") not in (None, "") else None,
        "country": addr.get("country"),
        "country_code": (addr.get("country_code") or "").lower() or None,
        "state": addr.get("state") or addr.get("province") or addr.get("region"),
        "district": addr.get("state_district") or addr.get("county"),
        "city": city,
        "locality": locality,
        "importance": float(raw.get("importance") or 0.0),
        "osm_class": raw.get("category") or raw.get("class"),
        "osm_type": raw.get("type"),
    }


def geocode(
    query: str,
    countrycodes: str | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """
    Candidate places for a name, best-first as ranked by Nominatim.

    Returns [] for "no such place". Raises GeocoderUnavailable when the live
    lookup itself failed, so the caller can tell "nowhere" from "couldn't ask".
    """
    query = " ".join((query or "").split())
    backend = _backend()
    if not query or backend == "off":
        return []

    key = _key(query, countrycodes)
    cached = _cache_get(key)
    if cached is not None:
        return cached[:limit]
    if backend != "nominatim":
        return []

    import requests

    s = _settings()
    headers = {"User-Agent": getattr(s, "GEOCODER_USER_AGENT", "PaperTrail/2.0")}
    params = {"q": query, "format": "jsonv2", "addressdetails": 1,
              "limit": max(limit, 1), "accept-language": "en", "namedetails": 1}
    if countrycodes:
        params["countrycodes"] = countrycodes

    last_exc: Exception | None = None
    for attempt in range(2):
        _throttle()
        try:
            resp = requests.get(NOMINATIM_URL, params=params, headers=headers,
                                timeout=HTTP_TIMEOUT)
            if resp.status_code in (429, 503):
                last_exc = RuntimeError(f"nominatim status {resp.status_code}")
                time.sleep(2.0 * (attempt + 1))
                continue
            resp.raise_for_status()
            results = [_normalise(r) for r in resp.json()]
            _cache_put(key, query, results)
            return results[:limit]
        except Exception as exc:  # network / JSON / HTTP errors
            last_exc = exc
            time.sleep(0.5)

    log.warning("geocoder: lookup failed for %r (%s)", query[:60], last_exc)
    raise GeocoderUnavailable(type(last_exc).__name__ if last_exc else "unknown")
