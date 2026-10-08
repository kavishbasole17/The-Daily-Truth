"""
Geocoder tests — normalisation, caching, and the offline/failure contracts.
`requests` is always patched; nothing here touches the network.
"""
from __future__ import annotations

import pytest

from paper_trail_core.nlp import geocoder as G

RAW = {
    "place_id": 1, "lat": "18.5366", "lon": "73.8932", "category": "place",
    "type": "locality", "place_rank": 25, "importance": 0.0667, "addresstype": "locality",
    "name": "Koregaon Park",
    "display_name": "Koregaon Park, Ghorpuri, Pune, Pune City Subdistrict, Pune, Maharashtra, 411001, India",
    "address": {"locality": "Koregaon Park", "suburb": "Ghorpuri", "city": "Pune",
                "state_district": "Pune", "state": "Maharashtra", "country": "India",
                "country_code": "in"},
}


class _Resp:
    def __init__(self, payload, status=200):
        self._p, self.status_code = payload, status

    def json(self):
        return self._p

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")


@pytest.fixture(autouse=True)
def _live_backend(monkeypatch):
    from config import settings
    monkeypatch.setattr(settings, "GEOCODER_BACKEND", "nominatim")
    monkeypatch.setattr(G, "_throttle", lambda: None)
    monkeypatch.setattr(G.time, "sleep", lambda *_: None)


def test_result_is_normalised():
    out = G._normalise(RAW)
    assert out["level"] == "locality" and out["city"] == "Pune"
    assert out["state"] == "Maharashtra" and out["country_code"] == "in"
    assert out["lat"] == pytest.approx(18.5366)


@pytest.mark.parametrize("addresstype,rank,expected", [
    ("country", 4, "country"), ("state", 8, "state"), ("county", 12, "district"),
    ("city", 16, "city"), ("suburb", 20, "locality"), ("road", 26, "street"),
    ("", 4, "country"), ("", 8, "state"), ("", 30, "site"), ("amenity", 30, "site"),
])
def test_level_mapping(addresstype, rank, expected):
    assert G.level_of({"addresstype": addresstype, "place_rank": rank}) == expected


def test_lookup_is_cached(monkeypatch):
    calls = []

    def _get(*a, **k):
        calls.append(1)
        return _Resp([RAW])
    monkeypatch.setattr("requests.get", _get)

    first = G.geocode("Koregaon Park")
    second = G.geocode("  koregaon   park ")          # same key after normalisation
    assert first == second and len(calls) == 1


def test_no_result_is_cached_too(monkeypatch):
    calls = []
    monkeypatch.setattr("requests.get", lambda *a, **k: calls.append(1) or _Resp([]))
    assert G.geocode("Zzyzxville") == []
    assert G.geocode("Zzyzxville") == []
    assert len(calls) == 1


def test_offline_backend_never_touches_the_network(monkeypatch):
    from config import settings
    monkeypatch.setattr(settings, "GEOCODER_BACKEND", "offline")

    def _boom(*a, **k):
        raise AssertionError("network call in offline mode")
    monkeypatch.setattr("requests.get", _boom)
    assert G.geocode("Pune") == []


def test_off_backend_returns_nothing(monkeypatch):
    from config import settings
    monkeypatch.setattr(settings, "GEOCODER_BACKEND", "off")
    assert G.geocode("Pune") == []


def test_network_failure_is_distinguished_from_no_such_place(monkeypatch):
    def _boom(*a, **k):
        raise ConnectionError("down")
    monkeypatch.setattr("requests.get", _boom)
    with pytest.raises(G.GeocoderUnavailable):
        G.geocode("Pune")


def test_rate_limit_response_is_treated_as_unavailable(monkeypatch):
    monkeypatch.setattr("requests.get", lambda *a, **k: _Resp([], status=429))
    with pytest.raises(G.GeocoderUnavailable):
        G.geocode("Pune")


def test_failures_are_not_cached(monkeypatch):
    state = {"n": 0}

    def _get(*a, **k):
        state["n"] += 1
        if state["n"] <= 2:                    # both attempts of the first lookup fail
            raise ConnectionError("down")
        return _Resp([RAW])
    monkeypatch.setattr("requests.get", _get)

    with pytest.raises(G.GeocoderUnavailable):
        G.geocode("Koregaon Park")
    assert G.geocode("Koregaon Park")[0]["name"] == "Koregaon Park"


def test_request_identifies_itself_and_asks_for_english(monkeypatch):
    seen = {}

    def _get(url, params=None, headers=None, timeout=None):
        seen.update(params=params, headers=headers)
        return _Resp([])
    monkeypatch.setattr("requests.get", _get)
    G.geocode("Pune")
    assert seen["params"]["accept-language"] == "en"
    assert "PaperTrail" in seen["headers"]["User-Agent"]
