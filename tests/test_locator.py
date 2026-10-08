"""
Locator tests — candidate extraction, resolution, claim assignment, scopes.

The geocoder is stubbed (conftest pins GEOCODER_BACKEND=offline anyway), so
these run offline. What they pin down is the *judgement*: which phrases are
considered places, when a place is trusted, and when the locator refuses to
guess.
"""
from __future__ import annotations

import pytest

from paper_trail_core.agents import locator as L
from paper_trail_core.nlp import gazetteer


def _place(name, level, *, city=None, district=None, state=None, locality=None,
           country="India", cc="in", importance=0.5, display=None, lat=1.0, lon=2.0):
    return {
        "name": name, "display_name": display or name, "level": level,
        "lat": lat, "lon": lon, "country": country, "country_code": cc,
        "state": state, "district": district, "city": city, "locality": locality,
        "importance": importance, "osm_class": "place", "osm_type": level,
    }


@pytest.fixture
def fake_geocoder(monkeypatch):
    """Install a dict-backed geocoder: name (lowercase) -> list of results."""
    table: dict[str, list[dict]] = {}
    calls: list[str] = []

    def _geocode(query, countrycodes=None, limit=5):
        calls.append(query)
        return table.get(query.lower(), [])[:limit]

    monkeypatch.setattr(L, "geocode", _geocode)
    table["_calls"] = calls          # type: ignore[assignment]
    return table


# ── Gazetteer ─────────────────────────────────────────────────────────────────

def test_gazetteer_knows_states_cities_and_countries():
    assert gazetteer.lookup("Maharashtra").level == "state"
    assert gazetteer.lookup("pune").state == "Maharashtra"
    assert gazetteer.lookup("India").country_code == "in"
    assert gazetteer.lookup("Atlantis") is None


def test_gazetteer_flags_names_that_are_also_words():
    assert gazetteer.is_ambiguous("Salem")
    assert not gazetteer.is_ambiguous("Pune")


# ── Candidate extraction ──────────────────────────────────────────────────────

def _surfaces(text):
    return {c.surface for c in L.extract_candidates(text)}


def test_place_after_a_cue_is_a_candidate():
    assert "Koregaon Park" in _surfaces("Heavy rain flooded roads at Koregaon Park on Monday.")


def test_capitalised_phrase_without_a_cue_is_not_a_candidate():
    """Without a cue, a capitalised phrase is as likely a person as a place."""
    assert "Narendra Modi" not in _surfaces("Narendra Modi said the scheme would expand.")
    assert "Rajesh Kumar" not in _surfaces("Officials met Rajesh Kumar on Tuesday.")


def test_person_following_a_speech_verb_is_not_a_place():
    assert "Rahul Verma" not in _surfaces("A statement from Rahul Verma said the road would reopen.")


def test_acronyms_are_not_places():
    assert "WHO" not in _surfaces("The report came from WHO officials in Geneva.")


def test_dateline_is_a_candidate_with_a_cue():
    cands = L.extract_candidates("PUNE: A fire broke out in a godown on Tuesday evening.")
    pune = next(c for c in cands if c.surface == "Pune")
    assert pune.dateline and pune.cue


def test_gazetteer_names_must_be_capitalised():
    assert "Goa" not in _surfaces("they said we should goa home early")
    assert "UK" in _surfaces("Officials in the UK confirmed it.")
    assert "UK" not in _surfaces("Officials in the uk confirmed it.")


def test_place_pairs_keep_their_qualifier():
    cands = L.extract_candidates("A fire at Koregaon Park, Pune injured three people.")
    kp = next(c for c in cands if c.surface == "Koregaon Park")
    assert kp.qualifier == "Pune"


def test_gazetteer_mentions_are_counted_once():
    cands = L.extract_candidates("Officials in Hyderabad said the road was closed.")
    assert next(c for c in cands if c.surface == "Hyderabad").count == 1


# ── Resolution ────────────────────────────────────────────────────────────────

def test_precise_place_is_reported_at_its_own_level(fake_geocoder):
    fake_geocoder["koregaon park, pune"] = [
        _place("Koregaon Park", "locality", locality="Koregaon Park", city="Pune",
               district="Pune", state="Maharashtra", importance=0.08)]
    text = "Heavy rain flooded roads at Koregaon Park, Pune on Monday."
    res = L.locate_text(text, [text])
    loc = res.claim_locations[0]
    assert loc["level"] == "locality"
    assert loc["city"] == "Pune" and loc["state"] == "Maharashtra"
    assert loc["confidence"] >= 0.5
    assert loc["source"] == "claim" and loc["method"] == "nominatim"
    assert loc["label"].startswith("Koregaon Park")


def test_context_decides_between_same_named_places(fake_geocoder):
    fake_geocoder["aurangabad"] = [
        _place("Aurangabad", "city", state="Bihar", importance=0.45),
        _place("Aurangabad", "city", state="Maharashtra", importance=0.50),
    ]
    text = "Floods hit Aurangabad in Bihar on Sunday."
    res = L.locate_text(text, [text])
    assert res.claim_locations[0]["state"] == "Bihar"


def test_unresolvable_ambiguity_means_no_location_not_a_guess(fake_geocoder):
    fake_geocoder["springfield"] = [
        _place("Springfield", "city", state="Illinois", country="United States", cc="us", importance=0.52),
        _place("Springfield", "city", state="Missouri", country="United States", cc="us", importance=0.51),
        _place("Springfield", "city", state="Massachusetts", country="United States", cc="us", importance=0.50),
    ]
    text = "Police near Springfield arrested two men on Friday."
    res = L.locate_text(text, [text])
    assert res.claim_locations[0] is None
    assert "Springfield" in res.notes["ambiguous"]


def test_claim_naming_an_unresolved_place_does_not_inherit_the_article_place(fake_geocoder):
    fake_geocoder["springfield"] = [
        _place("Springfield", "city", state="Illinois", country="United States", cc="us", importance=0.52),
        _place("Springfield", "city", state="Missouri", country="United States", cc="us", importance=0.51),
    ]
    text = "PUNE: Police near Springfield arrested two men on Friday."
    claim = "Police near Springfield arrested two men on Friday."
    res = L.locate_text(text, [claim])
    assert res.article_location is not None             # the dateline still resolves
    assert res.claim_locations[0] is None


def test_unknown_names_that_geocode_to_nothing_are_ignored(fake_geocoder):
    text = "Residents of Zzyzxville complained about the noise."
    res = L.locate_text(text, [text])
    assert res.claim_locations == [None]


def test_building_level_match_needs_a_place_word(fake_geocoder):
    """A company or agency must not be pinned to its head office."""
    fake_geocoder["tata motors"] = [_place("Tata Motors", "site", city="Mumbai", importance=0.0001)]
    text = "Workers at Tata Motors went on strike."
    assert L.locate_text(text, [text]).claim_locations == [None]


def test_institution_is_located_at_site_level(fake_geocoder):
    fake_geocoder["university of cambridge"] = [
        _place("University of Cambridge", "site", city="Cambridge", state="England",
               country="United Kingdom", cc="gb", importance=0.0001)]
    text = "Scientists at the University of Cambridge reported a result."
    loc = L.locate_text(text, [text]).claim_locations[0]
    assert loc and loc["level"] == "site" and loc["country_code"] == "gb"


def test_gazetteer_alone_locates_when_the_geocoder_is_silent():
    """Offline (the test default): states and countries still resolve."""
    text = "Officials in Maharashtra said the dam was safe."
    loc = L.locate_text(text, [text]).claim_locations[0]
    assert loc["level"] == "state" and loc["method"] == "gazetteer"
    assert loc["country_code"] == "in" and loc["confidence"] >= 0.5


def test_ambiguous_gazetteer_name_needs_a_cue():
    assert L.locate_text("Salem was a good year for wine.", ["Salem was a good year for wine."]) \
        .claim_locations == [None]


# ── Inheritance ───────────────────────────────────────────────────────────────

def test_claim_without_a_place_inherits_a_dateline_place():
    text = "PUNE: The mayor announced a new bridge. The bridge will cost 40 crore rupees."
    res = L.locate_text(text, ["The bridge will cost 40 crore rupees."])
    loc = res.claim_locations[0]
    assert loc and loc["name"] == "Pune" and loc["source"] == "article"
    assert loc["confidence"] < res.article_location["confidence"]       # discounted


def test_a_single_passing_mention_is_not_the_articles_place():
    text = "The tournament was won by a team that trained in Pune. The final score was 3-1."
    res = L.locate_text(text, ["The final score was 3-1."])
    assert res.article_location is None
    assert res.claim_locations == [None]


def test_user_supplied_region_is_used_for_claims_with_no_place():
    res = L.locate_text("The bridge will cost 40 crore rupees to build.",
                        ["The bridge will cost 40 crore rupees to build."],
                        region_hint="Kerala")
    loc = res.claim_locations[0]
    assert loc and loc["name"] == "Kerala" and loc["source"] == "user"


def test_claim_with_its_own_place_beats_the_hint():
    text = "A fire broke out in Mumbai on Sunday."
    loc = L.locate_text(text, [text], region_hint="Kerala").claim_locations[0]
    assert loc["name"] == "Mumbai" and loc["source"] == "claim"


def test_most_specific_place_in_a_claim_wins(fake_geocoder):
    fake_geocoder["koregaon park"] = [
        _place("Koregaon Park", "locality", locality="Koregaon Park", city="Pune",
               state="Maharashtra", importance=0.08)]
    text = "A fire at Koregaon Park in Pune, Maharashtra injured three people."
    loc = L.locate_text(text, [text]).claim_locations[0]
    assert loc["level"] == "locality"
    assert {a["name"] for a in loc["also"]} >= {"Pune", "Maharashtra"}


# ── Scopes ────────────────────────────────────────────────────────────────────

def test_scopes_widen_from_locality_to_country():
    scopes = L.scopes_for({"name": "Koregaon Park", "level": "locality", "city": "Pune",
                           "state": "Maharashtra", "country": "India", "country_code": "in"})
    assert [s["level"] for s in scopes] == ["locality", "city", "state", "country"]
    assert scopes[0]["term"] == "Koregaon Park Pune"
    assert all(s["country_code"] == "in" for s in scopes)


def test_scopes_start_at_the_places_own_level():
    state = L.scopes_for({"name": "Kerala", "level": "state", "state": "Kerala",
                          "country": "India", "country_code": "in"})
    assert [s["level"] for s in state] == ["state", "country"]
    country = L.scopes_for({"name": "India", "level": "country", "country": "India",
                            "country_code": "in"})
    assert [s["level"] for s in country] == ["country"]


def test_scopes_have_no_duplicate_terms():
    scopes = L.scopes_for({"name": "Delhi", "level": "state", "state": "Delhi",
                           "country": "India", "country_code": "in"})
    terms = [s["term"].lower() for s in scopes]
    assert len(terms) == len(set(terms))


# ── Node ──────────────────────────────────────────────────────────────────────

def test_node_returns_one_entry_per_claim():
    out = L.locate_node({"article_text": "Officials in Kerala said so.",
                         "claims": ["Officials in Kerala said so.", "Something else entirely."]})
    assert len(out["claim_locations"]) == 2
    assert out["claim_locations"][0]["name"] == "Kerala"
    assert out["claim_locations"][1] is None


def test_node_is_a_noop_when_region_aware_is_off(monkeypatch):
    from config import settings
    monkeypatch.setattr(settings, "REGION_AWARE", False)
    out = L.locate_node({"article_text": "Officials in Kerala said so.",
                         "claims": ["Officials in Kerala said so."]})
    assert out["claim_locations"] == [None]


def test_node_reports_an_unreachable_geocoder_as_info(monkeypatch):
    from paper_trail_core.nlp.geocoder import GeocoderUnavailable

    def _down(*_a, **_k):
        raise GeocoderUnavailable("ConnectionError")
    monkeypatch.setattr(L, "geocode", _down)
    out = L.locate_node({"article_text": "Heavy rain hit Kochi on Monday.",
                         "claims": ["Heavy rain hit Kochi on Monday."]})
    assert out["claim_locations"][0]["method"] == "gazetteer"          # still located
    assert any(d["kind"] == "geocoder_unavailable" and d["severity"] == "info"
               for d in out["degradations"])


def test_compact_drops_internal_fields():
    out = L.compact({"name": "Pune", "level": "city", "importance": 0.9, "osm_class": "x"})
    assert "importance" not in out and out["name"] == "Pune"
    assert L.compact(None) is None
