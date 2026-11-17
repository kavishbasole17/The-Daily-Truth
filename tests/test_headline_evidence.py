"""
Regression tests for the 2026-11-17 headline-evidence pass (AGENTS.md, Part 8).

  * a news headline about a story contradicted a detailed claim about the
    same story ("SpaceX launches Starship Flight 15" vs "SpaceX caught
    Starship ..."), because NLI reads a different event as contradiction
  * "around Earth" was geocoded to the town of Earth, Texas
  * "the Gulf of Mexico" was located in the country Mexico
"""
from __future__ import annotations

from paper_trail_core.agents import verifier as V
from paper_trail_core.agents.locator import extract_candidates
from paper_trail_core.nlp.nli import NLIResult

CLAIM = ("SpaceX successfully caught its Starship spacecraft using giant robotic "
         "arms at its Starbase facility in Texas.")


def _snip(text, **kw):
    base = {"text": text, "relevance": 0.7, "trust": 0.7, "domain": "abc7.com",
            "provider": "gnews_rss", "url": "https://abc7.com/x"}
    base.update(kw)
    return base


def _res(ent, con):
    return NLIResult(entail=ent, neutral=max(0.0, 1 - ent - con), contradict=con)


def test_headline_cannot_refute_a_detailed_claim():
    s = _snip("SpaceX launches Starship Flight 15 from South Texas.")
    cv = V.aggregate_claim(CLAIM, [s], [_res(0.005, 0.93)])
    assert cv["verdict"] != "contradiction"
    assert cv["contradict_mass"] == 0.0
    assert cv["evidence_meta"][0]["stance"] != "debunks"


def test_debunking_headline_still_refutes():
    s = _snip("Fact check: SpaceX did not catch Starship, video is fabricated.")
    cv = V.aggregate_claim(CLAIM, [s], [_res(0.0, 0.95)])
    assert cv["verdict"] == "contradiction"


def test_near_paraphrase_headline_still_refutes():
    s = _snip("SpaceX failed to catch Starship with robotic arms at Starbase, Texas.", relevance=0.85)
    cv = V.aggregate_claim(CLAIM, [s], [_res(0.0, 0.95)])
    assert cv["verdict"] == "contradiction"


def test_full_text_snippet_can_still_refute():
    body = ("SpaceX launched Starship on Thursday from Starbase. The upper stage "
            "splashed down in the Indian Ocean as planned; no catch was attempted.")
    s = _snip(body, provider="tavily")
    cv = V.aggregate_claim(CLAIM, [s], [_res(0.0, 0.95)])
    assert cv["verdict"] == "contradiction"


def test_headline_can_still_support():
    s = _snip("SpaceX catches Starship with robotic arms at Starbase in Texas.")
    cv = V.aggregate_claim(CLAIM, [s], [_res(0.95, 0.0)])
    assert cv["verdict"] == "entailment"


def _surfaces(text):
    return {c.surface.lower() for c in extract_candidates(text)}


def test_earth_is_not_a_place():
    assert "earth" not in _surfaces("Starship completed 14 orbits around Earth.")


def test_gulf_of_mexico_is_not_mexico():
    found = _surfaces("The booster was destroyed over the Gulf of Mexico.")
    assert "mexico" not in found


def test_mexico_alone_is_still_a_place():
    assert "mexico" in _surfaces("Protests spread across Mexico on Sunday.")
