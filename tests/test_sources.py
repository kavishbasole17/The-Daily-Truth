import pytest
from paper_trail_core.agents.sources import TRUSTED_NEWS_DOMAINS, is_indian, trust_for


def test_indian_outlet_outranks_foreign_wire():
    assert trust_for("thehindu.com") > trust_for("reuters.com")
    assert trust_for("https://www.ptinews.com/x") > trust_for("bbc.co.uk")


def test_foreign_news_is_scaled_down():
    assert trust_for("nytimes.com") == pytest.approx(0.63)


def test_reference_sources_are_not_scaled():
    assert trust_for("who.int") == 0.95
    assert trust_for("en.wikipedia.org") == 0.6


def test_unlisted_indian_domain_beats_unlisted_foreign():
    assert is_indian("example.co.in")
    assert trust_for("example.co.in") > trust_for("example.com")


def test_outlet_display_name_resolves():
    assert trust_for("The Hindu") == trust_for("thehindu.com")


def test_newsapi_queries_indian_domains():
    assert "ptinews.com" in TRUSTED_NEWS_DOMAINS
    assert "reuters.com" in TRUSTED_NEWS_DOMAINS


def test_every_indian_outlet_outranks_every_foreign_fallback():
    from paper_trail_core.agents.sources import _INDIA, _MAJOR, _WIRE
    lowest_indian = min(_INDIA.values())
    foreign = [trust_for(d) for d in {**_WIRE, **_MAJOR} if not is_indian(d)]
    assert lowest_indian > max(foreign)
    assert lowest_indian > trust_for("en.wikipedia.org")
