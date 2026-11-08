"""
Source trust tiers.

Evidence is not interchangeable. A Reuters wire report and an anonymous blog
post asserting opposite things are not a tie, and the verifier needs a number
to express that when it weighs one snippet against another.

An asymmetry worth stating, because it drives a design decision elsewhere:
**Wikipedia is a strong premise for established background facts and a weak
one for anything from the last few days.** It lags breaking news by design.
That is a large part of why "no evidence found" must never be read as "this is
fabricated" — for recent events, silence from our retrieval stack is the
expected outcome for true and false articles alike.
"""
from __future__ import annotations

from urllib.parse import urlparse

# Wire services — the primary sourcing layer most other outlets republish.
_WIRE = {
    "reuters.com": 1.0,
    "apnews.com": 1.0,
    "afp.com": 1.0,
    "pa.media": 0.95,
}

# Major national outlets with dedicated newsrooms and corrections policies.
_MAJOR = {
    "bbc.co.uk": 0.9, "bbc.com": 0.9,
    "npr.org": 0.9, "wsj.com": 0.9, "washingtonpost.com": 0.9,
    "nytimes.com": 0.9, "theguardian.com": 0.9, "ft.com": 0.9,
    "economist.com": 0.9, "bloomberg.com": 0.9, "cbsnews.com": 0.85,
    "abcnews.go.com": 0.85, "nbcnews.com": 0.85, "cnn.com": 0.8,
    "axios.com": 0.85, "politico.com": 0.85, "usatoday.com": 0.8,
    "aljazeera.com": 0.85, "dw.com": 0.85,
    # Asia-Pacific
    "japantimes.co.jp": 0.85, "scmp.com": 0.8, "abc.net.au": 0.9,
    "straitstimes.com": 0.8,
}

# Indian outlets. The product serves Indian readers, so these outrank foreign
# outlets of comparable standing: wires and the government's own releases first,
# then the national dailies.
_INDIA = {
    "ptinews.com": 1.0, "aninews.in": 0.95, "pib.gov.in": 1.0,
    "thehindu.com": 0.95, "indianexpress.com": 0.92,
    "hindustantimes.com": 0.92, "livemint.com": 0.9,
    "timesofindia.indiatimes.com": 0.9, "ndtv.com": 0.9,
    "indiatoday.in": 0.88, "business-standard.com": 0.9,
    "economictimes.indiatimes.com": 0.88, "deccanherald.com": 0.88,
    "theprint.in": 0.85, "scroll.in": 0.85, "thewire.in": 0.82,
    "news18.com": 0.82, "telegraphindia.com": 0.88, "thehindubusinessline.com": 0.9,
}

# Reference sources — excellent for settled facts, slow on breaking events.
_REFERENCE = {
    "wikipedia.org": 0.6,
    "en.wikipedia.org": 0.6,
    "who.int": 0.95,
    "un.org": 0.9,
    "nasa.gov": 0.95,
    "cern.ch": 0.95,
    "nih.gov": 0.95,
    "cdc.gov": 0.9,
    "imf.org": 0.9,
    "worldbank.org": 0.9,
}

TRUST: dict[str, float] = {**_WIRE, **_MAJOR, **_INDIA, **_REFERENCE}

DEFAULT_TRUST = 0.5
# An unlisted Indian domain (.in / .co.in) is still more relevant to an Indian
# reader than an unlisted foreign one, so it starts a little above the default.
INDIAN_DEFAULT_TRUST = 0.65

# Domains queried through NewsAPI's `domains` filter. `domains` rather than
# NewsAPI's fixed `sources` catalogue: `sources` has very thin non-US/UK
# coverage, and combining both in one request silently drops the `domains`
# filter, so international outlets never actually got queried.
TRUSTED_NEWS_DOMAINS = [d for d in {**_INDIA, **_WIRE, **_MAJOR} if "." in d]


# Display names NewsAPI reports that don't reduce to their domain by any
# mechanical rule. Without these, "Associated Press" and "The Hindu" fall back
# to the default 0.5 trust — halving the evidence weight of a wire service.
_NAME_OVERRIDES = {
    "associated press": "apnews.com",
    "ap news": "apnews.com",
    "agence france-presse": "afp.com",
    "bbc news": "bbc.co.uk",
    "bbc": "bbc.co.uk",
    "the press association": "pa.media",
    "al jazeera english": "aljazeera.com",
    "al jazeera": "aljazeera.com",
    "deutsche welle": "dw.com",
    "the new york times": "nytimes.com",
    "the wall street journal": "wsj.com",
    "the washington post": "washingtonpost.com",
    "the financial times": "ft.com",
    "financial times": "ft.com",
    "the economist": "economist.com",
    "abc news": "abcnews.go.com",
    "abc news (au)": "abc.net.au",
    "nbc news": "nbcnews.com",
    "cbs news": "cbsnews.com",
    "usa today": "usatoday.com",
    "the times of india": "timesofindia.indiatimes.com",
    "times of india": "timesofindia.indiatimes.com",
    "the indian express": "indianexpress.com",
    "world health organization": "who.int",
    "united nations": "un.org",
}


def _name_candidates(name: str) -> list[str]:
    """Normalised forms of an outlet name to try against the domain table."""
    base = name.lower().strip()
    stripped = base.removeprefix("the ")
    forms = [base, stripped]
    # "BBC News" -> "bbc"; "Reuters Africa" -> "reuters"
    for form in (base, stripped):
        for suffix in (" news", " online", " digital", " english", " uk", " us"):
            if form.endswith(suffix):
                forms.append(form[: -len(suffix)])
    return [f.replace(" ", "").replace("-", "").replace(".", "") for f in forms if f]


def domain_of(url_or_name: str) -> str:
    """Best-effort domain for a URL or an outlet display name."""
    if not url_or_name:
        return ""
    value = url_or_name.strip()

    if "://" in value or value.startswith("www."):
        host = urlparse(value if "://" in value else f"https://{value}").netloc.lower()
        return host.removeprefix("www.")

    if "." in value and " " not in value:
        return value.lower()

    # An outlet display name ("The Hindu", "Reuters") — resolve it to a known
    # domain so trust still applies when only a name is available.
    lowered = value.lower()
    if lowered in _NAME_OVERRIDES:
        return _NAME_OVERRIDES[lowered]

    candidates = _name_candidates(value)
    for domain in TRUST:
        # Compare against the registrable name with dots removed, so
        # "thehindu.com" matches "thehindu" and "bbc.co.uk" matches "bbc".
        stem = domain.split(".")[0].replace("-", "")
        if stem in candidates:
            return domain
    return lowered


def is_indian(domain: str) -> bool:
    """True for a listed Indian outlet or any .in / .co.in / .gov.in domain."""
    if not domain:
        return False
    if domain in _INDIA or any(domain.endswith("." + d) for d in _INDIA):
        return True
    return domain == "in" or domain.endswith(".in")


def _foreign_factor() -> float:
    try:
        from config import settings
        return float(settings.FOREIGN_NEWS_FACTOR)
    except Exception:  # pragma: no cover
        return 0.7


def trust_for(url_or_name: str) -> float:
    """
    Trust weight in [0, 1] for a source. Unknown sources get DEFAULT_TRUST.

    Foreign news outlets are scaled by FOREIGN_NEWS_FACTOR so Indian outlets
    outrank them; reference bodies (WHO, NASA, UN...) are not news and are left
    alone.
    """
    domain = domain_of(url_or_name)
    if not domain:
        return DEFAULT_TRUST
    base = _lookup(domain)
    if base is None:
        return INDIAN_DEFAULT_TRUST if is_indian(domain) else DEFAULT_TRUST
    if is_indian(domain) or _is_reference(domain):
        return base
    return round(base * _foreign_factor(), 4)


def _registrable_match(domain: str):
    if domain in TRUST:
        return domain
    parts = domain.split(".")
    for i in range(len(parts) - 1):
        parent = ".".join(parts[i:])
        if parent in TRUST:
            return parent
    return None


def _lookup(domain: str):
    key = _registrable_match(domain)
    return TRUST[key] if key else None


def _is_reference(domain: str) -> bool:
    key = _registrable_match(domain)
    return key in _REFERENCE if key else False
