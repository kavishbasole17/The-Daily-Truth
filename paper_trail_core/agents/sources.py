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

import re
from typing import Any
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
    # India
    "thehindu.com": 0.9, "indianexpress.com": 0.85,
    "hindustantimes.com": 0.85, "livemint.com": 0.85,
    "timesofindia.indiatimes.com": 0.8, "ndtv.com": 0.8,
    # Asia-Pacific
    "japantimes.co.jp": 0.85, "scmp.com": 0.8, "abc.net.au": 0.9,
    "straitstimes.com": 0.8,
}

# Reference sources — excellent for settled facts, slow on breaking events.
_REFERENCE = {
    "wikipedia.org": 0.85,
    "en.wikipedia.org": 0.85,
    "who.int": 0.95,
    "un.org": 0.9,
    "nasa.gov": 0.95,
    "cern.ch": 0.95,
    "nih.gov": 0.95,
    "cdc.gov": 0.9,
    "imf.org": 0.9,
    "worldbank.org": 0.9,
}

# Regional & Local News Outlets
_REGIONAL_MEDIA: dict[str, dict[str, Any]] = {
    "telugu": {
        "name": "Telugu / AP & Telangana",
        "keywords": ["telugu", "andhra", "telangana", "hyderabad", "tollywood", "amaravati", "vizag", "visakhapatnam", "nani", "chiranjeevi", "prabhas", "allu arjun", "mahesh babu", "ntr"],
        "domains": {
            "thehansindia.com": 0.8,
            "deccanchronicle.com": 0.85,
            "telanganatoday.com": 0.8,
            "cinemaexpress.com": 0.85,
            "123telugu.com": 0.8,
            "greatandhra.com": 0.75,
            "gulte.com": 0.75,
            "pinkvilla.com": 0.75,
            "thehindu.com": 0.9,
            "timesofindia.indiatimes.com": 0.8,
        }
    },
    "tamil": {
        "name": "Tamil / Tamil Nadu",
        "keywords": ["tamil", "tamil nadu", "chennai", "kollywood", "madurai", "coimbatore", "vijay", "ajith", "rajinikanth", "kamal haasan"],
        "domains": {
            "dtnext.in": 0.8,
            "thehindu.com": 0.9,
            "cinemaexpress.com": 0.85,
            "dinamalar.com": 0.8,
            "timesofindia.indiatimes.com": 0.8,
        }
    },
    "kannada": {
        "name": "Kannada / Karnataka",
        "keywords": ["kannada", "karnataka", "bangalore", "bengaluru", "sandalwood", "mysore", "yash", "sudeep"],
        "domains": {
            "deccanherald.com": 0.85,
            "thehindu.com": 0.9,
            "bangaloremirror.indiatimes.com": 0.8,
        }
    },
    "malayalam": {
        "name": "Malayalam / Kerala",
        "keywords": ["malayalam", "kerala", "kochi", "thiruvananthapuram", "mollywood", "calicut", "mohanlal", "mammootty"],
        "domains": {
            "onmanorama.com": 0.85,
            "mathrubhumi.com": 0.8,
            "thehindu.com": 0.9,
        }
    },
    "hindi_north": {
        "name": "North India / Bollywood",
        "keywords": ["delhi", "mumbai", "bollywood", "uttar pradesh", "bihar", "punjab", "rajasthan", "haryana", "noida", "shah rukh", "salman khan"],
        "domains": {
            "hindustantimes.com": 0.85,
            "indianexpress.com": 0.85,
            "ndtv.com": 0.8,
            "mid-day.com": 0.8,
            "freepressjournal.in": 0.8,
            "filmfare.com": 0.8,
            "bollywoodhungama.com": 0.8,
            "timesofindia.indiatimes.com": 0.8,
        }
    },
    "uk": {
        "name": "United Kingdom",
        "keywords": ["uk", "britain", "british", "london", "england", "scotland", "wales", "westminster", "downing street", "parliament"],
        "domains": {
            "bbc.co.uk": 0.9, "bbc.com": 0.9,
            "theguardian.com": 0.9,
            "telegraph.co.uk": 0.85,
            "independent.co.uk": 0.85,
            "news.sky.com": 0.85,
            "ft.com": 0.9,
        }
    },
    "us": {
        "name": "United States",
        "keywords": ["us", "usa", "america", "american", "washington", "new york", "california", "texas", "florida", "congress", "senate", "white house"],
        "domains": {
            "nytimes.com": 0.9, "washingtonpost.com": 0.9,
            "wsj.com": 0.9, "npr.org": 0.9,
            "cbsnews.com": 0.85, "nbcnews.com": 0.85, "abcnews.go.com": 0.85,
            "latimes.com": 0.85, "politico.com": 0.85,
        }
    },
    "middle_east": {
        "name": "Middle East / Gulf",
        "keywords": ["uae", "dubai", "gulf", "saudi", "qatar", "israel", "palestine", "middle east", "abu dhabi"],
        "domains": {
            "aljazeera.com": 0.85,
            "khaleejtimes.com": 0.8,
            "gulfnews.com": 0.8,
            "thenationalnews.com": 0.85,
        }
    }
}

# Flatten all regional domain trusts
_ALL_REGIONAL_DOMAINS: dict[str, float] = {}
for _r in _REGIONAL_MEDIA.values():
    _ALL_REGIONAL_DOMAINS.update(_r.get("domains", {}))

TRUST: dict[str, float] = {**_WIRE, **_MAJOR, **_REFERENCE, **_ALL_REGIONAL_DOMAINS}

DEFAULT_TRUST = 0.5

TRUSTED_NEWS_DOMAINS = [d for d in {**_WIRE, **_MAJOR, **_ALL_REGIONAL_DOMAINS} if "." in d]


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
    stripped = base[4:] if base.startswith("the ") else base
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
        return host[4:] if host.startswith("www.") else host

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


def trust_for(url_or_name: str) -> float:
    """Trust weight in [0, 1] for a source. Unknown sources get DEFAULT_TRUST."""
    domain = domain_of(url_or_name)
    if not domain:
        return DEFAULT_TRUST
    if domain in TRUST:
        return TRUST[domain]
    # Match subdomains against their registrable parent (e.g. edition.cnn.com).
    parts = domain.split(".")
    for i in range(len(parts) - 1):
        parent = ".".join(parts[i:])
        if parent in TRUST:
            return TRUST[parent]
    return DEFAULT_TRUST


def extract_regions(text: str) -> list[str]:
    """Identify regions or localities mentioned in text based on keywords/entities."""
    if not text:
        return []
    lowered = text.lower()
    matched = []
    for r_key, r_info in _REGIONAL_MEDIA.items():
        for kw in r_info.get("keywords", []):
            # Check for whole word or phrase match
            pattern = rf"\b{re.escape(kw)}\b"
            if re.search(pattern, lowered):
                if r_key not in matched:
                    matched.append(r_key)
                break
    return matched


def get_regional_domains(regions: list[str]) -> list[str]:
    """Retrieve trusted domains specific to the detected regions."""
    domains = []
    for r in regions:
        if r in _REGIONAL_MEDIA:
            domains.extend(list(_REGIONAL_MEDIA[r].get("domains", {}).keys()))
    return list(dict.fromkeys(domains))


def get_region_display_name(region_key: str) -> str:
    """Friendly display name for a region key."""
    if region_key in _REGIONAL_MEDIA:
        return _REGIONAL_MEDIA[region_key].get("name", region_key.title())
    return region_key.title()

