"""
Locator agent node — finds the place each claim is about, as precisely as the
text supports.

    article text ──► place candidates ──► resolve (gazetteer / geocoder)
                                              │
                      claim ◄── mentions ◄────┘      article-level fallback

What "as precisely as possible" means here is a *level*: site, street,
locality, city, district, state or country. A place is reported at the most
specific level it could be resolved to, together with a confidence. A name that
could be several different places and cannot be told apart from context
("Springfield") is reported as *not located* rather than guessed — a wrong
region sends retrieval to the wrong newsroom, which is worse than searching
globally.

Design constraints:

  * **Candidates need a reason to be considered.** A capitalised phrase is only
    treated as a possible place if it is in the gazetteer, follows a place cue
    ("in", "near", "residents of" ...), opens the article as a dateline, or is
    followed by a place noun ("Pune district"). That keeps people and
    organisations out of the geocoder and keeps the number of lookups small.
  * **Resolution uses the whole article.** Countries and states mentioned
    elsewhere in the text are the context that separates "Aurangabad,
    Maharashtra" from "Aurangabad, Bihar".
  * **Nothing here changes a verdict.** The output only widens where evidence is
    searched (see evidence_retriever). Fusion features are untouched.
  * **Fails soft.** If the geocoder is unreachable the stage falls back to the
    built-in gazetteer and says so; if everything fails the claims are simply
    treated as having no location.
"""
from __future__ import annotations

import logging
import os
import re
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from paper_trail_core.agents.state import PipelineState
from paper_trail_core.degradations import degrade
from paper_trail_core.nlp import gazetteer
from paper_trail_core.nlp.geocoder import GeocoderUnavailable, LEVEL_ORDER, geocode

log = logging.getLogger(__name__)

MAX_CANDIDATES = 8          # distinct places resolved per article
RESOLVE_BUDGET_S = 40.0     # wall-clock cap on geocoding per article
INHERIT_MIN_CONFIDENCE = 0.7
ARTICLE_DISCOUNT = 0.85     # confidence haircut for a place inherited from the article

# Weights of the candidate-vs-result score. They sum to 1.
_W_NAME, _W_IMPORTANCE, _W_CONTEXT, _W_CUE = 0.45, 0.25, 0.20, 0.10


def _min_confidence() -> float:
    try:
        from config import settings
        return float(settings.LOCATION_MIN_CONFIDENCE)
    except Exception:  # pragma: no cover
        return 0.5


# ── Candidate extraction ──────────────────────────────────────────────────────

_WORD = r"[A-Z][A-Za-z'’\-]*"
_RUN_RE = re.compile(
    rf"(?<![\w'’]){_WORD}(?:\s+(?:(?:of|de|del|da|la|le|al|el|upon)\s+)?{_WORD}){{0,3}}"
)

_CUE_PRE_RE = re.compile(
    r"(?:\b(?:in|at|near|from|across|around|outside|inside|within|throughout|"
    r"towards?|via|through|into)|"
    r"\b(?:mayor|residents?|people|city|state|capital|district|town|village|region|"
    r"outskirts|heart|centre|center|suburbs?|areas?|parts|streets|police|"
    r"officials|authorities|hospitals?|schools?|villagers)\s+of)\s+(?:the\s+)?$",
    re.IGNORECASE,
)
_CUE_POST_RE = re.compile(
    r"^\s+(?:district|city|town|village|region|area|taluk|tehsil|police|airport|"
    r"railway|station|dam|river|lake|bridge|market|hospital|valley|island|port|"
    r"highway|municipal|corporation)\b",
    re.IGNORECASE,
)
_DATELINE_RE = re.compile(
    r"^\s*([A-Z][A-Za-z .'’\-]{2,40}?)\s*(?:\([^)]{2,30}\))?\s*(?:—|–|-|:)\s"
)
# A building-level match ("University of Cambridge", "Khadakwasla Dam") is only
# trusted when the name itself says it is a place. Otherwise companies, agencies
# and products ("Tata Motors", "Reuters") would be pinned to their head office.
_SITE_WORDS_RE = re.compile(
    r"\b(?:university|college|school|institute|hospital|airport|station|stadium|temple|"
    r"mosque|church|market|park|dam|bridge|fort|palace|mall|port|museum|court|"
    r"highway|junction|square|lake|river|beach|garden|gardens|nagar|colony)\b",
    re.IGNORECASE,
)
_ORG_SUFFIX = frozenset("ltd inc corp party bank ministry council committee commission "
                        "association agency authority board".split())
_HONORIFICS = frozenset("mr mrs ms dr shri smt prof sir madam hon".split())
_SPEECH_VERBS = frozenset("said says told added stated announced claimed alleged "
                          "wrote tweeted posted".split())

_TLD_COUNTRY = {
    "in": "in", "pk": "pk", "bd": "bd", "lk": "lk", "np": "np", "uk": "gb",
    "au": "au", "nz": "nz", "ca": "ca", "za": "za", "ng": "ng", "ke": "ke",
    "sg": "sg", "my": "my", "ae": "ae", "de": "de", "fr": "fr", "jp": "jp",
}


@dataclass
class Candidate:
    surface: str
    cue: bool = False
    dateline: bool = False
    gaz: Optional[gazetteer.GazEntry] = None
    qualifier: Optional[str] = None     # "Koregaon Park, Pune" -> qualifier "Pune"
    count: int = 1
    first_pos: int = 0


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(s.lower().replace("’", "'").split())


def _trim_run(words: list[str]) -> list[str]:
    """
    Drop words that cannot be part of a place name from either end: leading
    ("The", "Prime", "Mr") and trailing ("Monday", "March").
    """
    if [w.lower() for w in words[:2]] == ["the", "hague"]:
        return words
    while words and words[0].lower() in gazetteer.NOT_PLACE_WORDS:
        words = words[1:]
    while words and words[-1].lower() in gazetteer.NOT_PLACE_WORDS:
        words = words[:-1]
    return words


_gaz_pattern: Optional[re.Pattern] = None


def _gaz_regex() -> re.Pattern:
    global _gaz_pattern
    if _gaz_pattern is None:
        names = sorted(gazetteer.all_names(), key=len, reverse=True)
        _gaz_pattern = re.compile(
            r"(?<![\w])(" + "|".join(re.escape(n) for n in names) + r")(?![\w])",
            re.IGNORECASE,
        )
    return _gaz_pattern


def _gazetteer_matches(text: str) -> list[tuple[int, int, gazetteer.GazEntry]]:
    """Every gazetteer name in the text, longest-first, non-overlapping."""
    out, taken = [], []
    for m in _gaz_regex().finditer(text):
        surface = m.group(1)
        # Short names (UK, US, UAE) must be written in capitals; everything else
        # must at least start with one, so "goa" the verb or "turkey" the bird
        # does not count.
        if len(surface) <= 3:
            if surface != surface.upper():
                continue
        elif not surface[0].isupper():
            continue
        if any(m.start() < e and s < m.end() for s, e in taken):
            continue
        entry = gazetteer.lookup(surface)
        if entry is None:
            continue
        taken.append((m.start(), m.end()))
        out.append((m.start(), m.end(), entry))
    return out


def extract_candidates(text: str) -> list[Candidate]:
    """Possible places in `text`, each with the reason it was considered."""
    found: dict[str, Candidate] = {}

    def add(surface: str, pos: int, **kw) -> Candidate:
        key = _norm(surface)
        cand = found.get(key)
        if cand is None:
            cand = found[key] = Candidate(surface=surface, first_pos=pos, **kw)
        else:
            cand.count += 1
            cand.cue = cand.cue or kw.get("cue", False)
            cand.dateline = cand.dateline or kw.get("dateline", False)
            cand.gaz = cand.gaz or kw.get("gaz")
        return cand

    m = _DATELINE_RE.match(text[:140])
    if m:
        surface = m.group(1).strip().title() if m.group(1).isupper() else m.group(1).strip()
        words = _trim_run(surface.split())
        if words:
            add(" ".join(words), m.start(1), cue=True, dateline=True,
                gaz=gazetteer.lookup(" ".join(words)))

    gaz_spans = _gazetteer_matches(text)
    for start, end, entry in gaz_spans:
        before = text[max(0, start - 40):start]
        cue = bool(_CUE_PRE_RE.search(before)) or bool(_CUE_POST_RE.match(text[end:end + 30]))
        add(text[start:end], start, cue=cue, gaz=entry)

    for run in _RUN_RE.finditer(text):
        words = _trim_run(run.group(0).split())
        if not words:
            continue
        surface = " ".join(words)
        if surface.endswith(("'s", "’s")):
            surface = surface[:-2]
        norm_words = _norm(surface).split()
        if not norm_words or norm_words[-1] in _ORG_SUFFIX:
            continue
        if len(norm_words) == 1 and (len(norm_words[0]) < 3 or surface.isupper()):
            continue                      # acronyms (WHO, FBI) are not places
        start = run.start()
        before = text[max(0, run.start() - 40):run.start()]
        after = text[run.end():run.end() + 30]
        cue = bool(_CUE_PRE_RE.search(before)) or bool(_CUE_POST_RE.match(after))
        if not cue:
            continue
        prev_word = before.split()[-1].lower().strip(".,") if before.split() else ""
        next_word = after.split()[0].lower().strip(".,;:") if after.split() else ""
        if prev_word in _HONORIFICS or next_word in _SPEECH_VERBS:
            continue
        if _norm(surface) in found:
            # Gazetteer names were already counted by the gazetteer scan above.
            if found[_norm(surface)].gaz is None:
                found[_norm(surface)].count += 1
            continue
        add(surface, start, cue=True, gaz=gazetteer.lookup(surface))

    # "Koregaon Park, Pune": keep the pair together for a more specific lookup.
    ordered = sorted(found.values(), key=lambda c: c.first_pos)
    for cand in ordered:
        if cand.gaz is not None and cand.gaz.level in ("country", "state"):
            continue
        m = re.search(re.escape(cand.surface) + r"\s*,\s*(" + _WORD + r"(?:\s+" + _WORD + r")*)",
                      text)
        if m:
            nxt = _norm(m.group(1))
            if nxt in found and _norm(found[nxt].surface) != _norm(cand.surface):
                cand.qualifier = found[nxt].surface

    def priority(c: Candidate):
        return (not c.dateline, c.gaz is None, -c.count, c.first_pos)

    return sorted(found.values(), key=priority)


# ── Context ───────────────────────────────────────────────────────────────────

@dataclass
class Context:
    countries: dict[str, int] = field(default_factory=dict)   # iso2 -> weight
    names: set[str] = field(default_factory=set)              # normalised place names in text

    def top_countries(self) -> list[str]:
        return [c for c, _ in sorted(self.countries.items(), key=lambda kv: -kv[1])]


def build_context(candidates: list[Candidate], url: Optional[str],
                  hint_place: Optional[dict]) -> Context:
    ctx = Context()
    for c in candidates:
        # Only what the text actually says is context. A gazetteer city's default
        # state ("Aurangabad" -> Maharashtra) is an assumption, not evidence, so
        # it must not vote; countries and states named in the text do.
        ctx.names.add(_norm(c.surface))
        if c.gaz is not None and c.gaz.level in ("country", "state"):
            ctx.countries[c.gaz.country_code] = ctx.countries.get(c.gaz.country_code, 0) + c.count
    if url:
        from paper_trail_core.agents.sources import domain_of
        tld = domain_of(url).rsplit(".", 1)[-1]
        cc = _TLD_COUNTRY.get(tld)
        if cc:
            ctx.countries[cc] = ctx.countries.get(cc, 0) + 1
    if hint_place:
        if hint_place.get("country_code"):
            cc = hint_place["country_code"]
            ctx.countries[cc] = ctx.countries.get(cc, 0) + 3
        for k in ("city", "state", "district"):
            if hint_place.get(k):
                ctx.names.add(_norm(hint_place[k]))
    return ctx


# ── Resolution ────────────────────────────────────────────────────────────────

_GENERIC_SUFFIX = ("district", "city", "town", "village", "division", "region", "taluk")


def _name_match(cand_norm: str, result: dict) -> float:
    name = _norm(result.get("name") or "")
    if cand_norm == name:
        return 1.0
    for suffix in _GENERIC_SUFFIX:
        if cand_norm == f"{name} {suffix}" or name == f"{cand_norm} {suffix}":
            return 0.9
    for k in ("city", "state", "district", "country", "locality"):
        if result.get(k) and _norm(result[k]) == cand_norm and result.get("level") == k:
            return 1.0
    if re.search(rf"(?<![\w]){re.escape(cand_norm)}(?![\w])", _norm(result.get("display_name") or "")):
        return 0.3
    return 0.0


def _score(cand: Candidate, result: dict, ctx: Context) -> float:
    name = _name_match(_norm(cand.surface), result)
    if name == 0.0:
        return 0.0
    importance = min(1.0, float(result.get("importance") or 0.0) / 0.6)
    country_ctx = 1.0 if result.get("country_code") in ctx.top_countries() else 0.0
    hier_ctx = 0.0
    own = _norm(cand.surface)
    for k in ("city", "district", "state"):
        v = result.get(k)
        if v and _norm(v) in ctx.names and _norm(v) != own:
            hier_ctx = 1.0
            break
    context = min(1.0, 0.6 * country_ctx + 0.7 * hier_ctx)
    cue = 1.0 if (cand.cue or cand.gaz is not None) else 0.0
    return (_W_NAME * name + _W_IMPORTANCE * importance
            + _W_CONTEXT * context + _W_CUE * cue)


def _area(result: dict) -> tuple:
    return (result.get("country_code"), _norm(result.get("state") or ""))


def _label(p: dict) -> str:
    parts: list[str] = []
    for part in (p.get("name"), p.get("locality"), p.get("city"), p.get("district"),
                 p.get("state"), p.get("country")):
        if part and _norm(part) not in {_norm(x) for x in parts}:
            parts.append(part)
    return ", ".join(parts)


def _finish(p: dict, confidence: float, method: str, mentions: list[str],
            ambiguous: bool = False) -> dict:
    out = {
        "name": p.get("name"), "level": p.get("level"),
        "lat": p.get("lat"), "lon": p.get("lon"),
        "country": p.get("country"), "country_code": p.get("country_code"),
        "state": p.get("state"), "district": p.get("district"),
        "city": p.get("city"), "locality": p.get("locality"),
        "confidence": round(max(0.0, min(1.0, confidence)), 3),
        "method": method, "mentions": mentions, "ambiguous": ambiguous,
    }
    out["label"] = _label(out)
    return out


def _gazetteer_place(cand: Candidate) -> dict:
    g = cand.gaz
    assert g is not None
    base = {"name": g.name, "level": g.level, "lat": None, "lon": None,
            "country": g.country, "country_code": g.country_code, "state": g.state,
            "district": None, "city": g.name if g.level == "city" else None,
            "locality": None}
    if g.level == "state":
        base["state"] = g.name
    return base


def _resolve(cand: Candidate, ctx: Context, budget: "Budget") -> Optional[dict]:
    """One candidate -> a finished place dict, or None if it cannot be trusted."""
    ambiguous_name = cand.gaz is not None and gazetteer.is_ambiguous(cand.surface)
    if cand.gaz is not None and not cand.cue and ambiguous_name:
        return None                         # "Salem" with no place cue is a name, not a place

    results: list[dict] = []
    if budget.can_geocode():
        query = f"{cand.surface}, {cand.qualifier}" if cand.qualifier else cand.surface
        try:
            results = geocode(query, limit=6)
            if not results and cand.qualifier:
                results = geocode(cand.surface, limit=6)
        except GeocoderUnavailable:
            budget.geocoder_down = True
            results = []

    if results:
        scored = sorted(((_score(cand, r, ctx), r) for r in results), key=lambda t: -t[0])
        best_score, best = scored[0]
        if best_score > 0.0:
            # A gazetteer entry pins the country: prefer a result that agrees with it.
            if cand.gaz is not None:
                agree = [(s, r) for s, r in scored
                         if r.get("country_code") == cand.gaz.country_code and s > 0]
                if agree:
                    best_score, best = agree[0]
            rival = next((s for s, r in scored[1:] if _area(r) != _area(best) and s > 0), None)
            conf = best_score
            ambiguous = False
            if rival is not None:
                margin = best_score - rival
                conf = best_score * (0.6 + 0.4 * min(1.0, margin / 0.25))
                ambiguous = margin < 0.15
            if cand.gaz is not None and not ambiguous_name:
                conf = max(conf, 0.9)
            if (best.get("level") == "site" and cand.gaz is None and not cand.dateline
                    and not _SITE_WORDS_RE.search(cand.surface)):
                return None
            mentions = [cand.surface] + ([cand.qualifier] if cand.qualifier else [])
            return _finish(best, conf, "nominatim", mentions[:1], ambiguous)

    if cand.gaz is not None:
        conf = 0.6 if ambiguous_name else 0.9
        return _finish(_gazetteer_place(cand), conf, "gazetteer", [cand.surface])
    return None


class Budget:
    def __init__(self) -> None:
        self.start = time.time()
        self.geocoder_down = False
        self.n_lookups = 0

    def can_geocode(self) -> bool:
        return not self.geocoder_down and (time.time() - self.start) < RESOLVE_BUDGET_S


# ── Claim assignment ──────────────────────────────────────────────────────────

def _mentions(text: str, surface: str) -> bool:
    return bool(re.search(rf"(?<![\w]){re.escape(surface)}(?![\w])", text, re.IGNORECASE))


def _specificity(level: str) -> int:
    return LEVEL_ORDER.index(level) if level in LEVEL_ORDER else len(LEVEL_ORDER)


def _public(place: dict, source: str, confidence: Optional[float] = None,
            also: Optional[list[dict]] = None) -> dict:
    out = dict(place)
    out["source"] = source
    if confidence is not None:
        out["confidence"] = round(confidence, 3)
    out["also"] = also or []
    return out


@dataclass
class LocateResult:
    claim_locations: list[Optional[dict]]
    article_location: Optional[dict]
    notes: dict[str, Any] = field(default_factory=dict)


def locate_text(
    article_text: str,
    claims: list[str],
    url: Optional[str] = None,
    region_hint: Optional[str] = None,
) -> LocateResult:
    """Resolve places in the article and assign the best one to each claim."""
    min_conf = _min_confidence()
    budget = Budget()
    notes: dict[str, Any] = {"ambiguous": [], "geocoder_down": False}

    hint_place: Optional[dict] = None
    if region_hint and region_hint.strip():
        hint_cand = Candidate(surface=region_hint.strip(), cue=True,
                              gaz=gazetteer.lookup(region_hint.strip()))
        hint_place = _resolve(hint_cand, Context(), budget)
        if hint_place is not None:
            hint_place["method"] = "user"

    candidates = extract_candidates(article_text)
    ctx = build_context(candidates, url, hint_place)

    resolved: list[tuple[Candidate, dict]] = []
    unresolved: list[str] = []      # places the text names but we could not pin down
    for cand in candidates[:MAX_CANDIDATES]:
        place = _resolve(cand, ctx, budget)
        if place is not None and place["ambiguous"]:
            notes["ambiguous"].append(cand.surface)
        if place is not None and place["confidence"] >= min_conf:
            resolved.append((cand, place))
        elif cand.cue or cand.gaz is not None:
            unresolved.append(cand.surface)
    notes["geocoder_down"] = budget.geocoder_down

    # The place the article as a whole is about. Only a place the article is
    # demonstrably *about* may be handed to claims that name none themselves: one
    # the caller supplied, a dateline, or one mentioned repeatedly. A single
    # passing mention is not enough.
    article_place: Optional[dict] = None
    if hint_place is not None and hint_place["confidence"] >= min_conf:
        article_place = _public(hint_place, "user")
    elif resolved:
        best = max(resolved, key=lambda cp: (cp[0].dateline, cp[0].count, cp[1]["confidence"]))
        if (best[0].dateline or best[0].count >= 2) and best[1]["confidence"] >= INHERIT_MIN_CONFIDENCE:
            article_place = _public(best[1], "article")

    claim_locations: list[Optional[dict]] = []
    for claim in claims:
        here = [(c, p) for c, p in resolved if _mentions(claim, c.surface)]
        if here:
            # Most specific place wins; the rest ride along as `also`.
            here.sort(key=lambda cp: (_specificity(cp[1]["level"]), -cp[1]["confidence"]))
            primary = here[0][1]
            also = [{"name": p["name"], "level": p["level"]} for _, p in here[1:4]]
            claim_locations.append(_public(primary, "claim", also=also))
        elif (article_place is not None
              and not any(_mentions(claim, name) for name in unresolved)
              and article_place["confidence"] * ARTICLE_DISCOUNT >= INHERIT_MIN_CONFIDENCE):
            claim_locations.append(
                _public(article_place, article_place["source"],
                        confidence=article_place["confidence"] * ARTICLE_DISCOUNT)
            )
        else:
            claim_locations.append(None)

    return LocateResult(claim_locations, article_place, notes)


# ── Search scopes (consumed by the evidence retriever) ────────────────────────

def scopes_for(place: dict) -> list[dict[str, str]]:
    """
    Search scopes for a located claim, most specific first.

    Each is {"level", "term", "country_code"}; the retriever works down the list
    until it has enough regional evidence, which is how an empty result for a
    street widens to its city, then its state, then its country.
    """
    cc = place.get("country_code") or ""
    level = place.get("level") or "country"
    name = place.get("name") or ""
    city = place.get("city")
    district = place.get("district")
    state = place.get("state")
    country = place.get("country")

    scopes: list[tuple[str, Optional[str]]] = []
    if level in ("site", "street", "locality"):
        anchor = city or district or state
        scopes.append(("locality", f"{name} {anchor}".strip() if anchor else name))
        if city:
            scopes.append(("city", city))
    elif level == "city":
        scopes.append(("city", f"{name} {state}".strip() if state and _norm(state) != _norm(name) else name))
    elif level == "district":
        scopes.append(("district", name))
    if level not in ("state", "country") and state:
        scopes.append(("state", state))
    if level == "state":
        scopes.append(("state", name))
    if country:
        scopes.append(("country", country))

    out, seen = [], set()
    for lvl, term in scopes:
        if term and _norm(term) not in seen:
            seen.add(_norm(term))
            out.append({"level": lvl, "term": term, "country_code": cc})
    return out


def compact(place: Optional[dict]) -> Optional[dict]:
    """The location fields worth returning to API clients."""
    if not place:
        return None
    keys = ("name", "label", "level", "lat", "lon", "country", "country_code", "state",
            "district", "city", "locality", "confidence", "method", "source",
            "mentions", "ambiguous", "also")
    return {k: place.get(k) for k in keys if k in place}


# ── Node ──────────────────────────────────────────────────────────────────────

def locate_node(state: PipelineState) -> dict:
    """LangGraph node: attach a location (or None) to every claim."""
    claims = state.get("claims", [])
    try:
        from config import settings
        enabled = bool(settings.REGION_AWARE) and settings.GEOCODER_BACKEND != "off"
    except Exception:  # pragma: no cover
        enabled = True

    if not claims or not enabled:
        return {"claim_locations": [None] * len(claims), "article_location": None,
                "degradations": []}

    result = locate_text(
        state.get("article_text", ""), claims,
        url=state.get("url"), region_hint=state.get("region_hint"),
    )

    degradations: list = []
    if result.notes.get("geocoder_down"):
        degradations += degrade(
            "locator", "geocoder_unavailable",
            "The geocoding service could not be reached, so only places in the "
            "built-in gazetteer (countries, Indian states, major cities) were "
            "recognised. Regional search may be coarser than the text allows.",
            severity="info",
        )
    if result.notes.get("ambiguous"):
        names = ", ".join(sorted(set(result.notes["ambiguous"]))[:4])
        degradations += degrade(
            "locator", "location_ambiguous",
            f"Could not tell which place was meant by: {names}. Where context did "
            "not settle it, the claim was treated as having no location.",
            severity="info",
        )

    n_located = sum(1 for loc in result.claim_locations if loc)
    log.info("locator: %d/%d claims located", n_located, len(claims))
    return {
        "claim_locations": result.claim_locations,
        "article_location": result.article_location,
        "degradations": degradations,
    }
