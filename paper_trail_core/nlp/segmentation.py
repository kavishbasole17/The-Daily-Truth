"""
Sentence segmentation.

The previous claim extractor did `text.split(".")`, which turned

    "raised its benchmark repo rate by 25 basis points to 6.75 percent"

into two "claims":

    "...raised its benchmark repo rate by 25 basis points to 6."
    "75 percent on Thursday, citing persistent food inflation."

Every article containing a decimal — which is essentially every news article
worth fact-checking — was shredded into fragments before it ever reached
retrieval. Those fragments then became search queries.

This is a self-contained regex segmenter rather than nltk/spaCy: both need a
runtime model download, neither is currently installed, and a capstone that
runs offline and deterministically is worth more here than the last few
points of segmentation accuracy. The approach is placeholder substitution —
mask every period that is *not* a sentence boundary, split, then unmask.
"""
from __future__ import annotations

import re

# Sentinel that cannot occur in input text (NUL is stripped by every loader).
_DOT = "\x00"
_URL = "\x01"

_ABBREVIATIONS = (
    # Titles
    "Mr", "Mrs", "Ms", "Dr", "Prof", "Sr", "Jr", "St", "Rev", "Hon",
    # Military / political
    "Gen", "Sen", "Rep", "Gov", "Lt", "Col", "Capt", "Adm", "Maj", "Sgt", "Cmdr",
    # Corporate
    "Inc", "Ltd", "Corp", "Co", "plc", "LLC",
    # Months
    "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep", "Sept", "Oct", "Nov", "Dec",
    # Editorial
    "vs", "etc", "approx", "est", "No", "Vol", "pp", "Fig", "Dept", "Univ", "Assn",
    "Ave", "Blvd", "Rd", "Mt", "Ft",
)

_RE_URL = re.compile(r"\b(?:https?://|www\.)\S+|\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_RE_DECIMAL = re.compile(r"(?<=\d)\.(?=\d)")
_RE_ABBREV = re.compile(r"\b(" + "|".join(_ABBREVIATIONS) + r")\.", re.IGNORECASE)
# Dotted acronyms: U.S., U.K., e.g., i.e., a.m., p.m., A.P.J.
_RE_ACRONYM = re.compile(r"\b(?:[A-Za-z]\.){2,}")
# A lone capital followed by another capitalised token — "J. Smith", "J. R. R. Tolkien"
_RE_INITIAL = re.compile(r"\b([A-Z])\.(?=\s+[A-Z])")
_RE_ELLIPSIS = re.compile(r"\.\.\.+")

# Boundary: terminal punctuation, optional closing quotes/brackets, whitespace,
# then something that looks like the start of a new sentence.
_RE_BOUNDARY = re.compile(r'(?<=[.!?])["\'”’)\]]*\s+(?=["\'“‘(\[]?[A-Z0-9])')

_RE_WS = re.compile(r"[ \t]+")

MIN_SENTENCE_CHARS = 25
MAX_SENTENCE_CHARS = 600


def normalise(text: str) -> str:
    """Fold the typographic variants that show up across the datasets."""
    if not isinstance(text, str):
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = (text
            .replace("‘", "'").replace("’", "'")
            .replace("“", '"').replace("”", '"')
            .replace("–", "-").replace("—", " - ")
            .replace(" ", " "))
    text = _RE_WS.sub(" ", text)
    return text.strip()


def _protect(text: str) -> tuple[str, list[str]]:
    """Mask non-boundary periods (and whole URLs) ahead of splitting."""
    urls: list[str] = []

    def _stash_url(m: re.Match) -> str:
        urls.append(m.group(0))
        return f"{_URL}{len(urls) - 1}{_URL}"

    text = _RE_URL.sub(_stash_url, text)
    text = _RE_ELLIPSIS.sub(lambda m: _DOT * len(m.group(0)), text)
    text = _RE_DECIMAL.sub(_DOT, text)
    text = _RE_ABBREV.sub(lambda m: m.group(1) + _DOT, text)
    text = _RE_ACRONYM.sub(lambda m: m.group(0).replace(".", _DOT), text)
    text = _RE_INITIAL.sub(lambda m: m.group(1) + _DOT, text)
    return text, urls


def _restore(text: str, urls: list[str]) -> str:
    text = text.replace(_DOT, ".")
    for i, url in enumerate(urls):
        text = text.replace(f"{_URL}{i}{_URL}", url)
    return text


def _split_overlong(sentence: str) -> list[str]:
    """
    Break up run-on text with no usable terminal punctuation.

    Scraped articles (ISOT's Fake.csv especially) sometimes arrive as one
    enormous unpunctuated block. Left alone it becomes a single "claim" that
    is useless as a retrieval query and gets truncated by the NLI tokenizer.
    """
    if len(sentence) <= MAX_SENTENCE_CHARS:
        return [sentence]

    parts: list[str] = []
    for chunk in re.split(r";\s+|,\s+and\s+", sentence):
        chunk = chunk.strip()
        if not chunk:
            continue
        if len(chunk) > MAX_SENTENCE_CHARS:
            chunk = chunk[:MAX_SENTENCE_CHARS].rsplit(" ", 1)[0]
        parts.append(chunk)
    return parts or [sentence[:MAX_SENTENCE_CHARS]]


def segment_sentences(text: str, min_chars: int = MIN_SENTENCE_CHARS) -> list[str]:
    """
    Split `text` into sentences, keeping decimals and abbreviations intact.

    Returns sentences in document order with surrounding whitespace stripped.
    Fragments shorter than `min_chars` are dropped — they are almost always
    datelines, bylines or list bullets, and never make checkable claims.
    """
    text = normalise(text)
    if not text:
        return []

    protected, urls = _protect(text)

    out: list[str] = []
    for line in protected.split("\n"):
        line = line.strip()
        if not line:
            continue
        for raw in _RE_BOUNDARY.split(line):
            sentence = _restore(raw, urls).strip()
            if not sentence:
                continue
            for piece in _split_overlong(sentence):
                piece = piece.strip()
                if len(piece) >= min_chars:
                    out.append(piece)
    return out
