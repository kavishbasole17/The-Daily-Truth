"""
Fetch a news article from a URL and extract its readable text.

Used by "check a link" (POST /articles/submit with only `url`). The server
fetches a URL a user typed, so this is guarded against SSRF:

  * only http/https, default ports 80/443 or an explicit public port
  * every hostname is resolved and *all* its addresses must be public
    (no loopback, private, link-local, multicast, reserved or metadata IPs)
  * redirects are followed by hand (max 4) so each hop is re-checked
  * a hard response-size cap and timeouts

Extraction uses `trafilatura` when installed (best quality) and falls back to
a small stdlib HTML parser that keeps <title>/og:title and <p> text.
"""
from __future__ import annotations

import ipaddress
import logging
import re
import socket
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

import requests

log = logging.getLogger(__name__)

MAX_BYTES = 3 * 1024 * 1024
MAX_REDIRECTS = 4
TIMEOUT = (5, 15)            # connect, read (seconds)
MIN_TEXT_CHARS = 200
USER_AGENT = "Mozilla/5.0 (compatible; DailyTruthBot/1.0; +fact-checking)"


class FetchError(Exception):
    """A user-presentable reason the article couldn't be fetched."""


@dataclass
class FetchedArticle:
    url: str          # final URL after redirects
    title: str
    text: str


# ── URL safety ──────────────────────────────────────────────────────────────────

def _is_public_ip(ip: str) -> bool:
    addr = ipaddress.ip_address(ip)
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
        addr = addr.ipv4_mapped
    return not (
        addr.is_private or addr.is_loopback or addr.is_link_local
        or addr.is_multicast or addr.is_reserved or addr.is_unspecified
    )


def validate_url(url: str) -> str:
    """Raise FetchError unless `url` is an http(s) URL on a public host."""
    url = (url or "").strip()
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise FetchError("Only http:// and https:// links can be checked.")
    if not parsed.hostname:
        raise FetchError("That doesn't look like a valid link.")
    if parsed.username or parsed.password:
        raise FetchError("Links with embedded credentials aren't allowed.")
    host = parsed.hostname
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError:
        raise FetchError("That link has an invalid port.")
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        raise FetchError(f"Couldn't find the website '{host}'.")
    ips = {info[4][0] for info in infos}
    if not ips or not all(_is_public_ip(ip) for ip in ips):
        raise FetchError("That link points to a private or local address.")
    return url


# ── Extraction ──────────────────────────────────────────────────────────────────

class _FallbackParser(HTMLParser):
    """Keeps the page title and paragraph text; skips scripts, nav, footers."""

    _SKIP = frozenset({"script", "style", "noscript", "nav", "footer", "header", "aside", "form"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.og_title = ""
        self.paragraphs: list[str] = []
        self._buf: list[str] = []
        self._in_title = False
        self._in_p = 0
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag == "p":
            self._in_p += 1
            self._buf = []
        elif tag == "meta":
            a = dict(attrs)
            if a.get("property") == "og:title" and a.get("content"):
                self.og_title = a["content"].strip()

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip:
            self._skip -= 1
        elif tag == "title":
            self._in_title = False
        elif tag == "p" and self._in_p:
            self._in_p -= 1
            text = " ".join("".join(self._buf).split())
            if len(text) >= 40:
                self.paragraphs.append(text)
            self._buf = []

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif self._in_p and not self._skip:
            self._buf.append(data)


def extract_text(html: str, url: str = "") -> tuple[str, str]:
    """Return (title, text) from an HTML page."""
    try:
        import trafilatura  # type: ignore
        text = trafilatura.extract(html, url=url or None, include_comments=False,
                                   include_tables=False) or ""
        meta = trafilatura.extract_metadata(html)
        title = (getattr(meta, "title", None) or "") if meta else ""
        if len(text) >= MIN_TEXT_CHARS:
            return title.strip(), text.strip()
    except ImportError:
        pass
    except Exception as exc:  # pragma: no cover - trafilatura edge cases
        log.warning("trafilatura failed on %s: %s", url, exc)

    p = _FallbackParser()
    p.feed(html)
    title = p.og_title or " ".join(p.title.split())
    return title, "\n\n".join(p.paragraphs)


# ── Fetch ───────────────────────────────────────────────────────────────────────

def _get(url: str) -> requests.Response:
    return requests.get(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
        timeout=TIMEOUT,
        allow_redirects=False,
        stream=True,
    )


def fetch_article(url: str) -> FetchedArticle:
    """Fetch and extract an article. Raises FetchError with a readable reason."""
    current = validate_url(url)
    for _ in range(MAX_REDIRECTS + 1):
        try:
            resp = _get(current)
        except requests.Timeout:
            raise FetchError("The website took too long to respond.")
        except requests.RequestException:
            raise FetchError("Couldn't connect to that website.")

        if resp.is_redirect or resp.status_code in (301, 302, 303, 307, 308):
            location = resp.headers.get("location")
            resp.close()
            if not location:
                raise FetchError("The website redirected without a destination.")
            current = validate_url(urljoin(current, location))
            continue
        break
    else:
        raise FetchError("Too many redirects.")

    try:
        if resp.status_code >= 400:
            raise FetchError(f"The website returned an error ({resp.status_code}).")
        ctype = resp.headers.get("content-type", "").lower()
        if ctype and "html" not in ctype and "xml" not in ctype:
            raise FetchError("That link isn't a web page (it may be a PDF, image or video).")

        chunks, size = [], 0
        for chunk in resp.iter_content(chunk_size=64 * 1024):
            size += len(chunk)
            if size > MAX_BYTES:
                raise FetchError("That page is too large to check.")
            chunks.append(chunk)
        raw = b"".join(chunks)
    finally:
        resp.close()

    encoding = resp.encoding or "utf-8"
    m = re.search(rb'<meta[^>]+charset=["\']?([\w-]+)', raw[:4096], re.IGNORECASE)
    if m:
        encoding = m.group(1).decode("ascii", "ignore") or encoding
    try:
        html = raw.decode(encoding, errors="replace")
    except LookupError:
        html = raw.decode("utf-8", errors="replace")

    title, text = extract_text(html, current)
    if len(text) < MIN_TEXT_CHARS:
        raise FetchError(
            "Couldn't find article text on that page. It may need a login, be behind "
            "a paywall, or load its content with JavaScript — paste the text instead."
        )
    return FetchedArticle(url=current, title=title[:300], text=text)


def looks_like_url(value: str | None) -> bool:
    return bool(value) and bool(re.fullmatch(r"https?://\S+", value.strip(), re.IGNORECASE))
