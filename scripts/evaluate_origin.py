"""
Evaluation harness for the origin trail.

The trail says who made a claim and where it was seen first. That is only worth
showing if it is right, and naming the wrong source is worse than naming none,
so this measures both how often the trail finds the origin and how often what
it names is wrong.

Slices (`--slice`), both built from data already in dataset/:
  publishers  FakeNewsNet politifact_fake.csv. Each fake story's `news_url` is
              the site that published it, so the gold origin is that domain.
              Measures whether retrieval reaches the publisher at all and
              whether the trail calls it the earliest source.
  claimants   LIAR false / pants-fire statements. PolitiFact's `speaker` is the
              gold claimant: a person ("donald-trump") or an anonymous channel
              ("facebook-posts", "chain-email", "viral-image"). Measures the
              fact-check lookup's claimant against it. Half the sample is
              channel speakers, the kind of origin this feature exists for.

Read the numbers with these limits in mind:
  * Both sets are US PolitiFact claims from 2007-2018. Indian claims and recent
    ones are not measured here.
  * NewsAPI only reaches back about a month, and many 2016-18 fake-news sites
    are offline, so `publishers` mostly measures Google News RSS reach. A low
    found-rate there is expected.
  * Without GOOGLE_FACTCHECK_API_KEY no claimant is ever named, so `claimants`
    reports zero recall. The summary prints the lookup status so that run is
    not mistaken for a real result.

Examples:
    python scripts/evaluate_origin.py --slice claimants --n 30
    python scripts/evaluate_origin.py --slice publishers --n 30
    EVIDENCE_CACHE_TTL_DAYS=0 python scripts/evaluate_origin.py --slice claimants   # replay
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import re
import sys
import time
from collections import Counter
from typing import Any, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from config import settings  # noqa: E402
from paper_trail_core.agents import origin_tracer as ot  # noqa: E402
from paper_trail_core.agents.sources import domain_of  # noqa: E402

log = logging.getLogger("evaluate_origin")

RESULTS_DIR = os.path.join(BASE_DIR, "results")
CACHE_DIR = os.path.join(RESULTS_DIR, "eval_cache", "origin")
FAKENEWSNET_PATH = os.path.join(BASE_DIR, "dataset", "FakeNewsNet", "politifact_fake.csv")
LIAR_DIR = os.path.join(BASE_DIR, "dataset", "LIAR Dataset")

LIAR_FALSE = {"false", "pants-fire"}

# PolitiFact speaker slugs for anonymous channels, mapped to the channel a
# fact-check's free-text claimant would name.
CHANNEL_SPEAKERS = {
    "facebook-posts": "facebook",
    "instagram-posts": "instagram",
    "tweets": "twitter",
    "chain-email": "email",
    "reader-email": "email",
    "blog-posting": "blog",
    "viral-image": "image",
    "video": "video",
}

# Free-text claimant -> channel. Order matters: "Facebook post with an image"
# is a Facebook claim.
_CHANNEL_WORDS = [
    ("facebook", ("facebook",)),
    ("instagram", ("instagram",)),
    ("whatsapp", ("whatsapp",)),
    ("twitter", ("twitter", "tweet", "x post", "posts on x")),
    ("tiktok", ("tiktok",)),
    ("video", ("youtube", "video")),
    ("email", ("email", "e-mail", "chain")),
    ("blog", ("blog",)),
    ("image", ("image", "meme", "photo", "picture", "graphic")),
]

# A "viral image" is shared on social platforms, and fact-checkers often name
# the platform instead of the format.
_IMAGE_PLATFORMS = {"image", "facebook", "instagram", "twitter", "whatsapp"}


# ── Slice loading ─────────────────────────────────────────────────────────────

def _stable_order(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministic pseudo-random order, so --n picks the same rows every run."""
    return sorted(rows, key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())


_UNRESOLVABLE_ARCHIVES = {"archive.is", "archive.today", "archive.ph"}
_ARCHIVE_RE = re.compile(r"^(?:https?://)?web\.archive\.org/web/[^/]+/(.+)$", re.I)


def _unwrap_archive(url: str) -> str:
    """Absolute URL of the original page; Wayback snapshots point to the site they archive."""
    m = _ARCHIVE_RE.match(url)
    if m:
        url = m.group(1)
    return url if url.startswith(("http://", "https://")) else "http://" + url


def load_publishers(n: int, path: str = FAKENEWSNET_PATH) -> list[dict[str, Any]]:
    csv.field_size_limit(sys.maxsize)       # tweet_ids lists exceed the default
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            title, url = (r.get("title") or "").strip(), (r.get("news_url") or "").strip()
            if not title or not url:
                continue
            url = _unwrap_archive(url)
            gold = _host(domain_of(url))
            # archive.is short links hide the original site; no usable gold.
            if gold and gold not in _UNRESOLVABLE_ARCHIVES:
                rows.append({"id": r["id"], "slice": "publishers", "claim": title,
                             "gold": gold, "gold_url": url})
    return _stable_order(rows)[:n] if n else _stable_order(rows)


def load_claimants(n: int, liar_dir: str = LIAR_DIR) -> list[dict[str, Any]]:
    channel, person = [], []
    for name in ("train.tsv", "valid.tsv", "test.tsv"):
        path = os.path.join(liar_dir, name)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as fh:
            for parts in csv.reader(fh, delimiter="\t"):
                if len(parts) < 5 or parts[1] not in LIAR_FALSE or not parts[4]:
                    continue
                row = {"id": parts[0], "slice": "claimants", "claim": parts[2].strip(),
                       "gold": parts[4], "gold_context": parts[13] if len(parts) > 13 else ""}
                (channel if parts[4] in CHANNEL_SPEAKERS else person).append(row)
    channel, person = _stable_order(channel), _stable_order(person)
    if not n:
        return channel + person
    half = n // 2
    return channel[:n - half] + person[:half]


# ── Scoring ───────────────────────────────────────────────────────────────────

def channel_of(claimant: str) -> Optional[str]:
    text = (claimant or "").lower()
    for channel, words in _CHANNEL_WORDS:
        if any(w in text for w in words):
            return channel
    return None


def claimant_matches(gold_speaker: str, predicted: Optional[str]) -> bool:
    """Whether a fact-check's free-text claimant names PolitiFact's speaker."""
    if not predicted:
        return False
    gold_channel = CHANNEL_SPEAKERS.get(gold_speaker)
    if gold_channel:
        got = channel_of(predicted)
        if gold_channel == "image":
            return got in _IMAGE_PLATFORMS
        return got == gold_channel
    # A person: every distinctive part of the slug must appear ("donald-trump").
    # Known leniency: "Donald Trump Jr." also matches; rare enough in LIAR to accept.
    tokens = [t for t in gold_speaker.lower().split("-") if len(t) > 2]
    text = predicted.lower()
    return bool(tokens) and all(t in text for t in tokens)


def _host(domain: str) -> str:
    return (domain or "").lower().split(":", 1)[0].removeprefix("www.")


def same_site(a: str, b: str) -> bool:
    """Same site, allowing subdomains either way (www., m., amp.) and ignoring ports."""
    a, b = _host(a), _host(b)
    if not a or not b:
        return False
    return a == b or a.endswith("." + b) or b.endswith("." + a)


def summarise_origins(claim_verdicts: list[dict[str, Any]]) -> dict[str, Any]:
    """Fold every claim's origin in one pipeline result into what is scored."""
    origins = [cv["origin"] for cv in claim_verdicts if cv.get("origin")]
    claimant = next((o["claimant"] for o in origins if o.get("claimant")), None)
    earliest = sorted(
        (o["earliest_seen"] for o in origins if o.get("earliest_seen")),
        key=lambda e: e.get("date") or "9999",
    )
    domains, without_search = [], []
    wider_hits = 0
    for o in origins:
        for e in o.get("timeline") or []:
            if e.get("found_by") == "wider_search":
                wider_hits += 1
            elif e.get("domain") and e["domain"] not in without_search:
                without_search.append(e["domain"])
            if e.get("domain") and e["domain"] not in domains:
                domains.append(e["domain"])
    bases = [o.get("basis", "none") for o in origins]
    statuses = [o.get("fact_check_status", "skipped") for o in origins]
    search_statuses = [o.get("search_status", "skipped") for o in origins]
    return {
        "has_origin": bool(origins),
        "basis": next((b for b in ("fact_check", "earliest_indexed") if b in bases), "none"),
        "fact_check_status": statuses[0] if statuses else "not_run",
        "claimant": claimant,
        "earliest_domain": earliest[0].get("domain") if earliest else None,
        "timeline_domains": domains,
        # The same trail as it was before the wider search existed, so one run
        # measures what that search adds.
        "domains_without_search": without_search,
        "wider_hits": wider_hits,
        "search_status": search_statuses[0] if search_statuses else "not_run",
    }


def score(row: dict[str, Any], origin: dict[str, Any]) -> dict[str, Any]:
    out = {"traced": origin["basis"] != "none"}
    if row["slice"] == "claimants":
        out["claimant_named"] = bool(origin["claimant"])
        out["claimant_correct"] = claimant_matches(row["gold"], origin["claimant"])
    else:
        out["publisher_found"] = any(same_site(d, row["gold"]) for d in origin["timeline_domains"])
        out["publisher_found_without_search"] = any(
            same_site(d, row["gold"]) for d in origin.get("domains_without_search", []))
        out["earliest_correct"] = same_site(origin["earliest_domain"] or "", row["gold"])
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval: honest error bars for the small samples used here."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4))


def _rate(records: list[dict], key: str, among: Optional[str] = None) -> dict[str, Any]:
    pool = [r for r in records if r.get(among)] if among else records
    k = sum(1 for r in pool if r.get(key))
    return {"k": k, "n": len(pool), "rate": round(k / len(pool), 4) if pool else None,
            "ci95": wilson(k, len(pool))}


def build_report(slice_name: str, records: list[dict]) -> dict[str, Any]:
    ok = [r for r in records if not r.get("error")]
    report: dict[str, Any] = {
        "slice": slice_name,
        "n": len(records),
        "errors": len(records) - len(ok),
        "fact_check_status": dict(Counter(r.get("fact_check_status") for r in ok)),
        "search_status": dict(Counter(r.get("search_status") for r in ok)),
        "wider_search_added_rows": _rate([dict(r, added=bool(r.get("wider_hits"))) for r in ok], "added"),
        "called_supported": _rate(ok, "called_supported"),
        "traced": _rate(ok, "traced"),
    }
    if slice_name == "claimants":
        report["claimant_named"] = _rate(ok, "claimant_named")
        # Precision is the honesty number: of the claimants shown, how many were right.
        report["claimant_precision"] = _rate(ok, "claimant_correct", among="claimant_named")
        report["claimant_recall"] = _rate(ok, "claimant_correct")
        for group, is_channel in (("channel", True), ("person", False)):
            sub = [r for r in ok if (r["gold"] in CHANNEL_SPEAKERS) == is_channel]
            report[f"claimant_recall_{group}"] = _rate(sub, "claimant_correct")
    else:
        report["publisher_found"] = _rate(ok, "publisher_found")
        report["publisher_found_without_search"] = _rate(ok, "publisher_found_without_search")
        report["earliest_correct"] = _rate(ok, "earliest_correct")
        report["earliest_precision"] = _rate(
            [dict(r, has_earliest=bool(r.get("earliest_domain"))) for r in ok],
            "earliest_correct", among="has_earliest")
    return report


# ── Running ───────────────────────────────────────────────────────────────────

def _cache_key(claim: str) -> str:
    # Includes whether fact-check lookup was possible, so adding a key later
    # re-runs claims instead of replaying their keyless results.
    cfg = json.dumps({
        "nli": settings.NLI_BACKEND, "evidence": settings.EVIDENCE_BACKEND,
        "rel_min": settings.RELEVANCE_MIN, "origin": settings.ORIGIN_TRACE,
        "factcheck": bool(settings.GOOGLE_FACTCHECK_API_KEY),
        "fc_min": settings.FACTCHECK_MIN_RELEVANCE,
        "wide": settings.ORIGIN_WIDE_SEARCH,
        "wide_filter": [ot.WIDER_MIN_RELEVANCE, ot.WIDER_MIN_COVERAGE, ot.WIDER_MIN_STANCE],
    }, sort_keys=True)
    return hashlib.sha256((cfg + "||" + claim).encode("utf-8")).hexdigest()


def run_one(row: dict[str, Any], use_cache: bool = True) -> dict[str, Any]:
    key = _cache_key(row["claim"])
    path = os.path.join(CACHE_DIR, key[:2], f"{key}.json")
    origin = None
    if use_cache and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                origin = json.load(fh)
        except (OSError, json.JSONDecodeError):
            origin = None

    if origin is None:
        from paper_trail_core.agents.pipeline import run_verification_pipeline
        try:
            vr = run_verification_pipeline(row["claim"])
        except Exception as exc:  # noqa: BLE001 — one bad row must not end the run
            log.warning("%s failed: %s", row["id"], exc)
            return {**row, "error": type(exc).__name__}
        origin = summarise_origins(vr.claim_verdicts)
        origin["called_supported"] = vr.verdict == "likely_real"
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(f"{path}.tmp", "w", encoding="utf-8") as fh:
                json.dump(origin, fh)
            os.replace(f"{path}.tmp", path)
        except OSError as exc:
            log.warning("could not cache %s (%s)", row["id"], exc)

    return {**row, **origin, **score(row, origin), "error": ""}


def render_markdown(report: dict[str, Any]) -> str:
    def line(name: str, r: dict) -> str:
        if r["rate"] is None:
            return f"| {name} | n/a | 0 | |"
        lo, hi = r["ci95"]
        return f"| {name} | {r['rate']:.1%} | {r['k']}/{r['n']} | {lo:.0%}–{hi:.0%} |"

    rows = [k for k, v in report.items() if isinstance(v, dict) and "rate" in v]
    out = [
        f"# Origin trail evaluation: {report['slice']}",
        "",
        f"n={report['n']}, errors={report['errors']}, "
        f"fact-check lookup status: {report['fact_check_status']}, "
        f"wider search status: {report['search_status']}",
        "",
        "| metric | rate | count | 95% CI |",
        "|---|---|---|---|",
        *[line(k, report[k]) for k in rows],
        "",
    ]
    if report["fact_check_status"].get("disabled"):
        out.append("**Fact-check lookup was disabled for some claims (no "
                   "GOOGLE_FACTCHECK_API_KEY), so claimant metrics understate the feature.**")
    return "\n".join(out) + "\n"


def write_csv(records: list[dict], path: str) -> None:
    cols = ["id", "slice", "claim", "gold", "error", "called_supported", "traced",
            "basis", "fact_check_status", "claimant", "claimant_named", "claimant_correct",
            "earliest_domain", "publisher_found", "publisher_found_without_search",
            "earliest_correct", "search_status", "wider_hits", "timeline_domains"]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow({**r, "timeline_domains": " ".join(r.get("timeline_domains") or [])})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slice", default="claimants", choices=["claimants", "publishers"])
    ap.add_argument("--n", type=int, default=30,
                    help="claims to evaluate (0 = all). NewsAPI's free tier is "
                         "100 requests/day, so keep this small on a first run.")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for output filenames")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log.setLevel(logging.INFO)

    rows = (load_claimants if args.slice == "claimants" else load_publishers)(args.n)
    if not settings.GOOGLE_FACTCHECK_API_KEY:
        log.warning("GOOGLE_FACTCHECK_API_KEY is not set: no claimant can be named.")
    log.info("evaluating %d claims from slice=%s", len(rows), args.slice)

    started = time.time()
    records = []
    for i, row in enumerate(rows, 1):
        records.append(run_one(row, not args.no_cache))
        if i % 10 == 0 or i == len(rows):
            log.info("  %d/%d (%.0fs elapsed)", i, len(rows), time.time() - started)

    report = build_report(args.slice, records)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    base = os.path.join(RESULTS_DIR, f"eval_origin_{args.slice}{tag}")
    with open(base + ".json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    with open(base + ".md", "w", encoding="utf-8") as fh:
        fh.write(render_markdown(report))
    write_csv(records, base + ".csv")

    print("\n" + render_markdown(report))
    print(f"wrote {os.path.relpath(base, BASE_DIR)}.{{json,md,csv}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
