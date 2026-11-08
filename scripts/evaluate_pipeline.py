"""
End-to-end evaluation harness for the agentic verification pipeline.

This is the script that makes every accuracy claim in this project checkable.
Before it existed there was no way to tell whether a change to the pipeline
helped or hurt, which is how a verifier that scored contradicting evidence as
support survived unnoticed.

Slices (`--slice`):
  golden      Hand-written articles in dataset/golden/golden_cases.jsonl. The
              only set guaranteed absent from every training corpus, so it is
              the honest read on novel input — but it is small, so treat it as
              a regression suite rather than a population estimate.
  holdout     The 20% partition the frozen models were never trained on,
              reproduced exactly (see paper_trail_core/eval/splits.py). Large
              and realistic. Adversarial and BuzzFeed rows are excluded from
              the headline; see that module for why.
  adversarial The full adversarial CSV, reported *separately and explicitly*
              as a memorisation baseline. ~80% of it is training data, so a
              high score here measures recall of training rows, nothing more.

Channels compared on every run:
  fused        the full pipeline verdict
  ensemble     the style classifier alone (what the system used to lean on)
  evidence     the evidence signal alone, ensemble pinned to 0.5

That ablation is the point: it shows whether the agentic layer earns its cost.

Examples:
    py scripts/evaluate_pipeline.py --slice golden
    py scripts/evaluate_pipeline.py --slice holdout --n-per-class 100
    py scripts/evaluate_pipeline.py --slice holdout --n-per-class 150 --workers 4
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from config import settings
from paper_trail_core.agents.features import FEATURE_NAMES
from paper_trail_core.agents.fusion import heuristic_probability
from paper_trail_core.agents.pipeline import run_verification_pipeline
from paper_trail_core.eval import metrics as M

log = logging.getLogger("evaluate_pipeline")

RESULTS_DIR = os.path.join(BASE_DIR, "results")
CACHE_DIR = os.path.join(RESULTS_DIR, "eval_cache")
GOLDEN_PATH = os.path.join(BASE_DIR, "dataset", "golden", "golden_cases.jsonl")
ADVERSARIAL_PATH = os.path.join(BASE_DIR, "dataset", "Adversarial", "adversarial_style.csv")


# ── Slice loading ─────────────────────────────────────────────────────────────

def load_golden() -> list[dict[str, Any]]:
    rows = []
    with open(GOLDEN_PATH, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            rows.append({
                "id": case["id"], "text": case["text"], "label": int(case["label"]),
                "source": f"golden/{case.get('category', 'na')}",
            })
    return rows


def load_adversarial() -> list[dict[str, Any]]:
    rows = []
    with open(ADVERSARIAL_PATH, "r", encoding="utf-8") as fh:
        for i, r in enumerate(csv.DictReader(fh)):
            rows.append({"id": f"adv{i:04d}", "text": r["text_content"],
                         "label": int(r["label"]), "source": "Adversarial"})
    return rows


def load_holdout(n_per_class: int) -> list[dict[str, Any]]:
    from paper_trail_core.eval.splits import (
        get_holdout_frame,
        headline_frame,
        stratified_sample,
    )
    frame = headline_frame(get_holdout_frame())
    sample = stratified_sample(frame, n_per_class=n_per_class)
    return [
        {"id": f"ho{i:05d}", "text": row.text_content,
         "label": int(row.label), "source": str(row.source)}
        for i, row in enumerate(sample.itertuples())
    ]


# ── Per-article execution, with caching ───────────────────────────────────────

def _cache_key(text: str) -> str:
    """
    Keyed on the article text *and* the pipeline configuration.

    Including config is what stops a stale cache from silently reporting the
    old pipeline's numbers after a threshold or backend change — the exact
    class of mistake this harness exists to catch.
    """
    cfg = json.dumps({
        "claim": settings.CLAIM_BACKEND, "nli": settings.NLI_BACKEND,
        "evidence": settings.EVIDENCE_BACKEND, "ensemble": settings.ENSEMBLE_BACKEND,
        "nli_model": settings.NLI_MODEL, "rel_min": settings.RELEVANCE_MIN,
        "max_claims": settings.MAX_CLAIMS, "checkworthy": settings.CHECKWORTHY_MIN,
    }, sort_keys=True)
    return hashlib.sha256((cfg + "||" + text).encode("utf-8")).hexdigest()


def run_one(row: dict[str, Any], use_cache: bool = True) -> dict[str, Any]:
    key = _cache_key(row["text"])
    path = os.path.join(CACHE_DIR, key[:2], f"{key}.json")

    if use_cache and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                cached = json.load(fh)
            cached.update({"id": row["id"], "label": row["label"], "source": row["source"]})
            return cached
        except (OSError, json.JSONDecodeError):
            pass

    started = time.time()
    vr = run_verification_pipeline(row["text"])
    feats = vr.fusion_features or {}

    record = {
        "id": row["id"], "label": row["label"], "source": row["source"],
        "verdict": vr.verdict, "confidence": vr.confidence,
        "fused_score": vr.fused_score, "fusion_method": vr.fusion_method,
        "ensemble_score": vr.ensemble_score,
        "n_claims": len(vr.claim_verdicts),
        "n_degradations": len(vr.degradations),
        "degradation_kinds": sorted({d["kind"] for d in vr.degradations}),
        "claim_labels": [cv.get("verdict") for cv in vr.claim_verdicts],
        "wall_s": round(time.time() - started, 3),
        **{f"f_{k}": feats.get(k, 0.0) for k in FEATURE_NAMES},
    }

    try:
        # Written via a unique temp file then renamed: with --workers > 1
        # several threads write this directory concurrently, and a partially
        # written record would be silently reused as a valid cached result on
        # the next run.
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(record, fh)
        os.replace(tmp, path)
    except OSError as exc:
        log.warning("could not cache %s (%s)", row["id"], exc)
    return record


# ── Ablation channels ─────────────────────────────────────────────────────────

def channel_scores(records: list[dict]) -> dict[str, list[float]]:
    """
    P(real) under each channel, for the same articles.

    `evidence` re-runs the heuristic fusion with the ensemble pinned at 0.5,
    isolating what the agentic layer contributes on its own.
    """
    fused = [r["fused_score"] for r in records]
    ensemble = [r["ensemble_score"] for r in records]

    evidence = []
    for r in records:
        feats = {k: r.get(f"f_{k}", 0.0) for k in FEATURE_NAMES}
        feats["ensemble_score"] = 0.5
        evidence.append(heuristic_probability(feats))

    return {"fused": fused, "ensemble_only": ensemble, "evidence_only": evidence}


# ── Reporting ─────────────────────────────────────────────────────────────────

def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=BASE_DIR, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _diagnostics(records: list[dict]) -> dict[str, Any]:
    n = len(records)
    claim_labels = [lbl for r in records for lbl in r["claim_labels"]]
    kinds: dict[str, int] = {}
    for r in records:
        for k in r["degradation_kinds"]:
            kinds[k] = kinds.get(k, 0) + 1
    walls = sorted(r["wall_s"] for r in records)
    return {
        "articles": n,
        "mean_claims_per_article": round(sum(r["n_claims"] for r in records) / max(1, n), 2),
        "mean_coverage": round(sum(r.get("f_coverage", 0.0) for r in records) / max(1, n), 4),
        "mean_relevance": round(sum(r.get("f_mean_relevance", 0.0) for r in records) / max(1, n), 4),
        "claim_label_counts": {
            lbl: claim_labels.count(lbl)
            for lbl in ("entailment", "contradiction", "neutral", "unverifiable")
        },
        "degradation_article_counts": dict(sorted(kinds.items(), key=lambda kv: -kv[1])),
        "abstention_rate": round(sum(1 for r in records if r["verdict"] == "uncertain") / max(1, n), 4),
        "wall_s_p50": walls[len(walls) // 2] if walls else 0.0,
        "wall_s_p95": walls[int(len(walls) * 0.95)] if walls else 0.0,
    }


def _by_source(records: list[dict]) -> dict[str, Any]:
    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(r["source"], []).append(r)
    out = {}
    for src, rows in sorted(groups.items()):
        y = [r["label"] for r in rows]
        p = [r["fused_score"] for r in rows]
        out[src] = {"n": len(rows), **M.classification_metrics(
            y, [1 if x >= 0.5 else 0 for x in p])}
    return out


def _verdict_metrics(records: list[dict]) -> dict[str, Any]:
    """
    Accuracy of the verdicts as actually delivered, honouring abstention.

    Distinct from thresholding fused_score at 0.5: the pipeline's hard gates
    (degraded NLI, zero coverage) force abstention regardless of score, and
    this is what a user actually receives.
    """
    committed = [r for r in records if r["verdict"] != "uncertain"]
    y = [r["label"] for r in committed]
    pred = [1 if r["verdict"] == "likely_real" else 0 for r in committed]
    m = M.classification_metrics(y, pred) if committed else {"n": 0}
    return {
        "coverage": round(len(committed) / max(1, len(records)), 4),
        "abstained": len(records) - len(committed),
        **m,
    }


def build_report(slice_name: str, records: list[dict], target_precision: float) -> dict[str, Any]:
    y = [r["label"] for r in records]
    channels = channel_scores(records)

    report: dict[str, Any] = {
        "slice": slice_name,
        "generated_at": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "config": {
            "claim_backend": settings.CLAIM_BACKEND,
            "nli_backend": settings.NLI_BACKEND,
            "nli_model": settings.NLI_MODEL,
            "evidence_backend": settings.EVIDENCE_BACKEND,
            "ensemble_backend": settings.ENSEMBLE_BACKEND,
            "relevance_min": settings.RELEVANCE_MIN,
            "max_claims": settings.MAX_CLAIMS,
            "target_precision": target_precision,
        },
        "n": len(records),
        "class_balance": {"real": sum(y), "fake": len(y) - sum(y)},
        "channels": {name: M.summarise_scores(y, scores) for name, scores in channels.items()},
        "delivered_verdicts": _verdict_metrics(records),
        "operating_point": M.select_thresholds(y, channels["fused"], target_precision),
        "coverage_precision_curve": M.coverage_precision_curve(y, channels["fused"]),
        "diagnostics": _diagnostics(records),
        "by_source": _by_source(records),
    }
    return report


def _fmt_channel(name: str, m: dict) -> str:
    if not m.get("n"):
        return f"| {name} | – | – | – | – | – | – |\n"
    return (
        f"| {name} | {m['accuracy']:.3f} | {m['macro_f1']:.3f} | "
        f"{m['roc_auc'] if m['roc_auc'] is not None else float('nan'):.3f} | "
        f"{m['brier']:.3f} | {m['ece']:.3f} | "
        f"{m['fake']['precision']:.3f} / {m['real']['precision']:.3f} |\n"
    )


def render_markdown(report: dict[str, Any]) -> str:
    md = [f"# Pipeline evaluation — `{report['slice']}`\n"]
    md.append(f"Generated {report['generated_at']} · commit `{report['git_sha']}`\n")
    md.append(f"**{report['n']} articles** "
              f"({report['class_balance']['real']} real / {report['class_balance']['fake']} fake)\n")

    if report["slice"] == "adversarial":
        md.append(
            "> **This is a memorisation baseline, not a generalisation result.** "
            "Roughly 80% of these rows were in the models' training split "
            "(oversampled 15x), and the remainder is template-identical to them. "
            "A high score here measures recall of training data.\n"
        )

    md.append("\n## Channel comparison\n")
    md.append("Threshold 0.5, no abstention — the headline accuracy number.\n\n")
    md.append("| channel | accuracy | macro F1 | ROC-AUC | Brier | ECE | precision fake / real |\n")
    md.append("|---|---|---|---|---|---|---|\n")
    for name in ("ensemble_only", "evidence_only", "fused"):
        md.append(_fmt_channel(name, report["channels"][name]))

    dv = report["delivered_verdicts"]
    md.append("\n## Verdicts as delivered (with abstention)\n\n")
    if dv.get("n"):
        md.append(f"- Committed on **{dv['coverage']:.1%}** of articles "
                  f"({dv['abstained']} abstained as uncertain)\n")
        md.append(f"- Accuracy on committed verdicts: **{dv['accuracy']:.3f}**\n")
        md.append(f"- Precision — fake: **{dv['fake']['precision']:.3f}**, "
                  f"real: **{dv['real']['precision']:.3f}**\n")
        c = dv["confusion"]
        md.append(f"- Confusion (committed): tp={c['tp']} tn={c['tn']} fp={c['fp']} fn={c['fn']}\n")
    else:
        md.append("- The pipeline abstained on every article.\n")

    op = report["operating_point"]
    md.append("\n## Operating point\n\n")
    md.append(f"Target precision {op['target_precision']:.2f} — "
              f"{'**met**' if op['target_met'] else '**NOT met**'}.\n\n")
    md.append(f"- Abstention band: `{op['low']:.3f} .. {op['high']:.3f}`\n")
    md.append(f"- Coverage at that band: **{op['coverage']:.1%}**\n")
    md.append(f"- Worst-class precision: **{op['min_precision']:.3f}**\n")

    md.append("\n## Coverage vs. precision\n\n")
    md.append("| band | coverage | accuracy | precision fake | precision real |\n|---|---|---|---|---|\n")
    for pt in report["coverage_precision_curve"][::4]:
        md.append(f"| {pt['low']:.2f}–{pt['high']:.2f} | {pt['coverage']:.1%} | "
                  f"{pt['accuracy']:.3f} | {pt['precision_fake']:.3f} | {pt['precision_real']:.3f} |\n")

    md.append("\n## Calibration (fused)\n\n")
    md.append("| predicted P(real) | n | mean confidence | observed | gap |\n|---|---|---|---|---|\n")
    for b in report["channels"]["fused"]["reliability"]:
        if not b["n"]:
            continue
        md.append(f"| {b['bin']} | {b['n']} | {b['mean_confidence']:.3f} | "
                  f"{b['observed_accuracy']:.3f} | {b['gap']:.3f} |\n")

    d = report["diagnostics"]
    md.append("\n## Pipeline diagnostics\n\n")
    md.append(f"- Mean claims per article: {d['mean_claims_per_article']}\n")
    md.append(f"- Mean evidence coverage: {d['mean_coverage']:.1%}\n")
    md.append(f"- Mean evidence relevance: {d['mean_relevance']:.3f}\n")
    md.append(f"- Claim verdicts: {d['claim_label_counts']}\n")
    md.append(f"- Abstention rate: {d['abstention_rate']:.1%}\n")
    md.append(f"- Latency p50 / p95: {d['wall_s_p50']:.1f}s / {d['wall_s_p95']:.1f}s\n")
    if d["degradation_article_counts"]:
        md.append(f"- Degradations (articles affected): {d['degradation_article_counts']}\n")

    md.append("\n## By source\n\n| source | n | accuracy | macro F1 |\n|---|---|---|---|\n")
    for src, m in report["by_source"].items():
        md.append(f"| {src} | {m['n']} | {m['accuracy']:.3f} | {m['macro_f1']:.3f} |\n")

    return "".join(md)


def write_feature_csv(records: list[dict], path: str) -> None:
    """Per-article feature rows — this file is the fusion training set."""
    cols = ["id", "source", "label", "fused_score", "ensemble_score", "verdict",
            "confidence"] + [f"f_{k}" for k in FEATURE_NAMES]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(r)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slice", default="golden", choices=["golden", "holdout", "adversarial"])
    ap.add_argument("--n-per-class", type=int, default=100,
                    help="holdout only: articles sampled per class")
    ap.add_argument("--limit", type=int, default=0, help="cap articles (debugging)")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel articles. Keep low — retrieval already "
                         "threads internally and providers rate-limit.")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--target-precision", type=float, default=settings.TARGET_PRECISION)
    ap.add_argument("--tag", default="", help="suffix for output filenames")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log.setLevel(logging.INFO)

    loaders = {
        "golden": load_golden,
        "adversarial": load_adversarial,
        "holdout": lambda: load_holdout(args.n_per_class),
    }
    rows = loaders[args.slice]()
    if args.limit:
        rows = rows[:args.limit]

    log.info("evaluating %d articles from slice=%s", len(rows), args.slice)
    started = time.time()

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            records = list(pool.map(lambda r: run_one(r, not args.no_cache), rows))
    else:
        records = []
        for i, row in enumerate(rows, 1):
            records.append(run_one(row, not args.no_cache))
            if i % 10 == 0 or i == len(rows):
                log.info("  %d/%d (%.0fs elapsed)", i, len(rows), time.time() - started)

    report = build_report(args.slice, records, args.target_precision)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    json_path = os.path.join(RESULTS_DIR, f"eval_{args.slice}{tag}.json")
    md_path = os.path.join(RESULTS_DIR, f"eval_{args.slice}{tag}.md")
    csv_path = os.path.join(RESULTS_DIR, f"eval_per_article_{args.slice}{tag}.csv")

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(report))
    write_feature_csv(records, csv_path)

    ch = report["channels"]
    dv = report["delivered_verdicts"]
    print(f"\n{'='*66}\n  slice={args.slice}  n={report['n']}  "
          f"({time.time() - started:.0f}s)\n{'='*66}")
    print(f"  ensemble_only  acc={ch['ensemble_only']['accuracy']:.3f}  "
          f"auc={ch['ensemble_only']['roc_auc']}  ece={ch['ensemble_only']['ece']:.3f}")
    print(f"  evidence_only  acc={ch['evidence_only']['accuracy']:.3f}  "
          f"auc={ch['evidence_only']['roc_auc']}  ece={ch['evidence_only']['ece']:.3f}")
    print(f"  fused          acc={ch['fused']['accuracy']:.3f}  "
          f"auc={ch['fused']['roc_auc']}  ece={ch['fused']['ece']:.3f}")
    if dv.get("n"):
        print(f"  delivered      acc={dv['accuracy']:.3f} at coverage={dv['coverage']:.1%} "
              f"({dv['abstained']} abstained)")
    print(f"\n  wrote {os.path.relpath(md_path, BASE_DIR)}")
    print(f"        {os.path.relpath(csv_path, BASE_DIR)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
