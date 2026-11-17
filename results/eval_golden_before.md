# Pipeline evaluation — `golden`
Generated 2026-11-16T19:51:39.258044+00:00 · commit `3c42fe3`
**30 articles** (16 real / 14 fake)

## Channel comparison
Threshold 0.5, no abstention — the headline accuracy number.

| channel | accuracy | macro F1 | ROC-AUC | Brier | ECE | precision fake / real |
|---|---|---|---|---|---|---|
| ensemble_only | 0.633 | 0.548 | 0.857 | 0.287 | 0.331 | 1.000 / 0.593 |
| evidence_only | 0.467 | 0.318 | 0.656 | 0.261 | 0.042 | 0.000 / 0.500 |
| fused | 0.633 | 0.576 | 0.888 | 0.206 | 0.306 | 0.800 / 0.600 |

## Verdicts as delivered (with abstention)

- Committed on **33.3%** of articles (20 abstained as uncertain)
- Accuracy on committed verdicts: **0.900**
- Precision — fake: **0.000**, real: **1.000**
- Confusion (committed): tp=9 tn=0 fp=0 fn=1

## Operating point

Target precision 0.90 — **met**.

- Abstention band: `0.270 .. 0.730`
- Coverage at that band: **40.0%**
- Worst-class precision: **0.909**

## Coverage vs. precision

| band | coverage | accuracy | precision fake | precision real |
|---|---|---|---|---|
| 0.50–0.50 | 100.0% | 0.633 | 0.800 | 0.600 |
| 0.45–0.55 | 100.0% | 0.633 | 0.800 | 0.600 |
| 0.40–0.60 | 90.0% | 0.667 | 0.750 | 0.652 |
| 0.35–0.65 | 83.3% | 0.720 | 1.000 | 0.682 |
| 0.30–0.70 | 63.3% | 0.842 | 1.000 | 0.824 |
| 0.25–0.75 | 26.7% | 1.000 | 1.000 | 1.000 |
| 0.20–0.80 | 10.0% | 1.000 | 0.000 | 1.000 |

## Calibration (fused)

| predicted P(real) | n | mean confidence | observed | gap |
|---|---|---|---|---|
| 0.20-0.27 | 1 | 0.229 | 0.000 | 0.229 |
| 0.27-0.33 | 2 | 0.305 | 0.000 | 0.305 |
| 0.33-0.40 | 1 | 0.385 | 1.000 | 0.616 |
| 0.40-0.47 | 1 | 0.410 | 0.000 | 0.410 |
| 0.53-0.60 | 2 | 0.582 | 0.000 | 0.582 |
| 0.60-0.67 | 1 | 0.629 | 0.000 | 0.629 |
| 0.67-0.73 | 12 | 0.699 | 0.417 | 0.282 |
| 0.73-0.80 | 7 | 0.767 | 1.000 | 0.233 |
| 0.80-0.87 | 3 | 0.835 | 1.000 | 0.165 |

## Pipeline diagnostics

- Mean claims per article: 1.97
- Mean evidence coverage: 20.6%
- Mean evidence relevance: 0.198
- Claim verdicts: {'entailment': 1, 'contradiction': 1, 'neutral': 11, 'unverifiable': 46}
- Abstention rate: 66.7%
- Latency p50 / p95: 27.7s / 79.5s
- Degradations (articles affected): {'factcheck_disabled': 30, 'forced_abstention': 20, 'no_evidence': 20, 'location_ambiguous': 5, 'no_regional_coverage': 4, 'region_widened': 2}

## By source

| source | n | accuracy | macro F1 |
|---|---|---|---|
| golden/conspiracy_spam | 4 | 0.500 | 0.333 |
| golden/econ_real | 2 | 1.000 | 0.500 |
| golden/financial_scam | 1 | 0.000 | 0.000 |
| golden/health_misinfo | 3 | 0.333 | 0.250 |
| golden/health_real | 1 | 1.000 | 0.500 |
| golden/india_real | 2 | 0.500 | 0.333 |
| golden/plausible_fabrication | 6 | 0.167 | 0.143 |
| golden/science_real | 5 | 1.000 | 0.500 |
| golden/wire_real | 5 | 1.000 | 0.500 |
| golden/wire_real_dry | 1 | 1.000 | 0.500 |
