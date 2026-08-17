# Pipeline evaluation — `golden`
Generated 2026-08-14T17:20:27.065269+00:00 · commit `01a12e3`
**30 articles** (16 real / 14 fake)

## Channel comparison
Threshold 0.5, no abstention — the headline accuracy number.

| channel | accuracy | macro F1 | ROC-AUC | Brier | ECE | precision fake / real |
|---|---|---|---|---|---|---|
| ensemble_only | 0.600 | 0.489 | 0.846 | 0.300 | 0.341 | 1.000 / 0.571 |
| evidence_only | 0.433 | 0.428 | 0.491 | 0.268 | 0.117 | 0.421 / 0.455 |
| fused | 0.533 | 0.475 | 0.473 | 0.331 | 0.350 | 0.500 / 0.542 |

## Verdicts as delivered (with abstention)

- Committed on **36.7%** of articles (19 held for human review)
- Accuracy on committed verdicts: **0.636**
- Precision — fake: **0.500**, real: **0.714**
- Confusion (committed): tp=5 tn=2 fp=2 fn=2

## Operating point

Target precision 0.90 — **NOT met**.

- Abstention band: `0.275 .. 0.725`
- Coverage at that band: **40.0%**
- Worst-class precision: **0.556**

## Coverage vs. precision

| band | coverage | accuracy | precision fake | precision real |
|---|---|---|---|---|
| 0.50–0.50 | 100.0% | 0.533 | 0.500 | 0.542 |
| 0.45–0.55 | 73.3% | 0.545 | 0.500 | 0.556 |
| 0.40–0.60 | 56.7% | 0.471 | 0.500 | 0.462 |
| 0.35–0.65 | 56.7% | 0.471 | 0.500 | 0.462 |
| 0.30–0.70 | 46.7% | 0.571 | 0.667 | 0.545 |
| 0.25–0.75 | 36.7% | 0.545 | 0.667 | 0.500 |
| 0.20–0.80 | 26.7% | 0.375 | 0.500 | 0.333 |
| 0.15–0.85 | 23.3% | 0.286 | 0.500 | 0.200 |
| 0.10–0.90 | 16.7% | 0.200 | 0.000 | 0.200 |
| 0.05–0.95 | 16.7% | 0.200 | 0.000 | 0.200 |

## Calibration (fused)

| predicted P(real) | n | mean confidence | observed | gap |
|---|---|---|---|---|
| 0.07-0.13 | 1 | 0.116 | 0.000 | 0.116 |
| 0.13-0.20 | 1 | 0.150 | 1.000 | 0.850 |
| 0.20-0.27 | 1 | 0.209 | 0.000 | 0.209 |
| 0.27-0.33 | 1 | 0.311 | 1.000 | 0.689 |
| 0.40-0.47 | 1 | 0.467 | 0.000 | 0.467 |
| 0.47-0.53 | 3 | 0.507 | 1.000 | 0.493 |
| 0.53-0.60 | 9 | 0.562 | 0.556 | 0.007 |
| 0.67-0.73 | 4 | 0.701 | 0.250 | 0.451 |
| 0.73-0.80 | 3 | 0.749 | 1.000 | 0.251 |
| 0.80-0.87 | 1 | 0.830 | 1.000 | 0.171 |
| 0.93-1.00 | 5 | 0.980 | 0.200 | 0.780 |

## Pipeline diagnostics

- Mean claims per article: 1.97
- Mean evidence coverage: 58.6%
- Mean evidence relevance: 0.382
- Claim verdicts: {'entailment': 1, 'contradiction': 6, 'neutral': 27, 'unverifiable': 25}
- Abstention rate: 63.3%
- Latency p50 / p95: 1.1s / 4.7s
- Degradations (articles affected): {'fusion_untrained': 30, 'forced_abstention': 7, 'no_evidence': 7}

## By source

| source | n | accuracy | macro F1 |
|---|---|---|---|
| golden/conspiracy_spam | 4 | 0.250 | 0.200 |
| golden/econ_real | 2 | 1.000 | 0.500 |
| golden/financial_scam | 1 | 0.000 | 0.000 |
| golden/health_misinfo | 3 | 0.000 | 0.000 |
| golden/health_real | 1 | 1.000 | 0.500 |
| golden/india_real | 2 | 0.500 | 0.333 |
| golden/plausible_fabrication | 6 | 0.333 | 0.250 |
| golden/science_real | 5 | 0.800 | 0.445 |
| golden/wire_real | 5 | 0.800 | 0.445 |
| golden/wire_real_dry | 1 | 1.000 | 0.500 |
