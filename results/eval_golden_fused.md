# Pipeline evaluation — `golden`
Generated 2026-08-14T17:58:06.921997+00:00 · commit `01a12e3`
**30 articles** (16 real / 14 fake)

## Channel comparison
Threshold 0.5, no abstention — the headline accuracy number.

| channel | accuracy | macro F1 | ROC-AUC | Brier | ECE | precision fake / real |
|---|---|---|---|---|---|---|
| ensemble_only | 0.600 | 0.489 | 0.846 | 0.300 | 0.341 | 1.000 / 0.571 |
| evidence_only | 0.633 | 0.597 | 0.819 | 0.257 | 0.102 | 0.714 / 0.609 |
| fused | 0.667 | 0.625 | 0.915 | 0.198 | 0.235 | 0.833 / 0.625 |

## Verdicts as delivered (with abstention)

- Committed on **56.7%** of articles (13 held for human review)
- Accuracy on committed verdicts: **0.882**
- Precision — fake: **1.000**, real: **0.867**
- Confusion (committed): tp=13 tn=2 fp=2 fn=0

## Operating point

Target precision 0.90 — **met**.

- Abstention band: `0.270 .. 0.730`
- Coverage at that band: **43.3%**
- Worst-class precision: **0.917**

## Coverage vs. precision

| band | coverage | accuracy | precision fake | precision real |
|---|---|---|---|---|
| 0.50–0.50 | 100.0% | 0.667 | 0.833 | 0.625 |
| 0.45–0.55 | 96.7% | 0.655 | 0.800 | 0.625 |
| 0.40–0.60 | 86.7% | 0.692 | 1.000 | 0.652 |
| 0.35–0.65 | 80.0% | 0.708 | 1.000 | 0.682 |
| 0.30–0.70 | 63.3% | 0.842 | 1.000 | 0.833 |
| 0.25–0.75 | 40.0% | 1.000 | 1.000 | 1.000 |
| 0.20–0.80 | 23.3% | 1.000 | 0.000 | 1.000 |
| 0.15–0.85 | 6.7% | 1.000 | 0.000 | 1.000 |
| 0.10–0.90 | 3.3% | 1.000 | 0.000 | 1.000 |

## Calibration (fused)

| predicted P(real) | n | mean confidence | observed | gap |
|---|---|---|---|---|
| 0.20-0.27 | 1 | 0.226 | 0.000 | 0.226 |
| 0.33-0.40 | 2 | 0.362 | 0.000 | 0.362 |
| 0.40-0.47 | 2 | 0.432 | 0.500 | 0.068 |
| 0.47-0.53 | 1 | 0.491 | 0.000 | 0.491 |
| 0.53-0.60 | 1 | 0.587 | 0.000 | 0.587 |
| 0.60-0.67 | 2 | 0.650 | 0.000 | 0.650 |
| 0.67-0.73 | 9 | 0.705 | 0.444 | 0.261 |
| 0.73-0.80 | 5 | 0.776 | 0.800 | 0.024 |
| 0.80-0.87 | 6 | 0.829 | 1.000 | 0.172 |
| 0.87-0.93 | 1 | 0.904 | 1.000 | 0.096 |

## Pipeline diagnostics

- Mean claims per article: 1.97
- Mean evidence coverage: 47.5%
- Mean evidence relevance: 0.359
- Claim verdicts: {'entailment': 1, 'contradiction': 2, 'neutral': 24, 'unverifiable': 32}
- Abstention rate: 43.3%
- Latency p50 / p95: 1.0s / 2.1s
- Degradations (articles affected): {'forced_abstention': 11, 'no_evidence': 11}

## By source

| source | n | accuracy | macro F1 |
|---|---|---|---|
| golden/conspiracy_spam | 4 | 0.500 | 0.333 |
| golden/econ_real | 2 | 1.000 | 0.500 |
| golden/financial_scam | 1 | 0.000 | 0.000 |
| golden/health_misinfo | 3 | 0.333 | 0.250 |
| golden/health_real | 1 | 1.000 | 0.500 |
| golden/india_real | 2 | 1.000 | 0.500 |
| golden/plausible_fabrication | 6 | 0.333 | 0.250 |
| golden/science_real | 5 | 1.000 | 0.500 |
| golden/wire_real | 5 | 0.800 | 0.445 |
| golden/wire_real_dry | 1 | 1.000 | 0.500 |
