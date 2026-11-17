# Pipeline evaluation — `golden`
Generated 2026-11-16T20:14:39.835338+00:00 · commit `3c42fe3`
**30 articles** (16 real / 14 fake)

## Channel comparison
Threshold 0.5, no abstention — the headline accuracy number.

| channel | accuracy | macro F1 | ROC-AUC | Brier | ECE | precision fake / real |
|---|---|---|---|---|---|---|
| ensemble_only | 0.633 | 0.548 | 0.857 | 0.287 | 0.331 | 1.000 / 0.593 |
| evidence_only | 0.467 | 0.318 | 0.627 | 0.248 | 0.031 | 0.000 / 0.500 |
| fused | 0.633 | 0.576 | 0.835 | 0.212 | 0.277 | 0.800 / 0.600 |

## Verdicts as delivered (with abstention)

- Committed on **36.7%** of articles (19 abstained as uncertain)
- Accuracy on committed verdicts: **0.818**
- Precision — fake: **0.000**, real: **0.900**
- Confusion (committed): tp=9 tn=0 fp=1 fn=1

## Operating point

Target precision 0.90 — **met**.

- Abstention band: `0.265 .. 0.735`
- Coverage at that band: **40.0%**
- Worst-class precision: **0.909**

## Coverage vs. precision

| band | coverage | accuracy | precision fake | precision real |
|---|---|---|---|---|
| 0.50–0.50 | 100.0% | 0.633 | 0.800 | 0.600 |
| 0.45–0.55 | 100.0% | 0.633 | 0.800 | 0.600 |
| 0.40–0.60 | 90.0% | 0.667 | 0.750 | 0.652 |
| 0.35–0.65 | 83.3% | 0.720 | 1.000 | 0.682 |
| 0.30–0.70 | 66.7% | 0.800 | 1.000 | 0.778 |
| 0.25–0.75 | 33.3% | 0.900 | 1.000 | 0.889 |
| 0.20–0.80 | 13.3% | 0.750 | 0.000 | 0.750 |
| 0.15–0.85 | 3.3% | 1.000 | 0.000 | 1.000 |

## Calibration (fused)

| predicted P(real) | n | mean confidence | observed | gap |
|---|---|---|---|---|
| 0.20-0.27 | 1 | 0.229 | 0.000 | 0.229 |
| 0.27-0.33 | 2 | 0.305 | 0.000 | 0.305 |
| 0.33-0.40 | 1 | 0.385 | 1.000 | 0.616 |
| 0.40-0.47 | 1 | 0.410 | 0.000 | 0.410 |
| 0.53-0.60 | 2 | 0.582 | 0.000 | 0.582 |
| 0.60-0.67 | 1 | 0.629 | 0.000 | 0.629 |
| 0.67-0.73 | 11 | 0.702 | 0.455 | 0.247 |
| 0.73-0.80 | 7 | 0.770 | 1.000 | 0.230 |
| 0.80-0.87 | 4 | 0.831 | 0.750 | 0.081 |

## Pipeline diagnostics

- Mean claims per article: 1.97
- Mean evidence coverage: 22.2%
- Mean evidence relevance: 0.227
- Claim verdicts: {'entailment': 1, 'contradiction': 0, 'neutral': 12, 'unverifiable': 46}
- Abstention rate: 63.3%
- Latency p50 / p95: 31.0s / 115.1s
- Degradations (articles affected): {'factcheck_disabled': 29, 'forced_abstention': 19, 'no_evidence': 19, 'location_ambiguous': 5, 'no_regional_coverage': 3, 'timeout': 1, 'region_widened': 1}

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
