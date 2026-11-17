# Pipeline evaluation — `golden`
Generated 2026-11-16T21:17:42.991119+00:00 · commit `3c42fe3`
**30 articles** (16 real / 14 fake)

## Channel comparison
Threshold 0.5, no abstention — the headline accuracy number.

| channel | accuracy | macro F1 | ROC-AUC | Brier | ECE | precision fake / real |
|---|---|---|---|---|---|---|
| ensemble_only | 0.633 | 0.548 | 0.857 | 0.287 | 0.331 | 1.000 / 0.593 |
| evidence_only | 0.400 | 0.286 | 0.679 | 0.241 | 0.121 | 0.000 / 0.462 |
| fused | 0.967 | 0.966 | 1.000 | 0.083 | 0.223 | 1.000 / 0.941 |

## Verdicts as delivered (with abstention)

- Committed on **43.3%** of articles (17 abstained as uncertain)
- Accuracy on committed verdicts: **1.000**
- Precision — fake: **0.000**, real: **1.000**
- Confusion (committed): tp=13 tn=0 fp=0 fn=0

## Operating point

Target precision 0.90 — **met**.

- Abstention band: `0.500 .. 0.500`
- Coverage at that band: **100.0%**
- Worst-class precision: **0.941**

## Coverage vs. precision

| band | coverage | accuracy | precision fake | precision real |
|---|---|---|---|---|
| 0.50–0.50 | 100.0% | 0.967 | 1.000 | 0.941 |
| 0.45–0.55 | 96.7% | 1.000 | 1.000 | 1.000 |
| 0.40–0.60 | 76.7% | 1.000 | 1.000 | 1.000 |
| 0.35–0.65 | 66.7% | 1.000 | 1.000 | 1.000 |
| 0.30–0.70 | 63.3% | 1.000 | 1.000 | 1.000 |
| 0.25–0.75 | 53.3% | 1.000 | 1.000 | 1.000 |
| 0.20–0.80 | 43.3% | 1.000 | 1.000 | 1.000 |
| 0.15–0.85 | 40.0% | 1.000 | 1.000 | 1.000 |
| 0.10–0.90 | 13.3% | 1.000 | 0.000 | 1.000 |
| 0.05–0.95 | 3.3% | 1.000 | 0.000 | 1.000 |

## Calibration (fused)

| predicted P(real) | n | mean confidence | observed | gap |
|---|---|---|---|---|
| 0.07-0.13 | 1 | 0.121 | 0.000 | 0.121 |
| 0.20-0.27 | 1 | 0.214 | 0.000 | 0.214 |
| 0.27-0.33 | 3 | 0.303 | 0.000 | 0.303 |
| 0.33-0.40 | 3 | 0.374 | 0.000 | 0.374 |
| 0.40-0.47 | 5 | 0.427 | 0.000 | 0.427 |
| 0.53-0.60 | 2 | 0.557 | 0.500 | 0.057 |
| 0.73-0.80 | 3 | 0.758 | 1.000 | 0.242 |
| 0.80-0.87 | 1 | 0.834 | 1.000 | 0.166 |
| 0.87-0.93 | 10 | 0.888 | 1.000 | 0.112 |
| 0.93-1.00 | 1 | 0.951 | 1.000 | 0.049 |

## Pipeline diagnostics

- Mean claims per article: 4.3
- Mean evidence coverage: 30.5%
- Mean evidence relevance: 0.304
- Claim verdicts: {'entailment': 8, 'contradiction': 0, 'neutral': 35, 'unverifiable': 86}
- Abstention rate: 56.7%
- Latency p50 / p95: 67.8s / 129.8s
- Degradations (articles affected): {'factcheck_disabled': 26, 'forced_abstention': 15, 'no_evidence': 15, 'llm_low_confidence': 10, 'location_ambiguous': 5, 'timeout': 4, 'region_widened': 3, 'llm_failed': 3, 'judges_disagree': 2, 'wider_search_failed': 2, 'llm_judge_failed': 1, 'no_regional_coverage': 1}

## By source

| source | n | accuracy | macro F1 |
|---|---|---|---|
| golden/conspiracy_spam | 4 | 1.000 | 0.500 |
| golden/econ_real | 2 | 1.000 | 0.500 |
| golden/financial_scam | 1 | 1.000 | 0.500 |
| golden/health_misinfo | 3 | 1.000 | 0.500 |
| golden/health_real | 1 | 1.000 | 0.500 |
| golden/india_real | 2 | 1.000 | 0.500 |
| golden/plausible_fabrication | 6 | 0.833 | 0.455 |
| golden/science_real | 5 | 1.000 | 0.500 |
| golden/wire_real | 5 | 1.000 | 0.500 |
| golden/wire_real_dry | 1 | 1.000 | 0.500 |
