# Pipeline evaluation — `holdout`
Generated 2026-08-14T17:56:06.367761+00:00 · commit `01a12e3`
**140 articles** (70 real / 70 fake)

## Channel comparison
Threshold 0.5, no abstention — the headline accuracy number.

| channel | accuracy | macro F1 | ROC-AUC | Brier | ECE | precision fake / real |
|---|---|---|---|---|---|---|
| ensemble_only | 0.921 | 0.921 | 0.983 | 0.058 | 0.045 | 0.915 / 0.927 |
| evidence_only | 0.421 | 0.361 | 0.464 | 0.254 | 0.033 | 0.296 / 0.451 |
| fused | 0.914 | 0.914 | 0.960 | 0.077 | 0.082 | 0.903 / 0.926 |

## Verdicts as delivered (with abstention)

- Committed on **35.0%** of articles (91 held for human review)
- Accuracy on committed verdicts: **1.000**
- Precision — fake: **1.000**, real: **1.000**
- Confusion (committed): tp=27 tn=22 fp=0 fn=0

## Operating point

Target precision 0.90 — **met**.

- Abstention band: `0.500 .. 0.500`
- Coverage at that band: **100.0%**
- Worst-class precision: **0.903**

## Coverage vs. precision

| band | coverage | accuracy | precision fake | precision real |
|---|---|---|---|---|
| 0.50–0.50 | 100.0% | 0.914 | 0.903 | 0.926 |
| 0.45–0.55 | 96.4% | 0.926 | 0.929 | 0.923 |
| 0.40–0.60 | 91.4% | 0.922 | 0.926 | 0.917 |
| 0.35–0.65 | 90.7% | 0.929 | 0.926 | 0.932 |
| 0.30–0.70 | 88.6% | 0.935 | 0.939 | 0.932 |
| 0.25–0.75 | 82.9% | 0.948 | 0.950 | 0.946 |
| 0.20–0.80 | 76.4% | 0.953 | 0.947 | 0.960 |
| 0.15–0.85 | 66.4% | 0.968 | 0.962 | 0.975 |
| 0.10–0.90 | 57.9% | 0.963 | 0.957 | 0.971 |
| 0.05–0.95 | 44.3% | 0.984 | 0.974 | 1.000 |

## Calibration (fused)

| predicted P(real) | n | mean confidence | observed | gap |
|---|---|---|---|---|
| 0.00-0.07 | 38 | 0.005 | 0.026 | 0.021 |
| 0.07-0.13 | 13 | 0.092 | 0.077 | 0.015 |
| 0.13-0.20 | 6 | 0.165 | 0.167 | 0.002 |
| 0.20-0.27 | 6 | 0.241 | 0.167 | 0.074 |
| 0.27-0.33 | 4 | 0.294 | 0.250 | 0.044 |
| 0.33-0.40 | 1 | 0.339 | 0.000 | 0.339 |
| 0.40-0.47 | 2 | 0.427 | 0.000 | 0.427 |
| 0.47-0.53 | 4 | 0.497 | 1.000 | 0.503 |
| 0.53-0.60 | 6 | 0.572 | 1.000 | 0.428 |
| 0.60-0.67 | 1 | 0.602 | 0.000 | 0.602 |
| 0.67-0.73 | 3 | 0.719 | 0.667 | 0.052 |
| 0.73-0.80 | 6 | 0.779 | 0.833 | 0.054 |
| 0.80-0.87 | 11 | 0.828 | 0.909 | 0.081 |
| 0.87-0.93 | 14 | 0.904 | 1.000 | 0.096 |
| 0.93-1.00 | 25 | 0.991 | 0.960 | 0.031 |

## Pipeline diagnostics

- Mean claims per article: 3.24
- Mean evidence coverage: 17.3%
- Mean evidence relevance: 0.232
- Claim verdicts: {'entailment': 3, 'contradiction': 8, 'neutral': 69, 'unverifiable': 374}
- Abstention rate: 65.0%
- Latency p50 / p95: 6.0s / 36.5s
- Degradations (articles affected): {'fusion_untrained': 140, 'forced_abstention': 80, 'no_evidence': 80, 'rate_limited': 65}

## By source

| source | n | accuracy | macro F1 |
|---|---|---|---|
| ISOT | 79 | 1.000 | 1.000 |
| LIAR | 5 | 0.800 | 0.762 |
| MultiFC | 56 | 0.804 | 0.803 |
