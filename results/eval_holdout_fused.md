# Pipeline evaluation — `holdout`
Generated 2026-08-14T18:00:45.361541+00:00 · commit `01a12e3`
**140 articles** (70 real / 70 fake)

## Channel comparison
Threshold 0.5, no abstention — the headline accuracy number.

| channel | accuracy | macro F1 | ROC-AUC | Brier | ECE | precision fake / real |
|---|---|---|---|---|---|---|
| ensemble_only | 0.921 | 0.921 | 0.983 | 0.058 | 0.045 | 0.915 / 0.927 |
| evidence_only | 0.421 | 0.361 | 0.459 | 0.254 | 0.033 | 0.296 / 0.451 |
| fused | 0.929 | 0.929 | 0.966 | 0.081 | 0.154 | 0.929 / 0.929 |

## Verdicts as delivered (with abstention)

- Committed on **42.1%** of articles (81 held for human review)
- Accuracy on committed verdicts: **0.983**
- Precision — fake: **1.000**, real: **0.972**
- Confusion (committed): tp=35 tn=23 fp=1 fn=0

## Operating point

Target precision 0.90 — **met**.

- Abstention band: `0.500 .. 0.500`
- Coverage at that band: **100.0%**
- Worst-class precision: **0.929**

## Coverage vs. precision

| band | coverage | accuracy | precision fake | precision real |
|---|---|---|---|---|
| 0.50–0.50 | 100.0% | 0.929 | 0.929 | 0.929 |
| 0.45–0.55 | 99.3% | 0.935 | 0.929 | 0.942 |
| 0.40–0.60 | 97.9% | 0.934 | 0.929 | 0.940 |
| 0.35–0.65 | 92.1% | 0.954 | 0.941 | 0.967 |
| 0.30–0.70 | 85.0% | 0.966 | 0.938 | 1.000 |
| 0.25–0.75 | 74.3% | 0.971 | 0.951 | 1.000 |
| 0.20–0.80 | 47.9% | 0.955 | 0.919 | 1.000 |
| 0.15–0.85 | 18.6% | 1.000 | 1.000 | 1.000 |
| 0.10–0.90 | 2.9% | 1.000 | 1.000 | 1.000 |

## Calibration (fused)

| predicted P(real) | n | mean confidence | observed | gap |
|---|---|---|---|---|
| 0.00-0.07 | 1 | 0.066 | 0.000 | 0.066 |
| 0.07-0.13 | 11 | 0.122 | 0.000 | 0.122 |
| 0.13-0.20 | 25 | 0.166 | 0.120 | 0.046 |
| 0.20-0.27 | 26 | 0.226 | 0.000 | 0.226 |
| 0.27-0.33 | 3 | 0.303 | 0.333 | 0.031 |
| 0.33-0.40 | 4 | 0.370 | 0.250 | 0.120 |
| 0.47-0.53 | 1 | 0.520 | 0.000 | 0.520 |
| 0.53-0.60 | 2 | 0.577 | 1.000 | 0.423 |
| 0.60-0.67 | 6 | 0.633 | 0.667 | 0.034 |
| 0.67-0.73 | 12 | 0.694 | 0.833 | 0.140 |
| 0.73-0.80 | 19 | 0.768 | 1.000 | 0.232 |
| 0.80-0.87 | 25 | 0.822 | 1.000 | 0.178 |
| 0.87-0.93 | 3 | 0.887 | 1.000 | 0.113 |
| 0.93-1.00 | 2 | 0.937 | 1.000 | 0.063 |

## Pipeline diagnostics

- Mean claims per article: 3.24
- Mean evidence coverage: 18.0%
- Mean evidence relevance: 0.236
- Claim verdicts: {'entailment': 3, 'contradiction': 8, 'neutral': 70, 'unverifiable': 373}
- Abstention rate: 57.9%
- Latency p50 / p95: 0.8s / 1.3s
- Degradations (articles affected): {'forced_abstention': 79, 'no_evidence': 79, 'rate_limited': 63}

## By source

| source | n | accuracy | macro F1 |
|---|---|---|---|
| ISOT | 79 | 1.000 | 1.000 |
| LIAR | 5 | 0.800 | 0.762 |
| MultiFC | 56 | 0.839 | 0.839 |
