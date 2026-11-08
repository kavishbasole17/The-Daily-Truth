# Accuracy: how to verify it, and the hard limits

## Verify it's running at full accuracy

```
GET /health/accuracy?deep=true
```

Every stage reports what is **actually** running (not just what is configured), whether that is
full accuracy, and the exact fix if not. The same report is logged at startup. Each verdict also
carries `degradations` listing anything that ran below full strength for that specific check.

Full accuracy needs all of:

| Requirement | Why |
|---|---|
| `ANTHROPIC_API_KEY` in `.env` | LLM claim extraction (`CLAIM_BACKEND=llm`), translation of non-English articles, chat replies with new checks |
| `torch`, `transformers`, `sentence-transformers` | NLI model (the core judgement) and evidence relevance gating |
| `tensorflow` | lstm / gru / cnn / cnn_lstm are inputs to the stacking ensemble |
| `scikit-learn==1.7.2` (pinned) | the version `models/` was saved with; newer versions fail to load `gradient_boosting` and warn of invalid results |
| Tesseract binary + Hindi pack | image (screenshot) checks; without it they abstain |
| `NEWS_API_KEY` (+ optionally `TAVILY_API_KEY` / `SERPAPI_KEY`) | live news evidence; a web-search key widens coverage beyond news + Wikipedia |

## Measured on the real pipeline (CPU, live evidence, no LLM key)

| Input | Result | Time |
|---|---|---|
| "The Eiffel Tower is located in Paris, France." | likely_real, 60% (NLI: supported) | 15 s |
| "The Eiffel Tower is located in Berlin, Germany." | **uncertain**: NLI found a contradiction (0.50) but the adjudicator abstained | 12 s |
| "Fake!" | rejected: 5 characters | instant |

The Berlin case shows the main design trade-off: verdicts are calibrated for **90% precision**,
so when evidence is only moderately strong the system says "uncertain" instead of risking a wrong
"fake". Lowering `CLAIM_CONTRADICT_MIN` (0.5) would commit more often but must be re-validated
with `scripts/evaluate_pipeline.py`, not just changed.

## Hard limits

### Input
| Limit | Value | What happens |
|---|---|---|
| Minimum text | **10 characters** | Rejected as too short (`uncertain`, critical degradation) |
| Short claim | 10–39 characters | Checked exactly as typed (no claim extraction) |
| Claim mode | ≤ 300 chars and ≤ 2 sentences | Decided on evidence alone; the style ensemble is skipped (it was trained on whole articles) |
| Maximum text | **50,000 characters** | API rejects with 422 |
| Claims per article | **5** (`MAX_CLAIMS`) | Only the 5 most check-worthy claims are verified |
| Sentence length (heuristic extractor) | 25–600 chars, 6–60 tokens | Shorter/longer sentences aren't considered claims |
| Translation | 50,000 chars, needs the API key | Without the key, non-English text is checked untranslated, usually `uncertain` |
| Images | 10 MB; JPEG/PNG/WEBP/BMP/TIFF | Rejected otherwise |
| OCR confidence | ≥ 0.55 | Below it, the check abstains rather than verify a bad transcription |
| Links | page ≤ 3 MB, ≤ 4 redirects, 15 s, ≥ 200 chars of article text | Paywalled, login-only or JavaScript-rendered pages fail with a reason; private/local addresses are refused |

### Evidence
| Limit | Value | Effect |
|---|---|---|
| NewsAPI free tier | 100 requests/day; articles from the last 30 days, with a ~24 h delay | Breaking news (< 1 day) and older stories rely on Wikipedia / regional RSS |
| Snippets per claim | 4 (+2 regional) | Caps evidence depth per claim |
| Evidence calls per article | 12 | Caps total searches |
| Relevance gate | cosine ≥ 0.45 | Off-topic snippets are dropped (prevents false contradictions) |
| Evidence cache | 7 days | Re-checking a claim within a week reuses cached evidence, so fast-moving stories can be stale. Set `EVIDENCE_CACHE_TTL_DAYS=1` for fresher news (costs NewsAPI quota) |
| HTTP timeout / stage timeout | 8 s per request / 90 s per stage | Slow sources count as "no evidence" for that stage |
| Wikipedia | ~1 request/second, 3 workers | Throughput ceiling on long articles |

### Judgement
| Limit | Value | Effect |
|---|---|---|
| NLI input | 256 tokens per (evidence, claim) pair; evidence capped at 1,500 chars | Long evidence passages are truncated. The model supports 512 (`NLI_MAX_LENGTH`), at ~2× CPU cost, but fusion was calibrated at 256, so re-evaluate before changing it |
| NLI pairs per article | 200 | Highest-relevance pairs kept |
| Commit thresholds | supported ≥ 0.45, contradicted ≥ 0.50 (claim mode); 90% target precision (articles) | Anything weaker is `uncertain` |
| "Could not verify" | no relevant evidence found | Not evidence of falsehood; recent, regional and non-English topics are often missed |
| Speed (CPU) | ~12–15 s per short claim | A GPU is ~10× faster with identical accuracy |

### Chat
| Limit | Value |
|---|---|
| New checks per reply | 2 |
| Replies in progress per chat | 1 |
| History sent to the model | last 8 exchanges + every check in the chat |
| Signed-out use | 1 free check per IP |
