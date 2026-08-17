# The verification pipeline

How an article becomes a verdict, why each stage works the way it does, and
what the system does when it cannot answer.

---

## The short version

```
article text
     │
     ▼
┌─────────────────────┐
│ 1. extract_claims   │  segment sentences, rank by check-worthiness
└─────────┬───────────┘  → up to 8 self-contained factual claims
          ▼
┌─────────────────────┐
│ 2. retrieve_evidence│  entity-anchored queries → NewsAPI + Wikipedia
└─────────┬───────────┘  → snippets gated by semantic relevance, cached
          ▼
┌─────────────────────┐
│ 3. verify_claims    │  DeBERTa-v3 NLI over (evidence sentence, claim) pairs
└─────────┬───────────┘  → per-claim support score in [-1, +1]
          ▼
┌─────────────────────┐
│ 4. run_ensemble     │  the 22-model stacking classifier's P(real)
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ 5. adjudicate       │  calibrated fusion → likely_real / likely_fake /
└─────────┬───────────┘                       uncertain (+ human review)
          ▼
      VerdictResult
```

Every stage runs behind a timeout with a defined fallback, and every fallback
records a **degradation** that travels with the verdict all the way to the UI.

---

## What "accurate" means here, and what it cannot mean

This system will make mistakes. It is worth saying plainly, because the
opposite claim is easy to make and always false.

Deciding whether a novel claim is true is an open research problem.
Professional fact-checkers disagree with each other on real cases. Any
fake-news detector reporting near-perfect accuracy is measuring its own
training data — which is exactly what happened in this project's earlier
numbers (see [Known limitations](#known-limitations)).

What the pipeline does instead:

| Property | How it is achieved |
|---|---|
| **Honest measurement** | Evaluated only on the partition the frozen models never trained on, plus a hand-written novel set. |
| **Calibrated confidence** | A reported 0.9 is meant to be right ~90% of the time. Measured as Expected Calibration Error, not asserted. |
| **Principled abstention** | Thresholds chosen to hold a target precision on committed verdicts; everything else returns `uncertain` and goes to the review queue. Coverage is reported alongside precision — one without the other is meaningless. |
| **No silent failure** | Every fallback, timeout, rate limit and missing model is recorded and surfaced. |
| **Determinism** | Same article → same verdict. Retrieval is cached, models are seeded and in eval mode. |

The point is not to never be wrong. The point is to be **trustworthy when it
commits, and honest when it can't**.

---

## Stage 1 — Claim extraction

`paper_trail_core/agents/claim_extractor.py`, `nlp/segmentation.py`,
`nlp/checkworthiness.py`

Sentences are segmented with a regex segmenter that masks every period which
is *not* a sentence boundary — decimals, abbreviations, dotted acronyms,
initials, ellipses — before splitting.

> This exists because the previous extractor called `text.split(".")`. In an
> article containing "raised to 6.75 percent", that produced two "claims":
> `"...raised to 6."` and `"75 percent on Thursday..."`. Every article with a
> figure in it — i.e. every article worth checking — was shredded into
> fragments, and those fragments became the search queries.

Surviving sentences are scored for **check-worthiness**, a lexical
approximation of ClaimBuster: figures, named entities, attribution verbs and
dates score up; opinions, questions and sentences whose subject is an
unresolved pronoun score down. Near-duplicates are removed (Jaccard prefilter,
then embedding cosine), and the top-scoring claims are returned **in document
order** so the UI trail reads top-to-bottom.

Backends: `heuristic` (default, local, deterministic), `llm` (optional, off,
with a hallucination guard requiring each returned claim to be grounded in an
actual article sentence), `naive` (the old splitter, kept only so a regression
test can prove the bug stays fixed).

## Stage 2 — Evidence retrieval

`paper_trail_core/agents/evidence_retriever.py`, `sources.py`, `evidence_cache.py`

- **Queries** are built from named entities plus the highest-IDF content words
  (IDF read from the already-fitted `models/classical/tfidf.joblib` — no new
  dependency, and domain-matched to news text). Whole sentences make terrible
  search queries.
- **Wikipedia** is queried with `generator=search` + `prop=extracts`, which
  returns matching pages *and* their clean plain-text intros in one request.
  The obvious search-then-fetch-each-summary approach costs four requests per
  claim and gets rate-limited immediately.
- **Relevance gating** embeds the claim and each candidate with MiniLM and
  drops anything below a cosine floor. This matters more than it looks: an NLI
  model given an irrelevant premise does not answer "I don't know", it answers
  confidently, and that confident noise outvotes real evidence.
- **Trust tiers** weight wire services above major outlets above general
  reference sources.
- **Caching** to `cache/evidence/` makes evaluation replayable offline and
  keeps NewsAPI's 100-request daily free tier from being the binding
  constraint on every rerun.

### Absence of evidence is not evidence of absence

The old retriever appended a synthetic `UNCORROBORATED` snippet when a search
came back empty, and the verifier converted that into `contradiction` at 0.6
confidence. So any claim the search providers missed counted as evidence of
fabrication.

Our retrieval is English-only, capped at ~100 news requests a day, and the free
NewsAPI tier only indexes roughly the last month. Missing a claim is the
*normal* outcome for anything older, regional, or non-English — for true and
false articles alike.

Retrieval failure is now a **status** (`no_hits`, `no_relevant`,
`rate_limited`, `error`), which lowers the coverage feature and pushes toward
abstention. It never argues that an article is fake.

## Stage 3 — NLI verification

`paper_trail_core/agents/verifier.py`, `nlp/nli.py`, `nlp/numeric.py`

Model: **`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`** — trained on FEVER,
the evidence-versus-claim fact-verification benchmark, which is precisely this
task, plus adversarial ANLI. About a quarter the size of `bart-large-mnli`.

> **Labels are read from `model.config.id2label`, never assumed.** The two
> checkpoints this project has used order their labels *inversely*
> (`bart-large-mnli` is `0=contradiction … 2=entailment`; this model is
> `0=entailment … 2=contradiction`). Hardcoding an index order means a model
> swap silently inverts every verdict while the pipeline runs cleanly.

Each evidence snippet is split into sentences and run as a separate premise —
FEVER premises are single sentences, and a multi-sentence premise dilutes the
one sentence that actually settles the question. All pairs for an article go
through in a single batched fp16 forward pass; the softmax is computed in fp32,
because half precision loses resolution in the distribution tails and the tails
are where confident contradiction lives.

### Aggregation

```
wᵢ           = scaled_relevanceᵢ × trustᵢ
entail_mass  = maxᵢ (wᵢ · P(entailment)ᵢ)
contra_mass  = maxᵢ (wᵢ · P(contradiction)ᵢ)
support      = entail_mass − effective_contradiction     ∈ [−1, +1]
```

Entailment and contradiction accumulate **separately**, and only those two are
ever compared. Neutral contributes nothing — it means absence of signal, not a
third opinion.

> The old code looped over snippets keeping whichever verdict had the highest
> confidence, comparing across label types. A `neutral` at 0.9 therefore beat
> an `entailment` at 0.7, and an article whose evidence was mostly off-topic
> came back neutral even when one snippet flatly refuted it.

`max` rather than `mean`, deliberately: one solid refutation from a wire
service settles a question, while ten vague corroborations do not. The
relevance×trust weighting is what stops a single junk snippet from exploiting
that asymmetry.

**Contradiction is held to a higher bar than entailment.** A FEVER-trained
model expects its premise to be the specific evidence sentence; hand it a
topically adjacent paragraph — a Chandrayaan-2 page for a Chandrayaan-3 claim —
and it often returns confident `contradiction` simply because details don't
line up. Measured on the golden set, that spurious-refutation mode flipped
several plainly true articles to "likely_fake"; the reverse barely occurred.

A **numeric conflict detector** runs alongside, because NLI models read two
different figures as "unrelated" rather than "opposed". Percentages are
compared in percentage points, years categorically, everything else relatively.

## Stage 4 — Stylistic ensemble

`paper_trail_core/agents/stylistic_ensemble_agent.py`

The 22-model stacking classifier, called through the public
`registry.predict(...)`. It contributes **one feature among several** rather
than driving the verdict — see [Known limitations](#known-limitations).

## Stage 5 — Adjudication

`paper_trail_core/agents/adjudicator.py`, `fusion.py`, `features.py`

A frozen 13-feature vector (`FEATURE_NAMES`) goes to a calibrated logistic
regression trained by `scripts/train_fusion.py`. The feature order is persisted
inside the model payload and asserted on load — feature drift between training
and serving is the classic way a fusion layer silently starts reading the wrong
column.

Without a trained model, a conservative hand-set fallback runs and says so. Its
evidence weight scales continuously with coverage, rather than flipping between
two constants on a binary test as the old one did.

**Two hard gates force abstention regardless of score:**

1. NLI ran degraded — a verifier that cannot detect contradiction contributes
   confidently wrong signal, which is strictly worse than none.
2. No claim could be checked at all — the only remaining signal is the style
   classifier, which on novel text is close to a coin flip.

Confidence is the calibrated probability of the class actually reported. (The
old formula capped uncertain confidence at 0.1, so the UI displayed
"10% Confidence" on every single uncertain verdict.)

---

## Running the evaluation

```bash
# Regression suite on hand-written novel articles
py scripts/evaluate_pipeline.py --slice golden

# The leak-free held-out partition
py scripts/evaluate_pipeline.py --slice holdout --n-per-class 150

# Train the fusion adjudicator on the features that produced
py scripts/train_fusion.py --input results/eval_per_article_holdout.csv

# Memorisation baseline — reported separately, never as a headline
py scripts/evaluate_pipeline.py --slice adversarial
```

Outputs land in `results/`: a markdown report, the full JSON, and the
per-article feature CSV. Every run records the git SHA and the full pipeline
config, so any number can be traced back to what produced it.

Each run reports three channels — `ensemble_only`, `evidence_only`, `fused` —
so the question "does the agentic layer earn its cost?" is answered with data
rather than assertion.

---

## Measured results

Full reports in `results/eval_golden_fused.md` and
`results/eval_holdout_fused.md` (the untagged `eval_*.md` files are the
pre-fusion baseline, kept for comparison). Two slices, reported separately
because they measure different things.

**Headline:** on articles it commits to, the system is **98.3% accurate
in-distribution** (42% coverage) and **88.2% accurate on novel hand-written
articles** (57% coverage), with **zero false "fake" verdicts on either slice**.
Everything else is routed to human review.

### `golden` — 30 hand-written articles, in no training corpus

The honest read on novel input, and the scenario the system is actually for.

| channel | accuracy | ROC-AUC | ECE |
|---|---|---|---|
| ensemble only | 0.600 | 0.846 | 0.341 |
| evidence only | 0.633 | 0.819 | 0.102 |
| **fused** | **0.667** | **0.915** | 0.235 |

**Delivered verdicts (with abstention): 88.2% accurate on the 56.7% of articles
the system committed to**, precision 1.000 on `likely_fake` and 0.867 on
`likely_real`. The other 43% were routed to human review.

The frozen ensemble alone gets 60% here while reporting ~95% confidence — that
gap between its accuracy and its stated confidence (ECE 0.341) is the failure
the rest of the pipeline exists to contain.

One case is worth singling out: a fabricated Stanford room-temperature
superconductor story that the ensemble scored **0.926 "real"** was correctly
called fake at 0.341 by the fused score. That is the evidence layer overriding
the style classifier, which is the whole design intent.

### `holdout` — 140 articles from the leak-free 20% partition

| channel | accuracy | ROC-AUC | ECE |
|---|---|---|---|
| ensemble only | 0.921 | 0.983 | 0.045 |
| evidence only | 0.421 | 0.459 | 0.033 |
| **fused** | **0.929** | 0.966 | 0.154 |

**Delivered verdicts: 98.3% accurate on the 42.1% committed**, precision 1.000
on `likely_fake` and 0.972 on `likely_real` — a single error across 58
committed verdicts (1 false "real", 0 false "fake").

Same distribution the models were trained on, so the ensemble does well here.
Broken out by source: ISOT 1.000, MultiFC 0.839, LIAR 0.800 — the ISOT figure
is the dateline artifact, not skill, and should not be quoted on its own.

**The evidence channel is retrieval-starved on this slice**: mean coverage
17.3%, and 65 of 140 articles hit search rate limits. These articles are from
2015–2017, outside the free NewsAPI window entirely. The `evidence_only` AUC of
0.459 here is a statement about our retrieval budget, not about the method —
on the golden slice, where retrieval can actually reach the subject matter, the
same channel reaches 0.819.

### What the fusion learned

Standardised coefficients (+ favours "real"), from `results/fusion_training.json`:

```
ensemble_score       +2.06     coverage             +0.53
n_claims_norm        +0.51     informative_ratio    -0.37
mean_trust           -0.37     entail_mass_max      -0.35
max_support          +0.29     numeric_conflict_max +0.22
```

Read this honestly: the ensemble dominates, and the evidence features carry
small, partly noisy weights. Given how little relevant evidence retrieval
returned on the training slice, that is the correct thing for the model to
learn — and it is exactly the kind of conclusion the old hand-set 0.15/0.85
blend asserted without checking.

## Known limitations

**The stacking ensemble does not generalise.** It classifies writing *style*,
not truth. On hand-written articles it had never seen, it scored blatant
conspiracy spam at 0.695 "real" and a fabricated Federal Reserve story at 0.961
"real" — it calls anything written in fluent prose real. It is retained as one
input feature and deliberately not trusted alone.

**The adversarial set cannot measure generalisation.** `export_models.py`
oversamples that source 15× into the training split, so ~80% of
`dataset/Adversarial/adversarial_style.csv` is training data and the rest is
template-identical to it. Scoring the hybrid there returns 300/300. The harness
reports that slice separately and labels it a memorisation baseline.

**The stacking meta-learner is trained on in-sample base predictions.**
`export_models.py:500-507` selects the meta-training rows as
`np.random.choice(len(X_train), ...)` — a random subset of `X_train`, i.e. the
same rows every base model was fitted on — directly above a comment claiming
they are "examples it has never predicted labels for". The meta-features are
therefore near-perfect in-sample predictions, which teaches the meta-learner
that its inputs are always right, and explains the saturated 0.99 outputs.

Fixing it needs only a refit of the meta-learner on out-of-fold predictions
(`cross_val_predict`); the base models themselves are fine. Left alone here
because retraining was explicitly out of scope — but it is the single
highest-value change available to this project, and the fusion layer currently
spends part of its capacity compensating for it.

**Retrieval is the binding constraint on accuracy.** English-only, ~100 NewsAPI
requests/day, roughly a one-month news window. Older, regional and non-English
topics frequently return nothing, and the honest response to that is
abstention — which is what the system does, at the cost of coverage.

### Working within the NewsAPI quota

The developer tier allows **100 requests per rolling 24 hours** (50 per 12).
Three things keep that from being the binding constraint on a demo:

1. **One NewsAPI query per claim**, not two — five claims per article means
   ~5 requests, so ~20 articles/day rather than 10.
2. **Claims dated outside the ~1-month window skip NewsAPI entirely**
   (`is_newsapi_reachable`). Querying it about a 1969 Moon landing is a
   guaranteed miss that costs quota a checkable claim needs.
3. **Everything is cached** (`cache/evidence/`), so re-analysing an article, or
   any article sharing a query, costs nothing.

When the quota does run out you get a `rate_limited` degradation, Wikipedia
keeps working, and affected claims are marked `unverifiable` — lowering
confidence and pushing toward abstention, never toward "fake". The cooldown is
timed (30 min), not a permanent latch, so a long-running server starts using
news search again once the window refills.

To remove the constraint entirely, either set `SEARCH_PROVIDER=wikipedia`
(keyless, unmetered, strong on settled facts and weak on breaking news), or
supply a `TAVILY_API_KEY` and set `SEARCH_PROVIDER=tavily` — its free tier is
roughly 1,000 searches/month, and it is designed for this kind of retrieval.

**ISOT carries a dataset artifact.** Its real rows begin with a
`"WASHINGTON (Reuters) -"` dateline that makes them trivially separable. Any
ISOT-inclusive number is reported broken out by source for this reason.

**The adversarial generator emits a grammatical tell.** Rows read
`"The European Space Agency's an orbital probe has spent years..."` — a
slot-filling bug in `scripts/generate_adversarial_dataset.py` that leaves a
possessive in front of an article. It appears in the "real"-labelled templates,
so it is a surface cue a text classifier can key on instead of learning
anything. Worth fixing before that set is used for anything load-bearing;
it is not, currently, since the set is only reported as a memorisation baseline.
