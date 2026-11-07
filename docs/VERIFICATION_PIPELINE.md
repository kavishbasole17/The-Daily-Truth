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
│ 1b. locate          │  which place is each claim about, and how precisely?
└─────────┬───────────┘  → gazetteer + OpenStreetMap geocoder, with a confidence
          ▼
┌─────────────────────┐
│ 2. retrieve_evidence│  entity-anchored queries → NewsAPI + Wikipedia
└─────────┬───────────┘  + regional news for located claims; relevance-gated, cached
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
└─────────┬───────────┘                       uncertain
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
| **Principled abstention** | Thresholds chosen to hold a target precision on committed verdicts; everything else returns `uncertain`, with the reason attached. Coverage is reported alongside precision — one without the other is meaningless. |
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

## Stage 1b — Locating claims (region-aware)

`paper_trail_core/agents/locator.py`, `nlp/gazetteer.py`, `nlp/geocoder.py`.

Many claims are about somewhere in particular, and the best evidence for them is
local: a street flooding in Pune is reported by Pune's newsrooms, not by a wire
service's world desk. This stage finds the place so that retrieval can look
there.

**What it produces** — per claim, a `location` (or none):

| field | meaning |
|---|---|
| `level` | how precisely the place was resolved: `site` · `street` · `locality` · `city` · `district` · `state` · `country` |
| `name`, `label` | e.g. `Koregaon Park` / `Koregaon Park, Pune, Maharashtra, India` |
| `lat`, `lon`, `country_code`, `state`, `district`, `city`, `locality` | the resolved place and its hierarchy |
| `confidence` | 0–1; places below `LOCATION_MIN_CONFIDENCE` (0.5) are dropped |
| `source` | `claim` (named in the claim), `article` (inherited from a dateline or a place the article is clearly about), or `user` (the `region` the caller supplied) |

**How it decides.**

1. **Candidates need a reason.** A capitalised phrase is only considered a
   place if it is in the built-in gazetteer, follows a place cue ("in", "near",
   "residents of"), is the article's dateline, or is followed by a place noun
   ("Pune district"). People, organisations and acronyms are not sent to the
   geocoder.
2. **Resolution uses the whole article.** Candidates are looked up in the
   gazetteer (countries, Indian states, major cities) and in OpenStreetMap
   Nominatim, and scored on name match, prominence, agreement with the countries
   and states the *text* mentions, and the cue. That is what separates
   "Aurangabad, Bihar" from "Aurangabad, Maharashtra".
3. **It refuses to guess.** A name that could be several places and is not
   settled by context ("Springfield") is reported as not located, and the claim
   does *not* fall back to the article's place. A building-level match is only
   trusted when the name is a place type (university, hospital, dam ...), so a
   company is never pinned to its head office.
4. **Claims with no place of their own** inherit the article's place only when
   the article is demonstrably about it (a dateline, a place mentioned more than
   once, or the user's `region`), at a discounted confidence.

**What it does not do.** It does not change any verdict or fusion feature. It
only decides where evidence is searched.

**Cost and failure.** Nominatim is free but limited to one request per second,
so every lookup is cached on disk (`cache/geo`) and at most 8 places are
resolved per article within a 40 s budget. If the geocoder is unreachable the
stage falls back to the gazetteer and records a `geocoder_unavailable`
degradation; if the stage itself fails, claims are simply treated as having no
location.

### Regional retrieval

For a located claim, `evidence_retriever.regional_evidence` adds regional news
**after** the normal global search. The global results are untouched and stay
first; regional ones go through the same relevance gate and are appended, up to
`REGION_MAX_EXTRA_SNIPPETS` beyond the usual cap.

The search starts at the most specific scope and **widens** only if it finds too
little: *locality → city → state → country*, stopping once
`REGION_MIN_SNIPPETS` relevant snippets are found and trying at most
`REGION_MAX_QUERIES` scopes. The verdict reports `search_scope` (the narrowest
scope that returned evidence), `regional_evidence` (how many snippets are from
the region) and a per-snippet `in_region` flag, and a `region_widened` or
`no_regional_coverage` degradation (severity `info`) when applicable.

Regional news comes from Google News RSS country editions (no key needed) or,
optionally, newsdata.io (`REGIONAL_NEWS_PROVIDER=newsdata`, free key). Google
asks that its RSS feeds be used for personal, non-commercial purposes; set
`REGIONAL_NEWS_PROVIDER=newsdata` (or `none`) for anything beyond a coursework
demo. Set `REGION_AWARE=false` to switch the whole feature off.

---

## Claim mode — one or two sentences

Everything above was built for **articles**. A person who types a single
sentence is asking a different question, and the article machinery gets it
wrong: the style classifier and the learned fusion were fitted on whole
articles, so on one sentence the classifier is out of distribution (it says
"real" at 0.99 about plainly false sentences) and the fusion follows it. On 40
fresh single claims the article path committed to a verdict with **21%
accuracy**, worse than guessing.

So an input of at most `CLAIM_MODE_MAX_CHARS` (300) characters **and**
`CLAIM_MODE_MAX_SENTENCES` (2) sentences is handled in *claim mode*. Anything
longer takes the article path, unchanged (`CLAIM_MODE=false` turns claim mode
off). One definition of "is this a claim?" lives in `nlp/claim_mode.py`.

| | article path | claim mode |
|---|---|---|
| deciding signal | style classifier + evidence, learned fusion | **evidence only**; the classifier is reported, not used |
| supported (entailment mass ≥ `CLAIM_ENTAIL_MIN`, 0.45) | — | `likely_real` |
| refuted (contradiction mass ≥ `CLAIM_CONTRADICT_MIN`, 0.5) and none supported | — | `likely_fake` |
| inconclusive, mixed, or no evidence | `uncertain` | `uncertain` — never `likely_fake` |

What else is different in claim mode:

- **Compound claims are split.** "X is A **and** is B" is checked as "X is A"
  and "X is B" (`split_compound`), because one evidence sentence rarely states
  both halves. Ambiguous sentences are left alone.
- **Each named subject is looked up on its own** ("Chennai", "Gateway of
  India", "Delhi"), not just one blended query that favours the rarest word.
- **A Wikipedia page titled in the claim counts as on-topic** even when the
  claim gets a detail wrong, because that wrong detail is exactly what lowers
  the claim/page similarity below the usual gate. Pages with a qualifier
  ("Gateway of India *(film)*") and disambiguation pages do not qualify.
- **Only a page about the claim's own subject may refute it** (or a
  near-paraphrase of the claim). A neighbouring page — the Blue Nile for a claim
  about the Nile, Agra Fort for one about the Red Fort — holds true facts about
  something else, and an NLI model reads that as contradiction. Neighbouring
  pages can still *support*.
- **Very short claims are accepted** (≥ `CLAIM_MODE_MIN_CHARS`, 10) and short
  plain-ASCII text is no longer rejected as an "unsupported language"
  (`langdetect` labelled a 48-character English sentence Catalan at 0.99999).

### Measured

Hand-written true/false claims about landmarks, places and facts (half true,
half false by swapping one detail), Wikipedia as the main evidence. The "tuned"
sets were used while building claim mode; the last set was written beforehand
and looked at once.

| set | path | committed | accuracy when committed | strict accuracy¹ |
|---|---|---|---|---|
| 40 claims, written before tuning | article path | 70% | **21%** | 15% |
| same, after tuning | claim mode | 72% | **100%** | 72% |
| 24 regional claims (tuned on) | article path | 67% | 38% | 25% |
| same, after tuning | claim mode | 62% | 100% | 62% |
| **40 fresh claims, never tuned on** | article path | 75% | **30%** | 22% |
| same | claim mode | 92% | **97%** (1 wrong) | 90% |
| golden, 30 articles | article path = claim mode (unchanged) | 57% | 76% | 43% |

¹ correct ÷ all, counting `uncertain` as not correct.

Read these with care:

- The claims are **entity–attribute facts** ("X is in Y", "X was built in
  1889"), which is what Wikipedia answers well. They say nothing about
  fabricated events, quotes, statistics, breaking news, or satire, where
  evidence is thin or absent; there the right answer is mostly `uncertain`.
- The sets are small (24–40 claims). 97% on 37 committed verdicts is a long way
  from a population estimate, and "towards 100%" is a statement about this kind
  of claim, not about misinformation in general.
- The remaining error mode is NLI misreading a true statement as contradicted
  (e.g. a true Hampi/Vijayanagara claim at 0.85). The conservative rule — only
  subject pages may refute — removed most of it but not all.
- Region-aware search made **no difference** on these claims (identical results
  with it on or off): they are encyclopedic, and regional headlines are not what
  decides them. Its value is for local news claims, which are not measured here.

---

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

## Origin trail — where a claim came from

`paper_trail_core/agents/origin_tracer.py`, runs between verification and the
ensemble.

For every claim the evidence does not support, the trail reports who made it
and where it was seen, oldest first. It draws on two sources:

- **Published fact-checks** (Google Fact Check Tools API). A fact-check's
  `claimant` and `claimDate` ("WhatsApp forward, 27 Oct") are the best answer
  available to "where did this start". A fact-check is used only if it reviews
  the same claim (cosine ≥ `FACTCHECK_MIN_RELEVANCE`): naming the wrong
  claimant is worse than naming none.
- **The verifier's per-snippet provenance** (`evidence_meta`): each evidence
  snippet's URL, date, trust and stance (`repeats` / `debunks` / `neutral`).
  Stance ignores trust, so a low-trust site repeating the claim still counts as
  repeating it.
- **A wider search** (`ORIGIN_WIDE_SEARCH`). Evidence retrieval asks trusted
  outlets, which is right for the verdict and wrong for finding where a false
  claim started. This searches Google News RSS with no source allowlist (no
  key; SerpAPI and Tavily join when keyed), using the claim and its most
  distinctive words. A hit is kept only if it is close to the claim (cosine ≥
  0.60) **and** repeats at least 40% of its distinctive words. A different
  story about the same people ("Kim Jong-nam's death" for "Kim Jong Un dead")
  passes the first test but not the second. Hits from fact-checkers are shown
  as fact-checks; NLI reads their question headlines ("Malia Obama expelled
  from Harvard?") as repeating the claim. Other hits need an NLI stance with
  probability ≥ 0.5. Rows found this way are marked "(search)" in the trace and
  "wider search" in the UI.

  Without the NLI model, kept hits are shown as **"mentions claim"**, with no
  stance guessed: word overlap would read "Fact check: X did not…" as a repeat
  of X. Without the embedding model too, wording alone decides. And because a
  verifier without NLI cannot see contradiction, its "supported" carries no
  weight then: every claim gets a trail, and its evidence rows are also shown
  as "mentions".

  In practice Google News indexes fact-checks far more than hoax pages, so for
  a claim that spread on WhatsApp it mostly finds the debunks, with dates.
  "UNESCO declared Jana Gana Mana the best national anthem" turned up four, the
  earliest from 2018, none of which trusted-outlet retrieval had found.

It is **report-only**. No verdict, fusion feature or measured result reads it.
The UI and trace label it "earliest source found, not proof of origin", because
the trail only sees what the search providers index: messaging apps and most
social posts are not indexed, and retrieval favours trusted outlets. An undated
source is never called the earliest.

Without `GOOGLE_FACTCHECK_API_KEY` the trail is built from retrieved evidence
alone, and the trace notes that.

### Evaluating it

`scripts/evaluate_origin.py` scores the trail against ground truth already in
`dataset/`:

| slice | gold origin | headline metric |
|---|---|---|
| `claimants` | LIAR false / pants-fire claims; PolitiFact's `speaker` (half anonymous channels such as `facebook-posts`, `chain-email`; half people) | **claimant precision**: of the claimants named, how many are right |
| `publishers` | FakeNewsNet `politifact_fake.csv`; the domain of each story's `news_url` (Wayback links unwrapped) | **publisher found**: the publishing site appears in the trail |

```bash
py scripts/evaluate_origin.py --slice claimants --n 30
py scripts/evaluate_origin.py --slice publishers --n 30
```

Every rate is reported with a 95% Wilson interval, since samples this small
carry wide error bars. Both slices are US claims from 2007–2018, so they say
nothing about Indian or recent claims. `publishers` mostly measures how far
Google News RSS reaches back, because NewsAPI only covers about a month.

### Measured: `publishers`, 30 FakeNewsNet stories (2026-11-07)

Live run, no fact-check key, `results/eval_origin_publishers.{md,csv}`:

| metric | rate | 95% CI |
|---|---|---|
| publisher found anywhere in the trail | 1/30 (3%) | 1–17% |
| … without the wider search | 0/30 | 0–11% |
| earliest source is the publisher | 1/30 (3%) | 1–17% |
| wider search added at least one row | 20/30 (67%) | 49–81% |

The one hit is label noise: FakeNewsNet lists a cnn.com story as fake. In
effect the trail **did not find a single fake-news publisher**. These 2016–18
sites are mostly offline and absent from Google News, which indexes
fact-checks rather than hoax pages.

**Precision of the wider search's rows.** The first run showed most of its
"contradicts claim" rows were other stories about the same people. Every row it
produced for 60 stories was then labelled by hand, before any filter was chosen
(`dataset/golden/wider_search_rows.jsonl`): *about* the claim, and *right*
(about it, with the right stance). The filter above was chosen on stories
31–60 and checked once on stories 1–30:

| | tuning (38 rows) | held out (72 rows) |
|---|---|---|
| right, unfiltered (0.45 relevance only) | 26% | 18% (13/72) |
| right, with the filter | 80% (12/15) | **50% (9/18)** |
| about the claim, no NLI | 80% | 56% (10/18) |
| about the claim, no models at all | 71% | 46% (11/24) |

The filter also drops 8 of the 18 held-out rows that were about the claim:
three correct NASA debunks repeat under 40% of the claim's exact words. What
it lets through wrongly is the same subject in a different event (Maxine
Waters' 2010 ethics charges for a 2017 "criminally charged" hoax), which shares
the wording. One annotator labelled all rows. Rerunning `publishers` with the
filter, the wider search added rows for 12 stories instead of 20; publisher
found stays 1/30.

`claimants` has not been run: without `GOOGLE_FACTCHECK_API_KEY` no claimant
can be named, so its headline metric would be zero by construction.

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
Everything else is returned as `uncertain`.

### `golden` — 30 hand-written articles, in no training corpus

The honest read on novel input, and the scenario the system is actually for.

| channel | accuracy | ROC-AUC | ECE |
|---|---|---|---|
| ensemble only | 0.600 | 0.846 | 0.341 |
| evidence only | 0.633 | 0.819 | 0.102 |
| **fused** | **0.667** | **0.915** | 0.235 |

**Delivered verdicts (with abstention): 88.2% accurate on the 56.7% of articles
the system committed to**, precision 1.000 on `likely_fake` and 0.867 on
`likely_real`. The other 43% were returned as `uncertain`.

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
