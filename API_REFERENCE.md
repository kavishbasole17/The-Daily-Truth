# Paper Trail API — Flutter handoff

Everything a Flutter client needs to talk to the Paper Trail backend
(FastAPI, `api/main.py`): how to connect, the one flow that matters, every
endpoint and field, how to render results (including the origin trail and
propagation graph), and ready-to-use Dart models. Field tables are taken from
the server's Pydantic schemas; the live, always-current version is the
Swagger UI at `/docs`.

## Start here

### 1. Run the backend

```bash
pip install -r requirements.txt
cp .env.example .env
uvicorn api.main:app --host 0.0.0.0 --port 8000   # 0.0.0.0 so phones on the LAN can reach it
```

`GET /health` returns `"status": "ok"` once models are loaded (first start takes
a minute while the NLI and embedding models load).

### 2. Pick the base URL

| Running on | Base URL |
|---|---|
| Android emulator | `http://10.0.2.2:8000` |
| iOS simulator, desktop, web | `http://localhost:8000` |
| Physical phone | `http://<your machine's LAN IP>:8000` (same Wi-Fi) |

The dev server is plain **HTTP**, which both platforms block by default:

- **Android:** add `android:usesCleartextTraffic="true"` to `<application>` in
  `android/app/src/main/AndroidManifest.xml` (debug builds only), and make sure
  the `INTERNET` permission is present.
- **iOS:** add an `NSAppTransportSecurity` → `NSAllowsLocalNetworking` (or a
  domain exception for your LAN IP) entry in `ios/Runner/Info.plist`.

CORS allows every origin, so a Flutter **web** build needs no proxy.

### 3. The main flow: verify a claim or article

```
POST /auth/token ──► token
POST /articles/submit  (Bearer token, {text})  ──► 202 {id, status: "processing"}
GET  /articles/{id}  every ~1.5 s  ──► status "processing" … "completed" | "failed"
render verdict + claim_verdicts[] (+ each claim's origin trail)
```

A one- or two-sentence claim takes about 10–30 s; a full article longer. Use
`POST /articles/submit-image` instead of `/submit` for a photo or screenshot.
Dart code for this whole flow is in [Dart client](#dart-client).

### 4. Which endpoints feed which screen

| Screen | Endpoints |
|---|---|
| Verify (the main feature) | `/auth/token`, `/articles/submit`, `/articles/submit-image`, `/articles/{id}` |
| Single-model / consensus classifier playground | `/models`, `/predict`, `/predict/all`, `/predict/batch` |
| Model leaderboard / analytics | `/results`, `/results/charts…`, `/models/registry` |
| Startup / loading state | `/health` |

## Conventions

- **JSON** in and out (except `/articles/submit-image`, which is multipart, and
  `/results/charts/{name}`, which returns a PNG).
- **Auth:** only the two `/articles/submit*` endpoints need
  `Authorization: Bearer <token>`. Everything else is open.
- **Timestamps** (`submitted_at`, `created_at`) are **UTC without an offset**,
  e.g. `"2026-11-07 10:00:05"` or `"2026-11-07T10:00:05"`. Parse with
  `DateTime.parse(s.replaceFirst(' ', 'T') + 'Z')`, not plain `DateTime.parse(s)`,
  which would read them as local time.
- **Dates inside the origin trail** are `YYYY-MM-DD` strings, or `""` when unknown.
- **New fields get added over time.** Ignore unknown keys and default missing
  ones; never fail a parse on an absent optional field.

## Errors

```json
{ "detail": "Article 42 not found" }
```

`detail` is a **string**, except for **422** validation errors, where it is a
**list** of field errors:

```json
{ "detail": [ { "type": "string_too_short", "loc": ["body", "text"],
                "msg": "String should have at least 5 characters", "input": "hi" } ] }
```

Show `detail` when it is a string; for a list, show the first item's `msg`.

| Status | Meaning |
|---|---|
| 401 | `/articles/submit*`: missing, invalid or expired token — get a new one and retry once. `/auth/token`: wrong username or password |
| 404 | Unknown model name, or unknown article id |
| 413 | `/predict/batch`: more than 100 texts. `/articles/submit-image`: image over 10 MB |
| 415 | `/articles/submit-image`: the part's content type is not `image/jpeg`, `png`, `webp`, `bmp` or `tiff` |
| 422 | Validation error (e.g. text under 5 characters) — `detail` is a list, see above |
| 503 | No models loaded yet (`/predict*`) |
| 500 | Internal error (a model failed to run) |

A verification that crashes on the server does **not** return an error: the
article's `status` becomes `"failed"` and `verdict` stays `null`.

---

## `GET /`

API info. No parameters.

**Response variables**

| Field | Type | Description |
|---|---|---|
| `name` | string | `"Paper Trail — Fake News Detection API"` |
| `version` | string | API version, currently `"2.0.0"` |
| `models_loaded` | int | Total models currently loaded (classical + DL + hybrid) |
| `docs` | string | Path to Swagger UI (`/docs`) |
| `redoc` | string | Path to ReDoc (`/redoc`) |

---

## `GET /health`

No parameters.

**Response variables (`HealthResponse`)**

| Field | Type | Description |
|---|---|---|
| `status` | string | `"ok"` or `"degraded — no models loaded"` |
| `models_loaded` | int | Total loaded models |
| `classical_loaded` | int | Loaded classical ML models |
| `dl_loaded` | int | Loaded deep learning models |
| `hybrid_loaded` | int | Loaded hybrid models (0 or 1) |

```json
{ "status": "ok", "models_loaded": 14, "classical_loaded": 14, "dl_loaded": 0, "hybrid_loaded": 0 }
```

---

## `GET /models`

Full model catalog with metrics. Use this to populate a model picker.

**Response variables (`ModelsListResponse`)**

| Field | Type | Description |
|---|---|---|
| `total` | int | Total models in the registry |
| `classical_count` | int | Count of Classical ML models |
| `dl_count` | int | Count of Deep Learning models |
| `hybrid_count` | int | Count of Hybrid models |
| `models` | list of `ModelInfo` | See below |

**`ModelInfo` fields**

| Field | Type | Description |
|---|---|---|
| `name` | string | Registry key used as `model_name` in predict calls, e.g. `"logistic_regression"` |
| `display_name` | string | Human-readable name, e.g. `"Logistic Regression"` |
| `category` | string | `"Classical ML"`, `"Deep Learning"`, or `"Hybrid"` |
| `feature_type` | string | `"tfidf"` or `"dense"` (classical only) |
| `loaded` | bool | Whether the model is actually loaded and usable |
| `metrics` | object \| null | `{ accuracy, f1, precision, recall, auc }` from training benchmark, or null |
| `n_base_models` | int \| null | Hybrid models only: how many base models it stacks |

```json
{
  "total": 22, "classical_count": 18, "dl_count": 4, "hybrid_count": 0,
  "models": [
    {
      "name": "logistic_regression",
      "display_name": "Logistic Regression",
      "category": "Classical ML",
      "feature_type": "tfidf",
      "loaded": true,
      "metrics": { "accuracy": 0.83, "f1": 0.83, "precision": 0.83, "recall": 0.83, "auc": 0.92 }
    }
  ]
}
```

---

## `POST /predict`

Classify a single text with one named model.

**Request variables (`PredictRequest`)**

| Field | Type | Required | Constraints | Description |
|---|---|---|---|---|
| `text` | string | yes | 5–50,000 chars | Raw news text to classify |
| `model_name` | string | yes | must match a `name` from `/models` | Which model to run |
| `threshold` | float | no | 0.0–1.0, default `0.5` | Decision threshold for the "real" label |

```json
{ "text": "Scientists discover new evidence of water on Mars.", "model_name": "logistic_regression", "threshold": 0.5 }
```

**Response variables (`PredictResponse`)**

| Field | Type | Description |
|---|---|---|
| `text_preview` | string | First 120 chars of the cleaned input text |
| `result` | `SinglePrediction` | See below |

**`SinglePrediction` fields**

| Field | Type | Description |
|---|---|---|
| `model_name` | string | Registry key of the model used |
| `display_name` | string | Human-readable model name |
| `category` | string | `"Classical ML"`, `"Deep Learning"`, or `"Hybrid"` |
| `prediction` | int | `0` = fake, `1` = real |
| `label` | string | `"fake"` or `"real"` |
| `confidence` | float | Probability of the predicted class |
| `prob_fake` | float | Probability of "fake" |
| `prob_real` | float | Probability of "real" |

```json
{
  "text_preview": "scientists discover new evidence of water on mars",
  "result": {
    "model_name": "logistic_regression", "display_name": "Logistic Regression",
    "category": "Classical ML", "prediction": 1, "label": "real",
    "confidence": 0.87, "prob_fake": 0.13, "prob_real": 0.87
  }
}
```

---

## `POST /predict/all`

Runs every loaded model against one text and returns a majority-vote consensus. Good for a "full analysis" screen.

**Request variables (`PredictAllRequest`)**

| Field | Type | Required | Constraints | Description |
|---|---|---|---|---|
| `text` | string | yes | 5–50,000 chars | Raw news text to classify |
| `threshold` | float | no | 0.0–1.0, default `0.5` | Decision threshold applied to every model |

```json
{ "text": "...", "threshold": 0.5 }
```

**Response variables (`PredictAllResponse`)**

| Field | Type | Description |
|---|---|---|
| `text_preview` | string | First 120 chars of the cleaned input text |
| `results` | list of `SinglePrediction` | One entry per loaded model (see fields above) |
| `consensus` | string | Majority-vote label: `"real"` or `"fake"` |
| `consensus_confidence` | float | Mean confidence of the models that voted for the winning label |

```json
{
  "text_preview": "...",
  "results": [ /* array of SinglePrediction */ ],
  "consensus": "real",
  "consensus_confidence": 0.79
}
```

---

## `POST /predict/batch`

One model, up to 100 texts. Good for a "check multiple headlines" feature.

**Request variables (`PredictBatchRequest`)**

| Field | Type | Required | Constraints | Description |
|---|---|---|---|---|
| `texts` | list of string | yes | at least 1 item, each 5–50,000 chars, max 100 items total | Texts to classify |
| `model_name` | string | yes | must match a `name` from `/models` | Which model to run on every text |
| `threshold` | float | no | 0.0–1.0, default `0.5` | Decision threshold |

```json
{ "texts": ["headline one", "headline two"], "model_name": "logistic_regression", "threshold": 0.5 }
```

**Response variables (`PredictBatchResponse`)**

| Field | Type | Description |
|---|---|---|
| `model_name` | string | Model used for all texts |
| `results` | list of dict | One entry per input text, `SinglePrediction`-shaped; failed items include an extra `"error"` field instead of prediction values |

---

## `GET /results`

Benchmark leaderboard from training. Useful for a "model performance" screen.

**Response variables (`ResultsResponse`)**

| Field | Type | Description |
|---|---|---|
| `total_models` | int | Number of rows/models in the benchmark |
| `best_model` | string \| null | Name of the highest-accuracy model |
| `best_accuracy` | float \| null | That model's accuracy |
| `rows` | list of `BenchmarkRow` | See below |

**`BenchmarkRow` fields**

| Field | Type | Description |
|---|---|---|
| `model` | string | Model name |
| `category` | string | Model category |
| `accuracy` | float \| null | Test accuracy |
| `f1` | float \| null | F1 score |
| `precision` | float \| null | Precision |
| `recall` | float \| null | Recall |
| `auc` | float \| null | AUC-ROC |
| `train_time` | float \| null | Training time (seconds) |

```json
{
  "total_models": 22, "best_model": "catboost", "best_accuracy": 0.94,
  "rows": [
    { "model": "catboost", "category": "Classical ML", "accuracy": 0.94, "f1": 0.94, "precision": 0.94, "recall": 0.94, "auc": 0.98, "train_time": 12.3 }
  ]
}
```

---

## `GET /results/charts`

Lists available benchmark chart PNGs.

**Response variables (`ChartsListResponse`)**

| Field | Type | Description |
|---|---|---|
| `charts` | list of `ChartInfo` | See below |

**`ChartInfo` fields**

| Field | Type | Description |
|---|---|---|
| `name` | string | Chart identifier, e.g. `"comparison_all_metrics"` |
| `filename` | string | PNG filename on disk |
| `url` | string | Path to fetch the image, e.g. `/results/charts/comparison_all_metrics` |

Valid chart names: `comparison_all_metrics`, `top15_comparison`, `auc_comparison`, `training_time`.

---

## `GET /results/charts/{name}`

Serves the chart PNG binary directly (`image/png`). `name` is a path parameter — one of the four valid chart names above.

---

## Verification pipeline

The evidence-grounded verifier: claims are extracted, checked against sources
with an NLI model, and fused into a verdict. Submission is **asynchronous** —
submit, get an `id` back with **202**, then poll `GET /articles/{id}`.

### `POST /auth/token`

**Request body (JSON)**

| Field | Type | Description |
|---|---|---|
| `username` | string | Demo: `admin` or `analyst` |
| `password` | string | Demo: `admin123` / `analyst123` |

**Response**

| Field | Type | Description |
|---|---|---|
| `access_token` | string | JWT; send as `Authorization: Bearer <access_token>` |
| `token_type` | string | `"bearer"` |
| `expires_in_minutes` | int | Token lifetime; on a 401 later, fetch a new one |

---

### `POST /articles/submit`

**Requires** `Authorization: Bearer <token>`. Returns **202** with an
`ArticleOut` whose `status` is `"processing"` and `verdict` is `null`.

**Request body (JSON)**

| Field | Type | Description |
|---|---|---|
| `text` | string | The claim or article, 5–50,000 chars. One or two sentences (≤ 300 chars) is judged as a *claim*, on evidence alone |
| `url` | string, optional | Source URL, stored for reference (URL-only submission is not supported yet) |
| `region` | string, optional | Place the text is about (`"Pune"`, `"Kerala"`), used when the text does not say |

```bash
curl -X POST http://localhost:8000/articles/submit \
  -H "Authorization: Bearer <token>" -H "Content-Type: application/json" \
  -d '{"text":"UNESCO has declared Jana Gana Mana the best national anthem in the world."}'
```

---

### `POST /articles/submit-image`

**Requires** a bearer token. `multipart/form-data`: `file` (JPEG, PNG, WebP,
BMP or TIFF, ≤ 10 MB), optional `url`, optional `region`. The image is OCR'd
(Hindi + English), then verified exactly like `/articles/submit`; the recovered
text appears in `text_preview` once completed. If OCR confidence is too low the
verdict is `uncertain` rather than a guess. Returns **202**.

The server checks the **part's** content type, not the filename: set it
explicitly (`MediaType('image', 'jpeg')`), or the default
`application/octet-stream` gets a 415. See [Dart client](#dart-client).

---

### `GET /articles/{id}`

No auth. Poll every 1–2 s until `status` is `completed` or `failed`
(a claim takes ~10–30 s, an article longer).

**Response: `ArticleOut`**

| Field | Type | Description |
|---|---|---|
| `id` | int | Article id |
| `url` | string \| null | As submitted |
| `text_preview` | string | First 200 chars of the text (OCR'd text for images) |
| `status` | string | `processing` \| `completed` \| `failed` |
| `stage` | string \| null | While processing: `queued`, then `verifying` |
| `submitted_at` | string | ISO timestamp |
| `verdict` | `VerdictOut` \| null | Set when `status` is `completed` |

**`VerdictOut`**

| Field | Type | Description |
|---|---|---|
| `verdict` | string | `likely_real` \| `likely_fake` \| `uncertain` |
| `confidence` | float | Calibrated probability of the reported verdict (0 when abstaining) |
| `fused_score` | float \| null | Calibrated P(real) before thresholding |
| `ensemble_score` | float | Style classifier P(real) — one input, not the verdict |
| `claim_verdicts` | `ClaimVerdictOut[]` | One per checked claim |
| `reasoning_trace` | string | Plain-text explanation, including each claim's `ORIGIN TRAIL`; render in a monospace/pre block |
| `degradations` | `{stage, kind, detail, severity}[]` | Anything that ran below full capability (`severity`: `info` \| `warn` \| `critical`) |
| `ai_model_version`, `created_at`, `id`, `article_id` | | Bookkeeping |

**`ClaimVerdictOut`** (all fields after `confidence` have defaults, so older stored rows still parse)

| Field | Type | Description |
|---|---|---|
| `claim` | string | The claim as extracted |
| `verdict` | string | `entailment` (supported) \| `contradiction` \| `neutral` \| `unverifiable` |
| `confidence` | float | Strength of that verdict |
| `evidence_snippets` | string[] | Evidence text shown to NLI |
| `sources` | string[] | Domain per snippet (same order as `evidence_snippets`) |
| `evidence_meta` | `EvidenceMeta[]` | Per snippet (same order): where it came from and what it says about the claim |
| `origin` | `Origin` \| null | The origin trail — present only for claims the evidence did **not** support (for every claim if NLI ran degraded) |
| `location`, `search_scope`, `regional_evidence`, `in_region` | | Where the claim is about, and which evidence is local |
| `event_corroborated`, `corroborating_sources`, `detail_mismatch` | | Recent independent reports of the same event (time-bound claims) |
| `support`, `entail_mass`, `contradict_mass`, `top_relevance`, `numeric_conflict`, `status` | | Verifier diagnostics |

**`EvidenceMeta`**

| Field | Type | Description |
|---|---|---|
| `url`, `domain`, `provider` | string | Where the snippet came from (`provider`: `newsapi`, `wikipedia`, `gnews_rss`, …) |
| `published_at` | string | As the provider gave it (ISO or RFC 822); `""` when unknown |
| `trust` | float | Source trust 0–1 |
| `stance` | string | `repeats` \| `debunks` \| `neutral` — what the text says, independent of trust |
| `stance_score` | float | NLI probability behind the stance |

#### The origin trail (`Origin`)

Who made a claim and where it was seen, oldest first. **Report-only**: it never
changes a verdict. It is the *earliest source found*, not proof of origin —
messaging apps and most social posts are not indexed.

| Field | Type | Description |
|---|---|---|
| `basis` | string | `fact_check` (a fact-check names the claimant) \| `earliest_indexed` (a dated source spreading it was found) \| `none` |
| `claimant` | string \| null | Who made the claim, per a published fact-check (`"WhatsApp forward"`, `"Facebook posts"`) |
| `claim_date` | string \| null | `YYYY-MM-DD`, per that fact-check |
| `claimant_source` | string \| null | The fact-checker's site |
| `earliest_seen` | `TimelineEntry` \| null | Earliest **dated** source spreading the claim (never an undated one) |
| `first_debunk` | `TimelineEntry` \| null | Earliest dated debunk or false/misleading fact-check |
| `timeline` | `TimelineEntry[]` | Up to 8 entries, dated ones oldest first, undated last |
| `fact_checks` | `FactCheck[]` | Matched fact-checks (Google Fact Check Tools) |
| `fact_check_status` | string | `ok` \| `disabled` (no `GOOGLE_FACTCHECK_API_KEY`) \| `rate_limited` \| `error` \| `skipped` (past the per-article cap) |
| `search_status` | string | Wider search: `ok` \| `no_nli` / `no_similarity` / `no_models` (ran with models missing) \| `disabled` \| `error` \| `rate_limited` \| `skipped` |

**`TimelineEntry`**

| Field | Type | Description |
|---|---|---|
| `date` | string | `YYYY-MM-DD`, or `""` when the source gave no date |
| `domain`, `url` | string | The source. Wider-search URLs are Google News redirect links |
| `kind` | string | `repeats` (spreading) \| `debunks` \| `fact_check` \| `mentions` (about the claim, stance unknown — NLI unavailable) |
| `label` | string | For `fact_check`: the rating (`"False"`), or `"reviewed"` when found by search without a rating |
| `trust` | float | Source trust 0–1 (fact-checkers use their own table) |
| `found_by` | string | `evidence` (the verifier's sources) \| `wider_search` (Google News beyond trusted outlets) \| `fact_check` |

**`FactCheck`**: `claim_text`, `claimant`, `claim_date`, `publisher`, `site`,
`url`, `rating` (publisher's wording), `rating_class` (`false` \| `misleading` \|
`true` \| `other`), `review_date`, `relevance` (how closely it matches the
claim; ≥ 0.6 to be used), `trust`.

**Example** — one claim from a completed response (trimmed):

```json
{
  "claim": "UNESCO has declared Jana Gana Mana the best national anthem in the world.",
  "verdict": "neutral",
  "confidence": 1.0,
  "sources": ["en.wikipedia.org"],
  "origin": {
    "basis": "none",
    "claimant": null,
    "earliest_seen": null,
    "first_debunk": {
      "date": "2019-08-19", "domain": "timesofindia.indiatimes.com",
      "url": "https://news.google.com/rss/articles/…", "kind": "debunks",
      "label": "", "trust": 0.9, "found_by": "wider_search"
    },
    "timeline": [
      {"date": "2016-03-14", "domain": "snopes.com",   "kind": "fact_check", "label": "reviewed", "trust": 0.85, "found_by": "wider_search", "url": "…"},
      {"date": "2018-07-03", "domain": "thequint.com", "kind": "fact_check", "label": "reviewed", "trust": 0.8,  "found_by": "wider_search", "url": "…"},
      {"date": "2019-03-19", "domain": "factly.in",    "kind": "fact_check", "label": "reviewed", "trust": 0.88, "found_by": "wider_search", "url": "…"},
      {"date": "2019-08-19", "domain": "timesofindia.indiatimes.com", "kind": "debunks", "label": "", "trust": 0.9, "found_by": "wider_search", "url": "…"}
    ],
    "fact_checks": [],
    "fact_check_status": "disabled",
    "search_status": "ok"
  }
}
```

**Drawing it.** See [Rendering guide](#rendering-guide) for the timeline list
and the propagation graph spec (lanes, markers, colours, captions).

---

### `GET /models/registry`

No auth. Benchmark metrics for every trained model, for the analytics
dashboard. Query: `category` (`Classical ML` \| `Deep Learning` \| `Hybrid`),
`feature_type` (`tfidf` \| `dense` \| `sequence` \| `hybrid`), `sort_by`
(`accuracy` \| `f1` \| `precision` \| `recall` \| `auc`, default `f1`), `top_n`.

---

## Rendering guide

Match the web app (`frontend/src/app/verify/page.jsx`) so both clients say the
same thing.

### Verdict and claim labels

| `verdict.verdict` | Show |
|---|---|
| `likely_real` | **LIKELY REAL** · `N% confidence` |
| `likely_fake` | **LIKELY MISINFORMATION** · `N% confidence` |
| `uncertain` | **UNCERTAIN** · `Not enough evidence to decide` when `confidence` is 0, else `N% confidence` |

| `claim_verdicts[].verdict` | Show | Notes |
|---|---|---|
| `entailment` | SUPPORTED | |
| `contradiction` | CONTRADICTED | |
| `neutral` | NO CLEAR SIGNAL | Evidence found, but inconclusive |
| `unverifiable` | COULD NOT VERIFY | Nothing found to check against. Add: "That is not evidence against it — recent, regional and non-English topics are often missed." |

- **`degradations`**: if non-empty, list each `detail` under "Limitations
  affecting this verdict".
- **`reasoning_trace`**: monospace, whitespace preserved, horizontally
  scrollable, with a copy button. It already contains every claim's origin trail
  as text.

### Origin trail (per claim, when `origin != null`)

1. Heading **Origin trail** with the caveat *"earliest source found, not proof
   of origin"*. Keep the caveat: the trail only sees what search engines index.
2. If `origin.claimant` is set: *"Claimed by **{claimant}** on {claim_date} ·
   per {claimant_source} fact-check"*.
3. The propagation graph (below), when it has at least two dated points.
4. The `timeline` as a list, one row per entry:

| Part | Shows |
|---|---|
| Date | `date`, or `undated` |
| Source | `domain`; tap opens `url` in the browser (`url_launcher`). Wider-search URLs are Google News redirects and open fine in a browser |
| Stance chip | `repeats` → "repeats claim" (red tint) · `debunks` → "contradicts claim" (green tint) · `fact_check` → "fact-check: {label}" (green tint; neutral if the label says the claim is true) · `mentions` → "mentions claim" (neutral) |
| Extra chips | "wider search" when `found_by == "wider_search"`; "earliest found" when the entry's `url` equals `origin.earliest_seen.url` |
| Trust | `trust {trust.toStringAsFixed(2)}` |

If `timeline` is empty: *"No source found repeating or checking this claim."*

### Propagation graph

How the claim spread over time. The data has dates, not copy relationships, so
the graph must never imply who copied whom.

- **Points**: every `timeline` entry with a non-empty `date`, plus an origin
  point at `claim_date` when `claimant` is set. Draw nothing if there are fewer
  than two points. Undated entries stay in the list only.
- **x** = date. **Lanes** (rows, top to bottom, only those that have points):
  Origin (the claimant) · Spreading (`repeats`) · Checking (`debunks`,
  `fact_check`) · Mentions (`mentions`). Label each lane on the left.
- **Markers** — shape and lane carry the meaning, so it never rests on colour
  alone:

| Lane | Marker | Light theme | Dark theme |
|---|---|---|---|
| Origin | hollow ring, 14 px | text colour | text colour |
| Spreading | filled circle, 12 px | `#C1121F` | `#E10600` |
| Checking | filled diamond, 10 px | `#16A34A` | `#2AAE5C` |
| Mentions | hollow circle, 12 px | grey | grey |

  Give each marker a 2 px ring in the card background colour, and fan out
  same-day markers in one lane vertically by about 11 px so they don't overlap.
  This red/green pair was checked for colour-blind readers in both themes;
  don't swap in other shades without rechecking.
- **Links**: from the *anchor* (the origin point, else the earliest Spreading
  point) to every later point, as thin (~1.25 px) curves at ~45% opacity in the
  target point's lane colour. No anchor → no links.
- **Axis**: a hairline with 3–5 date ticks (years for multi-year trails, months
  or days for short ones).
- **Tap** a marker to show date, source, stance, trust and "wider search" if it
  applies.
- **Caption** (keep the wording): *"Lines join each source to the first known
  appearance. They show order in time, not who copied whom."* — or, with no
  anchor, *"No dated source spreading the claim was found, so there is no
  starting point to draw from."* Append *"N undated sources are listed below
  but not plotted."* when there are any.

A `CustomPainter` is the simplest way to draw this. The web version is
`frontend/src/components/PropagationGraph.jsx` if you want to mirror its layout
maths.

---

## Dart models

Hand-written so they tolerate missing and unknown fields (older stored verdicts
lack the newer ones). Swap in `json_serializable`/`freezed` if you prefer — keep
the defaults.

```dart
// lib/api/models.dart

double _d(Object? v, [double fallback = 0]) => (v as num?)?.toDouble() ?? fallback;
String _s(Object? v) => v as String? ?? '';
List<T> _list<T>(Object? v, T Function(Map<String, dynamic>) f) =>
    ((v as List?) ?? const []).map((e) => f(e as Map<String, dynamic>)).toList();
Map<String, dynamic>? _obj(Object? v) => v as Map<String, dynamic>?;

/// Server timestamps are UTC without an offset; plain DateTime.parse would read them as local.
DateTime parseServerTime(String s) {
  final iso = s.replaceFirst(' ', 'T');
  final hasZone = RegExp(r'(Z|[+-]\d\d:?\d\d)$').hasMatch(iso);
  return DateTime.parse(hasZone ? iso : '${iso}Z');
}

class ArticleOut {
  ArticleOut({required this.id, this.url, required this.textPreview, required this.status,
      this.stage, required this.submittedAt, this.verdict});
  final int id;
  final String? url;
  final String textPreview;
  final String status;          // processing | completed | failed
  final String? stage;          // queued | verifying | null
  final DateTime submittedAt;   // UTC
  final VerdictOut? verdict;    // set when status == completed

  bool get isDone => status == 'completed' || status == 'failed';

  factory ArticleOut.fromJson(Map<String, dynamic> j) => ArticleOut(
        id: j['id'] as int,
        url: j['url'] as String?,
        textPreview: _s(j['text_preview']),
        status: j['status'] as String? ?? 'processing',
        stage: j['stage'] as String?,
        submittedAt: parseServerTime(j['submitted_at'] as String),
        verdict: _obj(j['verdict']) == null ? null : VerdictOut.fromJson(_obj(j['verdict'])!),
      );
}

class VerdictOut {
  VerdictOut({required this.verdict, required this.confidence, this.fusedScore,
      required this.ensembleScore, required this.claimVerdicts, required this.reasoningTrace,
      required this.degradations});
  final String verdict;         // likely_real | likely_fake | uncertain
  final double confidence;
  final double? fusedScore;
  final double ensembleScore;
  final List<ClaimVerdict> claimVerdicts;
  final String reasoningTrace;
  final List<Degradation> degradations;

  factory VerdictOut.fromJson(Map<String, dynamic> j) => VerdictOut(
        verdict: _s(j['verdict']),
        confidence: _d(j['confidence']),
        fusedScore: (j['fused_score'] as num?)?.toDouble(),
        ensembleScore: _d(j['ensemble_score'], 0.5),
        claimVerdicts: _list(j['claim_verdicts'], ClaimVerdict.fromJson),
        reasoningTrace: _s(j['reasoning_trace']),
        degradations: _list(j['degradations'], Degradation.fromJson),
      );
}

class Degradation {
  Degradation(this.stage, this.kind, this.detail, this.severity);
  final String stage, kind, detail, severity; // severity: info | warn | critical
  factory Degradation.fromJson(Map<String, dynamic> j) =>
      Degradation(_s(j['stage']), _s(j['kind']), _s(j['detail']), _s(j['severity']));
}

class ClaimVerdict {
  ClaimVerdict({required this.claim, required this.verdict, required this.confidence,
      required this.evidenceSnippets, required this.sources, required this.evidenceMeta,
      this.origin, this.location});
  final String claim;
  final String verdict;         // entailment | contradiction | neutral | unverifiable
  final double confidence;
  final List<String> evidenceSnippets;
  final List<String> sources;   // parallel to evidenceSnippets
  final List<EvidenceMeta> evidenceMeta; // parallel to evidenceSnippets
  final Origin? origin;         // only for claims the evidence did not support
  final Map<String, dynamic>? location;

  factory ClaimVerdict.fromJson(Map<String, dynamic> j) => ClaimVerdict(
        claim: _s(j['claim']),
        verdict: _s(j['verdict']),
        confidence: _d(j['confidence']),
        evidenceSnippets: ((j['evidence_snippets'] as List?) ?? const []).cast<String>(),
        sources: ((j['sources'] as List?) ?? const []).cast<String>(),
        evidenceMeta: _list(j['evidence_meta'], EvidenceMeta.fromJson),
        origin: _obj(j['origin']) == null ? null : Origin.fromJson(_obj(j['origin'])!),
        location: _obj(j['location']),
      );
}

class EvidenceMeta {
  EvidenceMeta(this.url, this.domain, this.publishedAt, this.provider, this.trust,
      this.stance, this.stanceScore);
  final String url, domain, publishedAt, provider;
  final double trust;
  final String stance;          // repeats | debunks | neutral
  final double stanceScore;
  factory EvidenceMeta.fromJson(Map<String, dynamic> j) => EvidenceMeta(_s(j['url']),
      _s(j['domain']), _s(j['published_at']), _s(j['provider']), _d(j['trust'], 0.5),
      j['stance'] as String? ?? 'neutral', _d(j['stance_score']));
}

class Origin {
  Origin({required this.basis, this.claimant, this.claimDate, this.claimantSource,
      this.earliestSeen, this.firstDebunk, required this.timeline, required this.factChecks,
      required this.factCheckStatus, required this.searchStatus});
  final String basis;           // fact_check | earliest_indexed | none
  final String? claimant;
  final String? claimDate;      // YYYY-MM-DD
  final String? claimantSource;
  final TimelineEntry? earliestSeen;
  final TimelineEntry? firstDebunk;
  final List<TimelineEntry> timeline;
  final List<FactCheck> factChecks;
  final String factCheckStatus;
  final String searchStatus;

  factory Origin.fromJson(Map<String, dynamic> j) => Origin(
        basis: j['basis'] as String? ?? 'none',
        claimant: j['claimant'] as String?,
        claimDate: j['claim_date'] as String?,
        claimantSource: j['claimant_source'] as String?,
        earliestSeen: _obj(j['earliest_seen']) == null
            ? null : TimelineEntry.fromJson(_obj(j['earliest_seen'])!),
        firstDebunk: _obj(j['first_debunk']) == null
            ? null : TimelineEntry.fromJson(_obj(j['first_debunk'])!),
        timeline: _list(j['timeline'], TimelineEntry.fromJson),
        factChecks: _list(j['fact_checks'], FactCheck.fromJson),
        factCheckStatus: j['fact_check_status'] as String? ?? 'skipped',
        searchStatus: j['search_status'] as String? ?? 'skipped',
      );
}

class TimelineEntry {
  TimelineEntry(this.date, this.domain, this.url, this.kind, this.label, this.trust, this.foundBy);
  final String date;            // YYYY-MM-DD or '' (undated)
  final String domain, url;
  final String kind;            // repeats | debunks | fact_check | mentions
  final String label;           // fact-check rating, or 'reviewed'
  final double trust;
  final String foundBy;         // evidence | wider_search | fact_check

  DateTime? get day => date.isEmpty ? null : DateTime.parse('${date}T00:00:00Z');

  factory TimelineEntry.fromJson(Map<String, dynamic> j) => TimelineEntry(_s(j['date']),
      _s(j['domain']), _s(j['url']), _s(j['kind']), _s(j['label']), _d(j['trust'], 0.5),
      j['found_by'] as String? ?? 'evidence');
}

class FactCheck {
  FactCheck(this.claimText, this.claimant, this.claimDate, this.publisher, this.site, this.url,
      this.rating, this.ratingClass, this.reviewDate, this.relevance, this.trust);
  final String claimText, claimant, claimDate, publisher, site, url, rating;
  final String ratingClass;     // false | misleading | true | other
  final String reviewDate;
  final double relevance, trust;
  factory FactCheck.fromJson(Map<String, dynamic> j) => FactCheck(_s(j['claim_text']),
      _s(j['claimant']), _s(j['claim_date']), _s(j['publisher']), _s(j['site']), _s(j['url']),
      _s(j['rating']), j['rating_class'] as String? ?? 'other', _s(j['review_date']),
      _d(j['relevance']), _d(j['trust'], 0.5));
}
```

## Dart client

Uses `package:http` (tested with 1.6.0) and `package:http_parser`. Checked
with `dart analyze` and run against a live server: submit, poll, full origin
parsing, and the 401 / 404 / 422 error paths.

- **Set the image part's content type.** `MultipartFile.fromPath` sends
  `application/octet-stream` by default, which the server rejects with 415.
- **UTF-8.** The server sends `application/json` with no charset. `http` 1.x
  decodes that as UTF-8, so `response.body` is fine; the client decodes
  `bodyBytes` as UTF-8 explicitly anyway, so Hindi and "—" survive if you swap
  in a client that falls back to Latin-1.

```dart
// lib/api/paper_trail_api.dart
import 'dart:async';
import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
import 'models.dart';

class ApiException implements Exception {
  ApiException(this.status, this.message);
  final int status;
  final String message;
  @override
  String toString() => 'ApiException($status): $message';
}

class PaperTrailApi {
  PaperTrailApi(this.baseUrl, {this.username = 'admin', this.password = 'admin123'});
  final String baseUrl;
  final String username, password;
  final http.Client _http = http.Client();
  String? _token;

  Uri _u(String path) => Uri.parse('$baseUrl$path');

  dynamic _json(http.Response r) => jsonDecode(utf8.decode(r.bodyBytes));

  void _check(http.Response r) {
    if (r.statusCode >= 200 && r.statusCode < 300) return;
    var message = 'HTTP ${r.statusCode}';
    try {
      final detail = (_json(r) as Map<String, dynamic>)['detail'];
      if (detail is String) {
        message = detail;
      } else if (detail is List && detail.isNotEmpty) {
        message = (detail.first as Map<String, dynamic>)['msg']?.toString() ?? message;
      }
    } catch (_) {}
    throw ApiException(r.statusCode, message);
  }

  Future<String> _getToken({bool refresh = false}) async {
    if (_token != null && !refresh) return _token!;
    final r = await _http.post(_u('/auth/token'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'username': username, 'password': password}));
    _check(r);
    return _token = (_json(r) as Map<String, dynamic>)['access_token'] as String;
  }

  /// An authenticated call; on 401 it fetches a fresh token and retries once.
  Future<http.Response> _authed(Future<http.Response> Function(String token) send) async {
    var r = await send(await _getToken());
    if (r.statusCode == 401) r = await send(await _getToken(refresh: true));
    _check(r);
    return r;
  }

  /// Queues a claim or article; returns the article id to poll.
  Future<int> submitText(String text, {String? region}) async {
    final r = await _authed((t) => _http.post(_u('/articles/submit'),
        headers: {'Content-Type': 'application/json', 'Authorization': 'Bearer $t'},
        body: jsonEncode({'text': text, if (region != null) 'region': region})));
    return (_json(r) as Map<String, dynamic>)['id'] as int;
  }

  /// Queues a photo/screenshot. [subtype] is 'jpeg', 'png', 'webp', 'bmp' or 'tiff'.
  Future<int> submitImage(String path, {required String subtype, String? region}) async {
    final r = await _authed((t) async {
      final req = http.MultipartRequest('POST', _u('/articles/submit-image'))
        ..headers['Authorization'] = 'Bearer $t'
        ..files.add(await http.MultipartFile.fromPath('file', path,
            contentType: MediaType('image', subtype)));
      if (region != null) req.fields['region'] = region;
      return http.Response.fromStream(await _http.send(req));
    });
    return (_json(r) as Map<String, dynamic>)['id'] as int;
  }

  Future<ArticleOut> getArticle(int id) async {
    final r = await _http.get(_u('/articles/$id'));
    _check(r);
    return ArticleOut.fromJson(_json(r) as Map<String, dynamic>);
  }

  /// Polls until the verification completes or fails.
  Future<ArticleOut> waitForVerdict(int id,
      {Duration every = const Duration(milliseconds: 1500),
      Duration timeout = const Duration(minutes: 3),
      void Function(String? stage)? onStage}) async {
    final deadline = DateTime.now().add(timeout);
    while (true) {
      final article = await getArticle(id);
      if (article.isDone) return article;
      onStage?.call(article.stage);
      if (DateTime.now().isAfter(deadline)) {
        throw TimeoutException('Verification $id is still running', timeout);
      }
      await Future<void>.delayed(every);
    }
  }
}
```

Usage:

```dart
final api = PaperTrailApi('http://10.0.2.2:8000');
final id = await api.submitText('UNESCO has declared Jana Gana Mana the best national anthem in the world.');
final article = await api.waitForVerdict(id, onStage: (s) => print('stage: $s'));
if (article.status == 'failed' || article.verdict == null) {
  // show "Verification failed — try again"
} else {
  final v = article.verdict!;
  for (final c in v.claimVerdicts) {
    print('${c.verdict}: ${c.claim}  origin rows: ${c.origin?.timeline.length ?? 0}');
  }
}
```

## Edge-case checklist

- `status == "failed"` → `verdict` is `null`. Show a retry, not a crash.
- `verdict == "uncertain"` with `confidence == 0` is a normal, honest result (an
  abstention), not an error. `degradations` says why.
- `claim_verdicts` can be empty (nothing checkable was found).
- `origin` is `null` for supported claims and for verdicts stored before the
  origin trail existed.
- `timeline` entries can be undated (`date == ""`): list them, don't plot them.
- `claimant` is `null` until the server has `GOOGLE_FACTCHECK_API_KEY`; the
  trail still shows dated sources without it.
- Text can contain non-ASCII (Hindi, "—", "₹"): make sure responses are decoded as UTF-8.
- Only call `/predict*` with `model_name` values whose `loaded` is `true` in
  `/models`; check `/health` on app start before enabling those screens.
- Tokens expire after 60 minutes; the client above refreshes on 401.
