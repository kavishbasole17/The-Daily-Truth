# API Reference — Fake News Detection Backend

Reference for building clients against the FastAPI backend (`api/main.py`), including a Flutter mobile app.

## Base URL

```
http://<host>:8000
```

- Interactive Swagger UI: `http://<host>:8000/docs`
- ReDoc: `http://<host>:8000/redoc`
- CORS is wide open (`allow_origins=['*']`) — no proxy needed from any client, including a web build of Flutter.
- On a physical device/emulator, `localhost` won't reach your dev machine — use your machine's LAN IP, or `10.0.2.2` for the Android emulator.

## Error format

All errors return FastAPI's standard shape:

```json
{ "detail": "human-readable message" }
```

| Status | Meaning |
|---|---|
| 404 | Unknown model name |
| 422 | Validation error (bad input, e.g. text too short) |
| 413 | `/predict/batch` — more than 100 texts submitted |
| 503 | No models loaded yet on the server |
| 500 | Internal error (model failed to run) |

---

## `GET /`

API info. No parameters.

**Response variables**

| Field | Type | Description |
|---|---|---|
| `name` | string | `"Fake News Detection API"` |
| `version` | string | API version, e.g. `"1.0.0"` |
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

## Notes for a Flutter client

- Only call `/predict` (or `/predict/all`, `/predict/batch`) with `model_name` values where `loaded: true` in the current `/models` response — anything else returns 404/422/503 depending on why.
- `dio` or `http` + `json_serializable`/`freezed` map cleanly onto the request/response tables above.
- Poll `/health` on app start to decide whether to show a "models still loading" state before enabling prediction screens.
