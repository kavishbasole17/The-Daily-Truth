from __future__ import annotations
from typing import Annotated, Optional
from pydantic import BaseModel, Field

MAX_TEXT_LEN = 50_000
BoundedText = Annotated[str, Field(min_length=5, max_length=MAX_TEXT_LEN)]


# ── Requests ──────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: BoundedText = Field(..., description="Raw news text to classify")
    model_name: str = Field(..., description="Model key, e.g. 'logistic_regression'")
    threshold: float = Field(0.5, ge=0.0, le=1.0,
                             description="Decision threshold for 'real' label")


class PredictAllRequest(BaseModel):
    text: BoundedText = Field(..., description="Raw news text to classify")
    threshold: float = Field(0.5, ge=0.0, le=1.0)


class PredictBatchRequest(BaseModel):
    texts: list[BoundedText] = Field(..., min_length=1, description="List of texts (max 100)")
    model_name: str = Field(..., description="Model key to use for all texts")
    threshold: float = Field(0.5, ge=0.0, le=1.0)


# ── Prediction responses ───────────────────────────────────────────────────────

class SinglePrediction(BaseModel):
    model_name: str
    display_name: str
    category: str
    prediction: int       # 0 = fake, 1 = real
    label: str            # "fake" | "real"
    confidence: float     # probability of predicted class
    prob_fake: float
    prob_real: float


class PredictResponse(BaseModel):
    text_preview: str     # first 120 chars of cleaned text
    result: SinglePrediction


class PredictAllResponse(BaseModel):
    text_preview: str
    results: list[SinglePrediction]
    consensus: str        # majority-vote label
    consensus_confidence: float


class PredictBatchResponse(BaseModel):
    model_name: str
    results: list[dict]


# ── Models list ────────────────────────────────────────────────────────────────

class ModelInfo(BaseModel):
    name: str
    display_name: str
    category: str
    feature_type: str
    loaded: bool
    metrics: Optional[dict] = None
    n_base_models: Optional[int] = None


class ModelsListResponse(BaseModel):
    total: int
    classical_count: int
    dl_count: int
    hybrid_count: int
    models: list[ModelInfo]


# ── Health ─────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    models_loaded: int
    classical_loaded: int
    dl_loaded: int
    hybrid_loaded: int


# ── Results / benchmark ────────────────────────────────────────────────────────

class BenchmarkRow(BaseModel):
    model: str
    category: str
    accuracy: Optional[float] = None
    f1: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    auc: Optional[float] = None
    train_time: Optional[float] = None


class ResultsResponse(BaseModel):
    total_models: int
    best_model: Optional[str] = None
    best_accuracy: Optional[float] = None
    rows: list[BenchmarkRow]


class ChartInfo(BaseModel):
    name: str
    filename: str
    url: str


class ChartsListResponse(BaseModel):
    charts: list[ChartInfo]
