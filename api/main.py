"""
FastAPI backend for Fake News Detection — Paper Trail.

Setup:
    1. python export_models.py          # train & save all models (run once)
    2. uvicorn api.main:app --reload    # start the API (from capstone-final/)

Existing endpoints (unchanged):
    GET  /                         API info
    GET  /health                   Health + loaded model counts
    GET  /models                   All available models with metrics
    POST /predict                  Classify with a single named model
    POST /predict/all              Classify with every model + majority-vote consensus
    POST /predict/batch            Classify a list of texts with one model (max 100)
    GET  /results                  Benchmark CSV results as JSON
    GET  /results/charts           List available chart images
    GET  /results/charts/{name}    Serve a chart PNG by name

New endpoints (agentic pipeline + persistence):
    POST /auth/token               Obtain a JWT bearer token
    POST /articles/submit          Run verification pipeline, persist result  [auth required]
    POST /articles/submit-image    OCR an image, then run verification pipeline [auth required]
    GET  /articles/{id}            Retrieve stored article + verdict
    GET  /models/registry          Benchmark data from registry.json
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from api.schemas import (
    BenchmarkRow, ChartInfo, ChartsListResponse,
    HealthResponse, ModelInfo, ModelsListResponse,
    PredictAllRequest, PredictAllResponse,
    PredictBatchRequest, PredictBatchResponse,
    PredictRequest, PredictResponse,
    ResultsResponse, SinglePrediction,
)
from api.model_loader import ModelRegistry, clean_text
from api.database import create_tables, ensure_columns

try:
    from config import settings
except ImportError:  # pragma: no cover
    settings = None
from api.routers import auth as auth_router
from api.routers import articles as articles_router
from api.routers import model_registry as reg_router

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(name)s  %(message)s',
)
logger = logging.getLogger(__name__)

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

registry = ModelRegistry()

CHART_META = {
    'comparison_all_metrics': 'All-metrics comparison (Accuracy, F1, Precision, Recall)',
    'top15_comparison':       'Top 15 models grouped bar chart',
    'auc_comparison':         'AUC-ROC scores by algorithm',
    'training_time':          'Training time by algorithm',
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialise SQLite tables for article persistence
    try:
        create_tables()
        ensure_columns()   # additive columns on an existing paper_trail.db
        logger.info('Database tables initialised.')
    except Exception as e:
        logger.error(f'Failed to create DB tables: {e}')

    # Load ML models from disk
    logger.info('Loading models from disk...')
    try:
        registry.load_all()
        logger.info(
            f'Ready: {registry.classical_count} classical + '
            f'{registry.dl_count} DL models loaded.'
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.warning(
            'API will start but /predict endpoints will fail until '
            'export_models.py is run.'
        )

    # Warm the verification pipeline's transformer singletons.
    #
    # Loading the NLI model and sentence encoder takes tens of seconds the
    # first time. Paying that during startup rather than inside the first
    # user's request keeps per-article latency in the seconds range, which is
    # what the polling UI is sized for.
    if getattr(settings, 'WARM_MODELS', True):
        try:
            from paper_trail_core.nlp import embeddings as _emb
            from paper_trail_core.nlp import nli as _nli
            logger.info('Warming verification models (NLI + embeddings)...')
            _nli.warmup()
            _emb.warmup()
            logger.info('Verification models ready.')
        except Exception as e:
            logger.warning(
                f'Could not warm verification models: {e}. The pipeline will '
                'load them on first use, or degrade and say so.'
            )
    yield


app = FastAPI(
    title='Paper Trail — Fake News Detection API',
    description=(
        'Classifies news text as **fake (0)** or **real (1)** using up to 22 '
        'trained ML/DL models, plus an agentic LangGraph verification pipeline '
        'with NLI claim checking and evidence retrieval.\n\n'
        '**Before starting:** run `python export_models.py` once to train and '
        'save all models to the `models/` directory.\n\n'
        '**Auth:** `POST /auth/token` with `admin`/`admin123` or `analyst`/`analyst123` '
        'to get a JWT for protected endpoints.'
    ),
    version='2.0.0',
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=False,
    allow_methods=['*'],
    allow_headers=['*'],
)

# ── New routers ────────────────────────────────────────────────────────────────
app.include_router(auth_router.router)
app.include_router(articles_router.router)
app.include_router(reg_router.router)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _require_models():
    if registry.total_loaded == 0:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                'No models are loaded. '
                'Run `python export_models.py` from the project root, '
                'then restart the API.'
            ),
        )


# ── Info ──────────────────────────────────────────────────────────────────────

@app.get('/', tags=['Info'])
async def root() -> dict[str, Any]:
    return {
        'name':          'Paper Trail — Fake News Detection API',
        'version':       '2.0.0',
        'models_loaded': registry.total_loaded,
        'docs':          '/docs',
        'redoc':         '/redoc',
    }


@app.get('/health', response_model=HealthResponse, tags=['Info'])
async def health() -> HealthResponse:
    return HealthResponse(
        status='ok' if registry.total_loaded > 0 else 'degraded — no models loaded',
        models_loaded=registry.total_loaded,
        classical_loaded=registry.classical_count,
        dl_loaded=registry.dl_count,
        hybrid_loaded=registry.hybrid_count,
    )


# ── Models ────────────────────────────────────────────────────────────────────

@app.get('/models', response_model=ModelsListResponse, tags=['Models'])
async def list_models() -> ModelsListResponse:
    """Returns metadata and evaluation metrics for every exported model."""
    models   = registry.list_models()
    classical = [m for m in models if m['category'] == 'Classical ML']
    dl        = [m for m in models if m['category'] == 'Deep Learning']
    hybrid    = [m for m in models if m['category'] == 'Hybrid']
    return ModelsListResponse(
        total=len(models),
        classical_count=len(classical),
        dl_count=len(dl),
        hybrid_count=len(hybrid),
        models=[ModelInfo(**m) for m in models],
    )


# ── Predict ───────────────────────────────────────────────────────────────────

@app.post('/predict', response_model=PredictResponse, tags=['Predict'])
async def predict(req: PredictRequest) -> PredictResponse:
    """Classify a single text with a specific model."""
    _require_models()
    try:
        result = registry.predict(req.text, req.model_name, req.threshold)
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail=str(e))
    except Exception as e:
        logger.exception(f'Prediction error for model {req.model_name}')
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=str(e))
    return PredictResponse(
        text_preview=clean_text(req.text)[:120],
        result=SinglePrediction(**result),
    )


@app.post('/predict/all', response_model=PredictAllResponse, tags=['Predict'])
async def predict_all(req: PredictAllRequest) -> PredictAllResponse:
    """
    Classify a text with every loaded model.
    Returns individual results plus a majority-vote consensus and mean confidence.
    """
    _require_models()
    try:
        results = registry.predict_all(req.text, req.threshold)
    except Exception as e:
        logger.exception('predict_all error')
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=str(e))

    if not results:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail='No models produced a result.',
        )

    votes            = [r['prediction'] for r in results]
    real_votes       = sum(votes)
    consensus_pred   = 1 if real_votes > len(votes) / 2 else 0
    consensus_label  = 'real' if consensus_pred == 1 else 'fake'

    winning_confs = [
        r['prob_real'] if consensus_pred == 1 else r['prob_fake']
        for r in results
    ]
    consensus_confidence = round(sum(winning_confs) / len(winning_confs), 6)

    return PredictAllResponse(
        text_preview=clean_text(req.text)[:120],
        results=[SinglePrediction(**r) for r in results],
        consensus=consensus_label,
        consensus_confidence=consensus_confidence,
    )


@app.post('/predict/batch', response_model=PredictBatchResponse, tags=['Predict'])
async def predict_batch(req: PredictBatchRequest) -> PredictBatchResponse:
    """Classify a list of texts with a single model. Maximum 100 texts per request."""
    _require_models()
    if len(req.texts) > 100:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail='Maximum 100 texts per batch request.',
        )
    try:
        results = registry.predict_batch(req.texts, req.model_name, req.threshold)
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.exception('predict_batch error')
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=str(e))

    for text, result in zip(req.texts, results):
        result['text_preview'] = clean_text(text)[:120]

    return PredictBatchResponse(model_name=req.model_name, results=results)


# ── Results / benchmark ────────────────────────────────────────────────────────

@app.get('/results', response_model=ResultsResponse, tags=['Results'])
async def get_results() -> ResultsResponse:
    """
    Returns benchmark results from the last training run as JSON.
    Reads `results/all_results.csv` generated by export_models.py.
    """
    csv_path = os.path.join(RESULTS_DIR, 'all_results.csv')
    if not os.path.exists(csv_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                'results/all_results.csv not found. '
                'Run `python export_models.py` first.'
            ),
        )

    df = pd.read_csv(csv_path)
    required = {'model', 'category'}
    missing  = required - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Malformed results CSV — missing columns: {missing}',
        )

    df = df.where(pd.notna(df), None)
    rows = [
        BenchmarkRow(
            model=row['model'],
            category=row['category'],
            accuracy=row.get('accuracy'),
            f1=row.get('f1'),
            precision=row.get('precision'),
            recall=row.get('recall'),
            auc=row.get('auc'),
            train_time=row.get('train_time'),
        )
        for _, row in df.iterrows()
    ]

    valid    = df.dropna(subset=['accuracy'])
    best_row = valid.loc[valid['accuracy'].idxmax()] if not valid.empty else None

    return ResultsResponse(
        total_models=len(rows),
        best_model=str(best_row['model']) if best_row is not None else None,
        best_accuracy=float(best_row['accuracy']) if best_row is not None else None,
        rows=rows,
    )


@app.get('/results/charts', response_model=ChartsListResponse, tags=['Results'])
async def list_charts() -> ChartsListResponse:
    """Lists benchmark chart images available to serve."""
    charts = []
    for name, _ in CHART_META.items():
        path = os.path.join(RESULTS_DIR, f'{name}.png')
        if os.path.exists(path):
            charts.append(ChartInfo(
                name=name,
                filename=f'{name}.png',
                url=f'/results/charts/{name}',
            ))
    return ChartsListResponse(charts=charts)


@app.get('/results/charts/{name}', tags=['Results'])
async def get_chart(name: str) -> FileResponse:
    """Serves a benchmark chart PNG by name (without extension)."""
    if name not in CHART_META:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Unknown chart '{name}'. "
                f"Available: {list(CHART_META.keys())}"
            ),
        )
    path = os.path.join(RESULTS_DIR, f'{name}.png')
    if not os.path.exists(path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Chart '{name}.png' not found. "
                'Run `python export_models.py` to generate charts.'
            ),
        )
    return FileResponse(path, media_type='image/png')
