"""GET /models/registry — expose model benchmark data for the analytics dashboard."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel

from paper_trail_core.registry.store import ModelEntry, get_all_models, get_top_models

router = APIRouter(prefix="/models", tags=["Models"])


class ModelRegistryEntry(BaseModel):
    name:         str
    display_name: str
    category:     str
    feature_type: str
    loaded:       bool
    path:         str
    train_time:   Optional[float]
    accuracy:     Optional[float]
    f1:           Optional[float]
    precision:    Optional[float]
    recall:       Optional[float]
    auc:          Optional[float]


class RegistryResponse(BaseModel):
    total:  int
    models: list[ModelRegistryEntry]


def _to_schema(e: ModelEntry) -> ModelRegistryEntry:
    return ModelRegistryEntry(
        name=e.name,
        display_name=e.display_name,
        category=e.category,
        feature_type=e.feature_type,
        loaded=e.loaded,
        path=e.path,
        train_time=e.train_time,
        accuracy=e.accuracy,
        f1=e.f1,
        precision=e.precision,
        recall=e.recall,
        auc=e.auc,
    )


@router.get(
    "/registry",
    response_model=RegistryResponse,
    summary="Model registry — benchmark metrics for all trained models",
)
def get_registry(
    category:     Optional[str] = Query(None, description="Filter: Classical ML | Deep Learning | Hybrid"),
    feature_type: Optional[str] = Query(None, description="Filter: tfidf | dense | sequence | hybrid"),
    sort_by:      str           = Query("f1", description="Sort metric: accuracy|f1|precision|recall|auc"),
    top_n:        Optional[int] = Query(None, description="Return only the top-N results"),
) -> RegistryResponse:
    """
    Returns benchmark data for every model in `models/registry.json`.

    Consumed by the analytics dashboard to render model-comparison charts.
    Filterable by category and feature_type; sortable by any metric.

    No auth required.
    """
    if sort_by not in {"accuracy", "f1", "precision", "recall", "auc"}:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid sort_by '{sort_by}'. Choose: accuracy|f1|precision|recall|auc",
        )

    if top_n is not None:
        entries = get_top_models(n=top_n, metric=sort_by, category=category, feature_type=feature_type)
    else:
        entries = get_all_models()
        if category:
            entries = [e for e in entries if e.category == category]
        if feature_type:
            entries = [e for e in entries if e.feature_type == feature_type]
        entries = sorted(entries, key=lambda e: getattr(e, sort_by) or 0.0, reverse=True)

    return RegistryResponse(total=len(entries), models=[_to_schema(e) for e in entries])
