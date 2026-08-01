"""
Review-queue endpoints.

GET  /review-queue              — list articles flagged needs_human_review=true (paginated)
POST /review-queue/{id}/resolve — analyst submits a manual verdict override
"""
from __future__ import annotations

import datetime
import logging
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from api.auth import get_current_user
from api.database import get_db
from api.orm import ReviewQueue, Verdict

log = logging.getLogger(__name__)
router = APIRouter(prefix="/review-queue", tags=["Review Queue"])


# ── Schemas ────────────────────────────────────────────────────────────────────

class ReviewQueueItem(BaseModel):
    id:             int
    article_id:     int
    verdict_id:     int
    status:         str
    ai_verdict:     str
    ai_confidence:  float
    created_at:     str
    resolved_at:    Optional[str]
    human_verdict:  Optional[str]
    human_notes:    Optional[str]
    assigned_to:    Optional[str]


class PaginatedQueue(BaseModel):
    items:     list[ReviewQueueItem]
    total:     int
    page:      int
    page_size: int
    pages:     int


class ResolveRequest(BaseModel):
    human_verdict: Literal["likely_fake", "likely_real", "uncertain"]
    human_notes:   Optional[str] = None


class ResolveResponse(BaseModel):
    id:            int
    article_id:    int
    status:        str
    human_verdict: str
    resolved_at:   str


# ── Helpers ────────────────────────────────────────────────────────────────────

def _item_from_rq(rq: ReviewQueue) -> ReviewQueueItem:
    v: Verdict = rq.verdict
    return ReviewQueueItem(
        id=rq.id,
        article_id=rq.article_id,
        verdict_id=rq.verdict_id,
        status=rq.status,
        ai_verdict=v.verdict,
        ai_confidence=v.confidence,
        created_at=rq.created_at.isoformat(),
        resolved_at=rq.resolved_at.isoformat() if rq.resolved_at else None,
        human_verdict=rq.human_verdict,
        human_notes=rq.human_notes,
        assigned_to=rq.assigned_to,
    )


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get(
    "",
    response_model=PaginatedQueue,
    summary="List articles flagged for human review",
)
def list_queue(
    page:      int   = Query(1,   ge=1,  description="Page number"),
    page_size: int   = Query(20,  ge=1, le=100, description="Items per page"),
    status_filter: Optional[str] = Query(None, alias="status",
                                          description="Filter by status (pending|in_review|resolved)"),
    db: Session = Depends(get_db),
) -> PaginatedQueue:
    """
    Returns all articles that the pipeline flagged `needs_human_review=true`,
    paginated.  No auth required for listing.
    """
    q = select(ReviewQueue).options(joinedload(ReviewQueue.verdict))
    if status_filter:
        q = q.where(ReviewQueue.status == status_filter)
    q = q.order_by(ReviewQueue.created_at.desc())

    all_items = db.execute(q).scalars().all()
    total = len(all_items)
    pages = max(1, (total + page_size - 1) // page_size)
    start = (page - 1) * page_size
    page_items = all_items[start: start + page_size]

    return PaginatedQueue(
        items=[_item_from_rq(rq) for rq in page_items],
        total=total,
        page=page,
        page_size=page_size,
        pages=pages,
    )


@router.post(
    "/{queue_id}/resolve",
    response_model=ResolveResponse,
    summary="Submit a human verdict for a queued article",
)
def resolve_queue_item(
    queue_id: int,
    req: ResolveRequest,
    db: Session = Depends(get_db),
    user: dict = Depends(get_current_user),
) -> ResolveResponse:
    """
    An analyst submits a manual verdict override for a review-queue item.

    - Stores both the AI verdict (unchanged) and the human verdict.
    - Sets the queue item's status to `resolved`.

    **Requires Bearer token** (analyst or admin role).
    """
    rq = db.get(ReviewQueue, queue_id)
    if rq is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Review queue item {queue_id} not found",
        )
    if rq.status == "resolved":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Item {queue_id} is already resolved.",
        )

    rq.human_verdict = req.human_verdict
    rq.human_notes   = req.human_notes
    rq.status        = "resolved"
    rq.resolved_at   = datetime.datetime.utcnow()
    rq.assigned_to   = user.get("username")

    db.commit()
    db.refresh(rq)

    log.info(
        "resolve_queue_item: queue_id=%d resolved by %s → %s",
        queue_id, user.get("username"), req.human_verdict,
    )

    return ResolveResponse(
        id=rq.id,
        article_id=rq.article_id,
        status=rq.status,
        human_verdict=rq.human_verdict,
        resolved_at=rq.resolved_at.isoformat(),
    )
