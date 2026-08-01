"""
Article submission and retrieval endpoints.

POST /articles/submit  — run the verification pipeline, persist the result.
GET  /articles/{id}    — retrieve a stored article + verdict.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy.orm import Session

from api.auth import get_current_user
from api.database import get_db
from api.orm import Article, ReviewQueue, Verdict
from paper_trail_core.agents.pipeline import run_verification_pipeline

log = logging.getLogger(__name__)
router = APIRouter(prefix="/articles", tags=["Articles"])


# ── Request / response schemas (local — avoids circular imports with api/schemas.py) ──

# Matches api/schemas.py's BoundedText cap — kept as a local constant (rather
# than importing api/schemas.py) to avoid the circular import called out
# above. An unbounded text field here would let a caller submit arbitrarily
# large payloads that get persisted to the DB and run through TF-IDF
# vectorization + claim extraction with no size limit.
MAX_ARTICLE_TEXT_LEN = 50_000


class ArticleSubmitRequest(BaseModel):
    text: Optional[str] = Field(
        None, min_length=5, max_length=MAX_ARTICLE_TEXT_LEN,
        description="Raw article text",
    )
    url:  Optional[str] = Field(None, max_length=2048, description="Source URL (stored for reference)")

    @model_validator(mode="after")
    def _require_text_or_url(self):
        if not self.text and not self.url:
            raise ValueError("Provide either 'text' or 'url'.")
        if not self.text and self.url:
            # Future: fetch text from URL — for now require text
            raise ValueError("URL-only submission not yet supported. Please include 'text'.")
        return self


class ClaimVerdictOut(BaseModel):
    claim:             str
    verdict:           str
    confidence:        float
    evidence_snippets: list[str] = []


class VerdictOut(BaseModel):
    id:                 int
    article_id:         int
    verdict:            str
    confidence:         float
    claim_verdicts:     list[ClaimVerdictOut]
    ensemble_score:     float
    reasoning_trace:    str
    needs_human_review: bool
    ai_model_version:   Optional[str]
    created_at:         str

    model_config = ConfigDict(from_attributes=True)


class ArticleOut(BaseModel):
    id:           int
    url:          Optional[str]
    text_preview: str           # first 200 chars
    status:       str
    submitted_at: str
    verdict:      Optional[VerdictOut] = None

    model_config = ConfigDict(from_attributes=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_article_out(article: Article) -> ArticleOut:
    verdict_out = None
    if article.verdicts:
        v = article.verdicts[-1]
        cvs_raw = json.loads(v.claim_verdicts) if v.claim_verdicts else []
        verdict_out = VerdictOut(
            id=v.id,
            article_id=v.article_id,
            verdict=v.verdict,
            confidence=v.confidence,
            claim_verdicts=[ClaimVerdictOut(**cv) for cv in cvs_raw],
            ensemble_score=v.ensemble_score,
            reasoning_trace=v.reasoning_trace,
            needs_human_review=v.needs_human_review,
            ai_model_version=v.ai_model_version,
            created_at=v.created_at.isoformat(),
        )
    return ArticleOut(
        id=article.id,
        url=article.url,
        text_preview=article.text[:200],
        status=article.status,
        submitted_at=article.submitted_at.isoformat(),
        verdict=verdict_out,
    )


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post(
    "/submit",
    response_model=ArticleOut,
    status_code=status.HTTP_201_CREATED,
    summary="Submit an article for verification",
)
def submit_article(
    req: ArticleSubmitRequest,
    db: Session = Depends(get_db),
    _user: dict = Depends(get_current_user),
) -> ArticleOut:
    """
    Submit a news article for automated fake-news detection.

    The pipeline:
    1. Extracts factual claims via LLM
    2. Retrieves evidence for each claim (web search)
    3. Classifies claim-evidence pairs with NLI
    4. Runs the stacking-ensemble model
    5. Adjudicates a final verdict

    If `confidence` falls between `ESCALATION_LOW` and `ESCALATION_HIGH` the article
    is flagged `needs_human_review=true` and added to the review queue.

    **Requires Bearer token** — obtain one at `POST /auth/token`.
    """
    # 1. Persist the article
    article = Article(text=req.text, url=req.url, status="processing")
    db.add(article)
    db.commit()
    db.refresh(article)
    log.info("submit_article: article_id=%d created", article.id)

    # 2. Run verification pipeline
    try:
        vr = run_verification_pipeline(req.text, req.url)
    except Exception as exc:
        log.error("submit_article: pipeline failed for article_id=%d: %s", article.id, exc)
        article.status = "failed"
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Verification pipeline error: {exc}",
        )

    # 3. Persist verdict
    verdict = Verdict(
        article_id=article.id,
        verdict=vr.verdict,
        confidence=vr.confidence,
        claim_verdicts=vr.claim_verdicts_json(),
        ensemble_score=vr.ensemble_score,
        reasoning_trace=vr.reasoning_trace,
        needs_human_review=vr.needs_human_review,
        ai_model_version="pipeline-v1",
    )
    db.add(verdict)
    db.flush()  # get verdict.id before commit

    # 4. Enqueue for human review if needed
    if vr.needs_human_review:
        rq = ReviewQueue(article_id=article.id, verdict_id=verdict.id)
        db.add(rq)
        log.info(
            "submit_article: article_id=%d flagged for human review", article.id
        )

    article.status = "completed"
    db.commit()
    db.refresh(article)

    return _build_article_out(article)


@router.get(
    "/{article_id}",
    response_model=ArticleOut,
    summary="Retrieve a stored article and its verdict",
)
def get_article(
    article_id: int,
    db: Session = Depends(get_db),
) -> ArticleOut:
    """
    Retrieve a previously submitted article by its ID, including the full verdict
    and reasoning trace.  No auth required.
    """
    article = db.get(Article, article_id)
    if article is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Article {article_id} not found",
        )
    return _build_article_out(article)
