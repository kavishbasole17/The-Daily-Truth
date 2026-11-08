"""
GET /share/{slug} — a shared check, readable by anyone with the link.

Only what's needed to read the verdict is exposed: no owner, no conversation,
no internal ids. Turning sharing off (DELETE /articles/{id}/share) makes the
slug 404 immediately.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.database import get_db
from api.orm import Article
from api.routers.articles import VerdictOut, _build_article_out

router = APIRouter(prefix="/share", tags=["Sharing"])


class SharedCheck(BaseModel):
    slug: str
    kind: str | None = None
    url: str | None = None
    source_title: str | None = None
    text: str
    submitted_at: str
    verdict: VerdictOut


@router.get("/{slug}", response_model=SharedCheck, summary="Read a shared check")
def get_shared(slug: str, db: Session = Depends(get_db)) -> SharedCheck:
    article = db.execute(
        select(Article).where(Article.share_slug == slug)
    ).scalar_one_or_none() if slug else None
    if article is None or article.status != "completed" or not article.verdicts:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="This shared check doesn't exist or was unshared.")
    out = _build_article_out(article)
    return SharedCheck(
        slug=slug,
        kind=article.kind,
        url=article.url,
        source_title=article.source_title,
        text=article.text or "",
        submitted_at=out.submitted_at,
        verdict=out.verdict,
    )
