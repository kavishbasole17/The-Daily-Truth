"""
SQLAlchemy ORM table definitions.

Schema notes (for Person D — source-credibility / analytics tables):
  - articles.id  is the shared foreign key for all cross-team joins.
  - verdicts.article_id references articles.id.
  - review_queue.article_id references articles.id.
  - Use INTEGER type for all FK columns so joins work across SQLite and PostgreSQL.
"""
from __future__ import annotations

import datetime

from sqlalchemy import (
    Boolean, DateTime, Float, ForeignKey, Integer, String, Text, func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from api.database import Base


class Article(Base):
    __tablename__ = "articles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    # Raw text stored verbatim (no cleaning) so the API can echo it back
    text: Mapped[str] = mapped_column(Text, nullable=False)
    # pending → processing → completed | failed
    status: Mapped[str] = mapped_column(String(32), default="pending")
    submitted_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now()
    )

    verdicts: Mapped[list["Verdict"]] = relationship("Verdict", back_populates="article")
    review_items: Mapped[list["ReviewQueue"]] = relationship(
        "ReviewQueue", back_populates="article"
    )


class Verdict(Base):
    __tablename__ = "verdicts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    article_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("articles.id"), nullable=False, index=True
    )
    # likely_fake | likely_real | uncertain
    verdict: Mapped[str] = mapped_column(String(32), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    # JSON-encoded list of ClaimVerdict dicts
    claim_verdicts: Mapped[str | None] = mapped_column(Text, nullable=True)
    ensemble_score: Mapped[float] = mapped_column(Float, nullable=False)
    reasoning_trace: Mapped[str] = mapped_column(Text, nullable=False)
    needs_human_review: Mapped[bool] = mapped_column(Boolean, default=False)
    ai_model_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now()
    )

    article: Mapped["Article"] = relationship("Article", back_populates="verdicts")
    review_items: Mapped[list["ReviewQueue"]] = relationship(
        "ReviewQueue", back_populates="verdict"
    )


class ReviewQueue(Base):
    __tablename__ = "review_queue"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    article_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("articles.id"), nullable=False, index=True
    )
    verdict_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("verdicts.id"), nullable=False
    )
    # pending | in_review | resolved
    status: Mapped[str] = mapped_column(String(32), default="pending")
    assigned_to: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now()
    )
    resolved_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime, nullable=True
    )
    # Analyst-supplied ground truth
    human_verdict: Mapped[str | None] = mapped_column(String(32), nullable=True)
    human_notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    article: Mapped["Article"] = relationship("Article", back_populates="review_items")
    verdict: Mapped["Verdict"] = relationship("Verdict", back_populates="review_items")
