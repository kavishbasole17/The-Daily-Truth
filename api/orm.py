"""
SQLAlchemy ORM table definitions.

Schema notes (for Person D — source-credibility / analytics tables):
  - articles.id  is the shared foreign key for all cross-team joins.
  - verdicts.article_id references articles.id.
  - Use INTEGER type for all FK columns so joins work across SQLite and PostgreSQL.
"""
from __future__ import annotations

import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from api.database import Base


class User(Base):
    """
    An account. Local accounts carry a password hash; Google accounts carry the
    Google subject id instead and have no password.
    """
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    email: Mapped[str | None] = mapped_column(String(255), unique=True, index=True, nullable=True)
    full_name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    hashed_password: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # local | google
    auth_provider: Mapped[str] = mapped_column(String(32), default="local")
    google_sub: Mapped[str | None] = mapped_column(String(255), unique=True, nullable=True)
    # admin | analyst | user
    role: Mapped[str] = mapped_column(String(32), default="user")
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now()
    )

    conversations: Mapped[list[Conversation]] = relationship(
        "Conversation", back_populates="user", cascade="all, delete-orphan"
    )


class Conversation(Base):
    """A chat thread on the Paper Trail page: an ordered run of articles checked."""
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(String(120), nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    user: Mapped[User] = relationship("User", back_populates="conversations")
    articles: Mapped[list[Article]] = relationship(
        "Article", back_populates="conversation", order_by="Article.id"
    )
    questions: Mapped[list[ChatQuestion]] = relationship(
        "ChatQuestion", back_populates="conversation", order_by="ChatQuestion.id",
        cascade="all, delete-orphan",
    )


class GuestUsage(Base):
    """
    Counts verification queries made without signing in, keyed by client IP.
    Lives server-side so the free-query allowance can't be reset by clearing
    browser storage.
    """
    __tablename__ = "guest_usage"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    client_key: Mapped[str] = mapped_column(String(128), unique=True, index=True, nullable=False)
    query_count: Mapped[int] = mapped_column(Integer, default=0)
    last_used_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class Article(Base):
    __tablename__ = "articles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    # Owner. NULL for the free guest query (and rows written before accounts).
    user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=True, index=True
    )
    conversation_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("conversations.id"), nullable=True, index=True
    )
    # Who submitted a guest article, so it can be claimed into a conversation
    # by the same client after they sign in.
    guest_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    # text | image | url | chat (a claim the assistant checked mid-conversation)
    kind: Mapped[str | None] = mapped_column(String(16), nullable=True)
    # For a link check: the page title the fetcher found.
    source_title: Mapped[str | None] = mapped_column(String(300), nullable=True)
    # Why a check failed, in words a user can act on ("the site needs a login").
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Public read-only link id (/v/<slug>); NULL until the owner shares it.
    share_slug: Mapped[str | None] = mapped_column(String(32), unique=True, index=True, nullable=True)
    url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    # Raw text stored verbatim (no cleaning) so the API can echo it back
    text: Mapped[str] = mapped_column(Text, nullable=False)
    # pending → processing → completed | failed
    status: Mapped[str] = mapped_column(String(32), default="pending")
    # Current pipeline stage while status == "processing", so the polling UI can
    # show progress rather than an opaque spinner.
    stage: Mapped[str | None] = mapped_column(String(64), nullable=True)
    submitted_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now()
    )

    verdicts: Mapped[list[Verdict]] = relationship("Verdict", back_populates="article")
    conversation: Mapped[Conversation] = relationship(
        "Conversation", back_populates="articles"
    )
    feedback: Mapped[list[Feedback]] = relationship(
        "Feedback", back_populates="article", cascade="all, delete-orphan"
    )


class Feedback(Base):
    """
    A user's rating of a verdict. Kept as labelled data: a thumbs-down with a
    reason is exactly the signal needed to re-fit the fusion adjudicator
    (scripts/train_fusion.py) on cases the pipeline got wrong.
    """
    __tablename__ = "feedback"
    __table_args__ = (UniqueConstraint("article_id", "user_id", name="uq_feedback_article_user"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    article_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("articles.id"), nullable=False, index=True
    )
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    rating: Mapped[int] = mapped_column(Integer, nullable=False)        # +1 | -1
    # wrong_verdict | bad_sources | missed_claim | other (only for -1)
    reason: Mapped[str | None] = mapped_column(String(32), nullable=True)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    # What the verdict was when it was rated, so later re-runs don't blur the label.
    verdict_at_rating: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    article: Mapped[Article] = relationship("Article", back_populates="feedback")


class ChatQuestion(Base):
    """A follow-up question asked in a conversation, answered from its evidence."""
    __tablename__ = "chat_questions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("conversations.id"), nullable=False, index=True
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    # JSON list of {n, claim, source, url, snippet}
    citations: Mapped[str | None] = mapped_column(Text, nullable=True)
    # llm | extractive | heuristic
    mode: Mapped[str] = mapped_column(String(16), default="llm")
    # Chat messages are answered in the background: pending -> thinking -> done | failed
    status: Mapped[str | None] = mapped_column(String(16), nullable=True, default="done")
    # thinking | verifying | writing — what the assistant is doing right now
    stage: Mapped[str | None] = mapped_column(String(16), nullable=True)
    # JSON {article_id, claim_index|null}: the finding this message is about
    focus: Mapped[str | None] = mapped_column(Text, nullable=True)
    # JSON list of Article ids for checks the assistant ran while replying
    article_ids: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())

    conversation: Mapped[Conversation] = relationship("Conversation", back_populates="questions")


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
    # Deprecated and never read: human review was removed from the pipeline. The
    # column stays so databases created before the removal (where it is NOT NULL
    # with no server default) keep accepting inserts.
    needs_human_review: Mapped[bool] = mapped_column(Boolean, default=False)
    # JSON-encoded list of Degradation dicts — records every point at which the
    # pipeline ran below full capability for this verdict.
    degradations: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Calibrated P(real) before thresholding, kept for calibration analysis.
    fused_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    ai_model_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now()
    )

    article: Mapped[Article] = relationship("Article", back_populates="verdicts")
