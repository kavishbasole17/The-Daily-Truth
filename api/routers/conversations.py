"""
Chat history for signed-in users.

GET    /conversations           — the user's chats, most recent first
GET    /conversations/{id}      — one chat with every check in it (full text + verdict)
PATCH  /conversations/{id}      — rename a chat
DELETE /conversations/{id}      — delete a chat and its checks
POST   /conversations/claim     — after signing in, move this client's guest
                                  check(s) into a new chat so the thread continues
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.auth import get_current_user, guest_key
from api.database import get_db
from api.orm import Article, Conversation, Verdict
from api.routers.articles import ArticleOut, _build_article_out, _chat_title

router = APIRouter(prefix="/conversations", tags=["Chat history"])


class ConversationSummary(BaseModel):
    id: int
    title: str
    created_at: str
    updated_at: str
    message_count: int
    last_verdict: Optional[str] = None


class ChatTurn(ArticleOut):
    text: str  # full submitted text (or OCR transcription for an image)


class ConversationDetail(BaseModel):
    id: int
    title: str
    created_at: str
    updated_at: str
    turns: list[ChatTurn]


class RenameRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=120)


class ClaimRequest(BaseModel):
    article_ids: list[int] = Field(..., min_length=1, max_length=20)


def _iso(dt: Optional[datetime]) -> str:
    return dt.isoformat() if dt else ""


def _owned(db: Session, conversation_id: int, user: dict) -> Conversation:
    conv = db.get(Conversation, conversation_id)
    if conv is None or conv.user_id != user["id"]:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return conv


def _summary(conv: Conversation) -> ConversationSummary:
    last = conv.articles[-1] if conv.articles else None
    last_verdict = last.verdicts[-1].verdict if last is not None and last.verdicts else None
    return ConversationSummary(
        id=conv.id,
        title=conv.title,
        created_at=_iso(conv.created_at),
        updated_at=_iso(conv.updated_at or conv.created_at),
        message_count=len(conv.articles),
        last_verdict=last_verdict,
    )


def _detail(conv: Conversation) -> ConversationDetail:
    return ConversationDetail(
        id=conv.id,
        title=conv.title,
        created_at=_iso(conv.created_at),
        updated_at=_iso(conv.updated_at or conv.created_at),
        turns=[
            ChatTurn(**_build_article_out(a).model_dump(), text=a.text or "")
            for a in conv.articles
        ],
    )


@router.get("", response_model=list[ConversationSummary], summary="List my chats")
def list_conversations(
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[ConversationSummary]:
    convs = db.execute(
        select(Conversation)
        .where(Conversation.user_id == user["id"])
        .order_by(Conversation.updated_at.desc(), Conversation.id.desc())
    ).scalars().all()
    return [_summary(c) for c in convs]


@router.post("/claim", response_model=ConversationDetail, status_code=status.HTTP_201_CREATED,
             summary="Move my signed-out check(s) into a new chat")
def claim_guest_articles(
    req: ClaimRequest,
    request: Request,
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ConversationDetail:
    """
    Only unowned articles submitted from this same client (guest key) can be
    claimed, so one person can't pull someone else's guest check into their
    history by guessing ids.
    """
    key = guest_key(request)
    articles = db.execute(
        select(Article)
        .where(Article.id.in_(req.article_ids),
               Article.user_id.is_(None),
               Article.guest_key == key)
        .order_by(Article.id)
    ).scalars().all()
    if not articles:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="No claimable checks found")

    conv = Conversation(user_id=user["id"], title=_chat_title(articles[0].text or "Image check"))
    db.add(conv)
    db.flush()
    for a in articles:
        a.user_id = user["id"]
        a.conversation_id = conv.id
        a.guest_key = None
    db.commit()
    db.refresh(conv)
    return _detail(conv)


@router.get("/{conversation_id}", response_model=ConversationDetail, summary="Open a chat")
def get_conversation(
    conversation_id: int,
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ConversationDetail:
    return _detail(_owned(db, conversation_id, user))


@router.patch("/{conversation_id}", response_model=ConversationSummary, summary="Rename a chat")
def rename_conversation(
    conversation_id: int,
    req: RenameRequest,
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ConversationSummary:
    conv = _owned(db, conversation_id, user)
    conv.title = req.title.strip()
    db.commit()
    db.refresh(conv)
    return _summary(conv)


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT,
               summary="Delete a chat")
def delete_conversation(
    conversation_id: int,
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Response:
    conv = _owned(db, conversation_id, user)
    for a in list(conv.articles):
        for v in list(a.verdicts):
            db.delete(v)
        db.delete(a)
    db.delete(conv)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
