"""
Chat history for signed-in users.

GET    /conversations           — the user's chats, most recent first
GET    /conversations/{id}      — one chat with every check in it (full text + verdict)
PATCH  /conversations/{id}      — rename a chat
DELETE /conversations/{id}      — delete a chat and its checks
POST   /conversations/claim     — after signing in, move this client's guest
                                  check(s) into a new chat so the thread continues
POST   /conversations/{id}/ask  — (legacy, synchronous) evidence-only answer (api/qa.py)
POST   /conversations/{id}/messages        — talk with the assistant about this chat:
                                  answered in the background from every check and
                                  earlier exchange; it may run new checks (api/chat.py)
GET    /conversations/{id}/messages/{mid}  — poll a reply
"""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from api.auth import get_current_user, guest_key
from api.chat import chat_reply
from api.database import get_db
from api.orm import Article, ChatQuestion, Conversation
from api.qa import answer_question, dumps_citations
from api.routers.articles import ArticleOut, _build_article_out, _chat_title

router = APIRouter(prefix="/conversations", tags=["Chat history"])


class ConversationSummary(BaseModel):
    id: int
    title: str
    created_at: str
    updated_at: str
    message_count: int
    last_verdict: str | None = None


class ChatTurn(ArticleOut):
    text: str  # full submitted text (or OCR transcription for an image)


class CitationOut(BaseModel):
    n: int
    claim: str = ""
    source: str = ""
    url: str = ""
    snippet: str = ""


class FocusIn(BaseModel):
    article_id: int
    claim_index: int | None = Field(None, ge=0, description="Omit to discuss the whole verdict")


class QuestionOut(BaseModel):
    id: int
    question: str
    answer: str
    citations: list[CitationOut] = []
    mode: str                     # llm | extractive | heuristic
    created_at: str
    status: str = "done"          # pending | thinking | done | failed
    stage: str | None = None   # thinking | verifying | writing
    focus: FocusIn | None = None
    checks: list[ArticleOut] = [] # checks the assistant ran while replying
    error: str | None = None


class ChatMessageIn(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    focus: FocusIn | None = None


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=2000)


class ConversationDetail(BaseModel):
    id: int
    title: str
    created_at: str
    updated_at: str
    turns: list[ChatTurn]
    # Follow-up Q&A, oldest first. Interleave with `turns` by timestamp.
    questions: list[QuestionOut] = []


class RenameRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=120)


class ClaimRequest(BaseModel):
    article_ids: list[int] = Field(..., min_length=1, max_length=20)


def _iso(dt: datetime | None) -> str:
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
        # your checks + your follow-up messages (checks the assistant ran belong to those)
        message_count=sum(1 for a in conv.articles if a.kind != "chat") + len(conv.questions),
        last_verdict=last_verdict,
    )


def _detail(conv: Conversation) -> ConversationDetail:
    return ConversationDetail(
        id=conv.id,
        title=conv.title,
        created_at=_iso(conv.created_at),
        updated_at=_iso(conv.updated_at or conv.created_at),
        # Checks the assistant ran mid-conversation belong to the message that
        # ran them, not to the list of user-submitted checks.
        turns=[
            ChatTurn(**_build_article_out(a).model_dump(), text=a.text or "")
            for a in conv.articles if a.kind != "chat"
        ],
        questions=[_question_out(q, conv) for q in conv.questions],
    )


def _loads(raw, default):
    try:
        return json.loads(raw) if raw else default
    except ValueError:
        return default


def _question_out(q: ChatQuestion, conv: Conversation | None = None) -> QuestionOut:
    cites = _loads(q.citations, [])
    focus = _loads(q.focus, None)
    ids = _loads(q.article_ids, [])
    by_id = {a.id: a for a in (conv or q.conversation).articles} if ids else {}
    return QuestionOut(
        id=q.id, question=q.question, answer=q.answer or "",
        citations=[CitationOut(**c) for c in cites],
        mode=q.mode or "llm", created_at=_iso(q.created_at),
        status=q.status or "done", stage=q.stage,
        focus=FocusIn(**focus) if focus else None,
        checks=[_build_article_out(by_id[i]) for i in ids if i in by_id],
        error=q.error_message,
    )


def _finished_checks(conv: Conversation) -> list[dict]:
    return [
        {**_build_article_out(a).model_dump(), "text": a.text or ""}
        for a in conv.articles if a.status == "completed"
    ]


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


@router.post("/{conversation_id}/ask", response_model=QuestionOut,
             status_code=status.HTTP_201_CREATED,
             summary="Ask a follow-up question about this chat's checks")
def ask_question(
    conversation_id: int,
    req: AskRequest,
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> QuestionOut:
    """
    Answers from the claims, verdicts and evidence already in this chat, citing
    evidence as [n]. Without an LLM configured, returns the most relevant
    evidence instead (`mode = "extractive"`).
    """
    conv = _owned(db, conversation_id, user)
    finished = [a for a in conv.articles if a.status == "completed"]
    if not finished:
        raise HTTPException(status.HTTP_409_CONFLICT,
                            detail="Ask once at least one check in this chat has finished.")

    checks = [
        {**_build_article_out(a).model_dump(), "text": a.text or ""}
        for a in finished
    ]
    history = [(q.question, q.answer) for q in conv.questions]
    result = answer_question(req.question.strip(), checks, history)

    q = ChatQuestion(
        conversation_id=conv.id,
        question=req.question.strip(),
        answer=result.answer,
        citations=dumps_citations(result.citations),
        mode=result.mode,
    )
    db.add(q)
    conv.updated_at = func.now()
    db.commit()
    db.refresh(q)
    return _question_out(q)


# ── Conversational follow-ups ──────────────────────────────────────────────────

def _run_chat_job(question_id: int) -> None:
    """
    Background worker for one chat message. Opens its own session (the request
    has returned). Checks the assistant decides to run are created as articles
    in this conversation (kind="chat") and verified with the same pipeline job
    as user submissions, so they persist, can be shared and rated, and become
    evidence for every later message.
    """
    from api.database import SessionLocal
    from api.routers.articles import _run_pipeline_job

    db = SessionLocal()
    try:
        q = db.get(ChatQuestion, question_id)
        if q is None:
            return
        conv = q.conversation
        q.status, q.stage = "thinking", "thinking"
        db.commit()

        checks = _finished_checks(conv)
        history = [{"question": h.question, "answer": h.answer}
                   for h in conv.questions if h.id != q.id and h.status in (None, "done")]
        focus = _loads(q.focus, None)
        ran: list[int] = []

        def on_stage(stage: str) -> None:
            q.stage = stage
            db.commit()

        def verify(statement: str) -> dict:
            article = Article(text=statement, status="processing", stage="queued", kind="chat",
                              user_id=conv.user_id, conversation_id=conv.id)
            db.add(article)
            db.commit()
            ran.append(article.id)
            q.article_ids = json.dumps(ran)   # lets the UI show the check while it runs
            db.commit()
            _run_pipeline_job(article.id, statement, None, None)
            db.expire_all()
            article = db.get(Article, article.id)
            return {**_build_article_out(article).model_dump(), "text": article.text or ""}

        reply = chat_reply(q.question, focus, checks, history, verify, on_stage)

        q.answer = reply.answer
        q.citations = dumps_citations(reply.citations)
        q.mode = reply.mode
        q.article_ids = json.dumps(sorted(set(ran) | set(reply.article_ids)))
        q.status, q.stage = "done", None
        conv.updated_at = func.now()
        db.commit()
    except Exception:
        import logging
        logging.getLogger(__name__).exception("chat job %d failed", question_id)
        db.rollback()
        q = db.get(ChatQuestion, question_id)
        if q is not None:
            q.status, q.stage = "failed", None
            q.error_message = "Something went wrong while replying. Please try again."
            db.commit()
    finally:
        db.close()


@router.post("/{conversation_id}/messages", response_model=QuestionOut,
             status_code=status.HTTP_202_ACCEPTED,
             summary="Talk with the assistant about this chat")
def send_message(
    conversation_id: int,
    req: ChatMessageIn,
    background: BackgroundTasks,
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> QuestionOut:
    """
    Continue the conversation. The reply draws on every check in this chat and
    the earlier exchange; set `focus` to discuss one finding (a claim, or a
    whole verdict). The assistant may run new checks on statements you make
    or ask about — those appear in `checks` and become part of the chat.

    Returns **202** with `status="pending"`; poll
    `GET /conversations/{id}/messages/{message_id}` until `done` or `failed`.
    """
    conv = _owned(db, conversation_id, user)
    if req.focus is not None:
        target = db.get(Article, req.focus.article_id)
        if target is None or target.conversation_id != conv.id:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="That finding isn't in this chat.")
    # One reply at a time. A reply stuck for 5+ minutes (e.g. the server
    # restarted mid-job) no longer blocks the chat.
    # created_at is stored as naive UTC (SQLite CURRENT_TIMESTAMP), so compare naive UTC.
    stale_before = datetime.now(UTC).replace(tzinfo=None) - timedelta(minutes=5)
    if any(q.status in ("pending", "thinking") and q.created_at and q.created_at > stale_before
           for q in conv.questions):
        raise HTTPException(status.HTTP_409_CONFLICT,
                            detail="Still replying to your last message. One moment.")

    q = ChatQuestion(
        conversation_id=conv.id,
        question=req.message.strip(),
        answer="",
        status="pending",
        stage="thinking",
        focus=json.dumps(req.focus.model_dump()) if req.focus else None,
        mode="llm",
    )
    db.add(q)
    db.commit()
    db.refresh(q)
    background.add_task(_run_chat_job, q.id)
    return _question_out(q, conv)


@router.get("/{conversation_id}/messages/{message_id}", response_model=QuestionOut,
            summary="Poll an assistant reply")
def get_message(
    conversation_id: int,
    message_id: int,
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> QuestionOut:
    conv = _owned(db, conversation_id, user)
    q = db.get(ChatQuestion, message_id)
    if q is None or q.conversation_id != conv.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Message not found")
    return _question_out(q, conv)
