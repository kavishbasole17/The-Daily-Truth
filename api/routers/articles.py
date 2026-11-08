"""
Article submission and retrieval endpoints.

POST /articles/submit        — run the verification pipeline over text, persist the result.
POST /articles/submit-image  — OCR an uploaded image, then run the same pipeline.
GET  /articles/{id}          — retrieve a stored article + verdict.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Request, UploadFile, status
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import func
from sqlalchemy.orm import Session

from api.auth import consume_guest_query, get_optional_user, guest_key
from api.database import get_db
from api.orm import Article, Conversation, Verdict
from paper_trail_core.agents.pipeline import run_verification_pipeline, run_verification_pipeline_image

log = logging.getLogger(__name__)
router = APIRouter(prefix="/articles", tags=["Articles"])

AI_MODEL_VERSION = "pipeline-v2-evidence-fusion"

try:
    from config import settings
    _OCR_MAX_IMAGE_MB = settings.OCR_MAX_IMAGE_MB
except ImportError:  # pragma: no cover
    _OCR_MAX_IMAGE_MB = 10.0

# Content-type allowlist rather than trusting the client-supplied filename —
# an uploaded file's extension is not evidence of its actual format, and
# PIL.Image.open (used downstream by paper_trail_core/agents/ocr_agent.py)
# only needs the bytes to be a real image, not a name that ends in .jpg.
_EXT_BY_CONTENT_TYPE = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
}


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
    conversation_id: Optional[int] = Field(
        None, description="Signed-in only: append to this chat. Omit to start a new chat.",
    )
    url:  Optional[str] = Field(None, max_length=2048, description="Source URL (stored for reference)")
    region: Optional[str] = Field(
        None, max_length=120,
        description="Optional place the article is about (e.g. 'Pune' or 'Kerala'). "
                    "Used as context when the text does not name its location.",
    )

    @model_validator(mode="after")
    def _require_text_or_url(self):
        if not self.text and not self.url:
            raise ValueError("Provide either 'text' or 'url'.")
        if not self.text and self.url:
            # Future: fetch text from URL — for now require text
            raise ValueError("URL-only submission not yet supported. Please include 'text'.")
        return self


class LocationOut(BaseModel):
    """Where a claim is about, resolved as precisely as the text allows."""
    name:         Optional[str] = None
    label:        Optional[str] = None      # "Koregaon Park, Pune, Maharashtra, India"
    level:        Optional[str] = None      # site | street | locality | city | district | state | country
    lat:          Optional[float] = None
    lon:          Optional[float] = None
    country:      Optional[str] = None
    country_code: Optional[str] = None
    state:        Optional[str] = None
    district:     Optional[str] = None
    city:         Optional[str] = None
    locality:     Optional[str] = None
    confidence:   float = 0.0
    method:       Optional[str] = None      # nominatim | gazetteer | user
    source:       Optional[str] = None      # claim | article | user
    mentions:     list[str] = []
    ambiguous:    bool = False
    also:         list[dict] = []           # other places named in the same claim


class ClaimVerdictOut(BaseModel):
    claim:             str
    verdict:           str
    confidence:        float
    evidence_snippets: list[str] = []

    # Diagnostics added by the rebuilt verifier. All optional with defaults, so
    # verdict rows written by the previous version still deserialise.
    support:           float = 0.0
    entail_mass:       float = 0.0
    contradict_mass:   float = 0.0
    top_relevance:     float = 0.0
    numeric_conflict:  float = 0.0
    sources:           list[str] = []
    status:            str = "ok"

    # Region-aware verification. All optional: absent for claims with no place,
    # and for verdict rows stored before this existed.
    location:          Optional[LocationOut] = None
    search_scope:      Optional[str] = None    # narrowest region that returned evidence
    regional_evidence: int = 0                 # evidence snippets from the claim's region
    in_region:         list[bool] = []         # parallel to evidence_snippets
    event_corroborated: bool = False           # recent independent outlets report the same event
    corroborating_sources: list[str] = []
    detail_mismatch: bool = False              # event confirmed, but the claim's figures differ


class DegradationOut(BaseModel):
    stage:    str
    kind:     str
    detail:   str
    severity: str


class VerdictOut(BaseModel):
    id:                 int
    article_id:         int
    verdict:            str
    confidence:         float
    claim_verdicts:     list[ClaimVerdictOut]
    ensemble_score:     float
    reasoning_trace:    str
    ai_model_version:   Optional[str]
    created_at:         str
    fused_score:        Optional[float] = None
    degradations:       list[DegradationOut] = []

    model_config = ConfigDict(from_attributes=True)


class ArticleOut(BaseModel):
    id:           int
    url:          Optional[str]
    text_preview: str           # first 200 chars
    status:       str           # pending | processing | completed | failed
    submitted_at: str
    stage:        Optional[str] = None
    kind:         Optional[str] = None          # text | image
    conversation_id: Optional[int] = None       # None for a signed-out (guest) query
    verdict:      Optional[VerdictOut] = None

    model_config = ConfigDict(from_attributes=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_article_out(article: Article) -> ArticleOut:
    verdict_out = None
    if article.verdicts:
        v = article.verdicts[-1]
        cvs_raw = json.loads(v.claim_verdicts) if v.claim_verdicts else []
        degradations_raw = json.loads(v.degradations) if getattr(v, "degradations", None) else []
        verdict_out = VerdictOut(
            id=v.id,
            article_id=v.article_id,
            verdict=v.verdict,
            confidence=v.confidence,
            claim_verdicts=[ClaimVerdictOut(**cv) for cv in cvs_raw],
            ensemble_score=v.ensemble_score,
            reasoning_trace=v.reasoning_trace,
            ai_model_version=v.ai_model_version,
            created_at=v.created_at.isoformat(),
            fused_score=getattr(v, "fused_score", None),
            degradations=[DegradationOut(**d) for d in degradations_raw],
        )
    return ArticleOut(
        id=article.id,
        url=article.url,
        text_preview=article.text[:200],
        status=article.status,
        submitted_at=article.submitted_at.isoformat(),
        stage=article.stage,
        kind=article.kind,
        conversation_id=article.conversation_id,
        verdict=verdict_out,
    )


def _chat_title(text: str) -> str:
    """A short sidebar title from the first line of the submission."""
    first = " ".join((text or "").split())
    return (first[:60].rstrip() + "…") if len(first) > 60 else (first or "Untitled check")


def _assign_owner(
    db: Session,
    request: Request,
    user: Optional[dict],
    conversation_id: Optional[int],
    title_text: str,
) -> dict:
    """
    Decide who owns a new submission.

    Signed-in: the article joins `conversation_id` (which must be theirs) or a
    new conversation titled from the text. Signed-out: one of the guest's free
    queries is spent (401 `guest_limit_reached` once they're gone) and the
    article is unowned, tagged with the guest key so it can be claimed into a
    chat after sign-in.
    """
    if user is None:
        key = guest_key(request)
        consume_guest_query(db, key)
        return {"user_id": None, "conversation_id": None, "guest_key": key}

    if conversation_id is not None:
        conv = db.get(Conversation, conversation_id)
        if conv is None or conv.user_id != user["id"]:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Conversation not found")
        conv.updated_at = func.now()
    else:
        conv = Conversation(user_id=user["id"], title=_chat_title(title_text))
        db.add(conv)
        db.flush()
    return {"user_id": user["id"], "conversation_id": conv.id, "guest_key": None}


def _can_read(article: Article, user: Optional[dict]) -> bool:
    """Owned articles are private to their owner (admins see everything)."""
    if article.user_id is None:
        return True
    return user is not None and (user["id"] == article.user_id or user["role"] == "admin")


def _run_pipeline_job(article_id: int, text: str, url: Optional[str],
                      region: Optional[str] = None) -> None:
    """
    Background worker: run the pipeline and persist the result.

    Opens its own session — the request's session is closed as soon as the 202
    is returned, so reusing it here would fail. Any exception is caught and
    recorded as a failed article rather than escaping into the background task
    runner, where it would vanish into a log and leave the row stuck on
    "processing" forever.
    """
    from api.database import SessionLocal

    db = SessionLocal()
    try:
        article = db.get(Article, article_id)
        if article is None:
            log.error("pipeline job: article_id=%d disappeared", article_id)
            return

        article.status = "processing"
        article.stage = "verifying"
        db.commit()

        vr = run_verification_pipeline(text, url, region)

        verdict = Verdict(
            article_id=article_id,
            verdict=vr.verdict,
            confidence=vr.confidence,
            claim_verdicts=vr.claim_verdicts_json(),
            ensemble_score=vr.ensemble_score,
            reasoning_trace=vr.reasoning_trace,
            degradations=vr.degradations_json(),
            fused_score=vr.fused_score,
            ai_model_version=AI_MODEL_VERSION,
        )
        db.add(verdict)

        article.status = "completed"
        article.stage = None
        db.commit()
        log.info("pipeline job: article_id=%d done — %s (%.3f)",
                 article_id, vr.verdict, vr.confidence)

    except Exception as exc:
        log.error("pipeline job: article_id=%d failed — %s", article_id, exc, exc_info=True)
        db.rollback()
        try:
            article = db.get(Article, article_id)
            if article is not None:
                article.status = "failed"
                article.stage = None
                db.commit()
        except Exception:
            log.exception("pipeline job: could not mark article_id=%d failed", article_id)
    finally:
        db.close()


def _run_pipeline_image_job(article_id: int, image_path: str, url: Optional[str],
                            region: Optional[str] = None) -> None:
    """
    Background worker for an image submission: OCR the image, run the same
    evidence-grounded pipeline used for text, persist the result, and clean up
    the temp file the upload was written to — mirrors `_run_pipeline_job`
    above, see its docstring for why this opens its own DB session.
    """
    from api.database import SessionLocal

    db = SessionLocal()
    try:
        article = db.get(Article, article_id)
        if article is None:
            log.error("pipeline job (image): article_id=%d disappeared", article_id)
            return

        article.status = "processing"
        article.stage = "verifying"
        db.commit()

        vr = run_verification_pipeline_image(image_path, url, region)

        # The article had no text at submission time — this is the first
        # point the OCR (and, if applicable, translated) transcription is
        # known, so it's what text_preview and any future re-read of this
        # article will show.
        if vr.article_text:
            article.text = vr.article_text

        verdict = Verdict(
            article_id=article_id,
            verdict=vr.verdict,
            confidence=vr.confidence,
            claim_verdicts=vr.claim_verdicts_json(),
            ensemble_score=vr.ensemble_score,
            reasoning_trace=vr.reasoning_trace,
            degradations=vr.degradations_json(),
            fused_score=vr.fused_score,
            ai_model_version=AI_MODEL_VERSION,
        )
        db.add(verdict)

        article.status = "completed"
        article.stage = None
        db.commit()
        log.info("pipeline job (image): article_id=%d done — %s (%.3f)",
                 article_id, vr.verdict, vr.confidence)

    except Exception as exc:
        log.error("pipeline job (image): article_id=%d failed — %s", article_id, exc, exc_info=True)
        db.rollback()
        try:
            article = db.get(Article, article_id)
            if article is not None:
                article.status = "failed"
                article.stage = None
                db.commit()
        except Exception:
            log.exception("pipeline job (image): could not mark article_id=%d failed", article_id)
    finally:
        db.close()
        try:
            os.remove(image_path)
        except OSError:
            pass


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post(
    "/submit",
    response_model=ArticleOut,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit an article for verification (async)",
)
def submit_article(
    req: ArticleSubmitRequest,
    background: BackgroundTasks,
    request: Request,
    db: Session = Depends(get_db),
    user: Optional[dict] = Depends(get_optional_user),
) -> ArticleOut:
    """
    Submit a news article for automated fake-news detection.

    Returns **202 Accepted** immediately with `status="processing"`. Poll
    `GET /articles/{id}` until `status` becomes `completed` or `failed`; the
    verdict is attached when it does.

    This is asynchronous because the pipeline does real work — evidence
    retrieval against live sources plus transformer NLI — which takes seconds,
    not milliseconds, and a synchronous call would sit at the mercy of however
    slow an external search provider happens to be that minute.

    The pipeline:
      1. Extracts check-worthy factual claims from the article
      1b. Locates each claim: the place it is about, resolved as precisely as the
          text supports (see `claim_verdicts[].location`)
      2. Retrieves evidence for each claim from trusted sources — and, for a
         located claim, regional news from its area, widening from street to
         city to state to country until there is enough
      3. Classifies each (evidence, claim) pair with an NLI model
      4. Scores the article with the stacking ensemble
      5. Fuses both signals into a calibrated verdict, abstaining when the
         evidence does not support a confident answer

    Articles the system will not commit to come back as `uncertain`, with the
    reason in `reasoning_trace` and `degradations`.

    **Auth:** signed-out callers get `GUEST_QUERY_LIMIT` free submissions
    (default 1); after that a Bearer token is required (401 with
    `detail.code = "guest_limit_reached"`). Signed-in submissions are saved to
    a conversation — pass `conversation_id` to continue one.
    """
    owner = _assign_owner(db, request, user, req.conversation_id, req.text)
    article = Article(text=req.text, url=req.url, status="processing", stage="queued",
                      kind="text", **owner)
    db.add(article)
    db.commit()
    db.refresh(article)
    log.info("submit_article: article_id=%d queued", article.id)

    background.add_task(_run_pipeline_job, article.id, req.text, req.url, req.region)

    return _build_article_out(article)


@router.post(
    "/submit-image",
    response_model=ArticleOut,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a photo/screenshot of an article for OCR + verification (async)",
)
async def submit_article_image(
    background: BackgroundTasks,
    request: Request,
    file: UploadFile = File(..., description="Photo or screenshot of the article"),
    url: Optional[str] = Form(None, max_length=2048),
    region: Optional[str] = Form(None, max_length=120),
    conversation_id: Optional[int] = Form(None),
    db: Session = Depends(get_db),
    user: Optional[dict] = Depends(get_optional_user),
) -> ArticleOut:
    """
    Submit a photographed or screenshotted article image for verification.

    Runs OCR first (Tesseract, Hindi+English —
    `paper_trail_core/agents/ocr_agent.py`) to recover the article text, then
    feeds the transcription through the same evidence-grounded pipeline used
    by `POST /articles/submit`. If OCR confidence falls below the configured
    floor, the pipeline abstains (`uncertain`) rather than verifying a
    possibly-wrong transcription.

    Returns **202 Accepted** immediately; poll `GET /articles/{id}` exactly as
    for a text submission — the recovered text appears in `text_preview` once
    processing completes.

    **Auth:** same guest allowance and conversation handling as
    `POST /articles/submit`.
    """
    content_type = (file.content_type or "").lower()
    ext = _EXT_BY_CONTENT_TYPE.get(content_type)
    if ext is None:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported image type '{file.content_type or 'unknown'}'. "
                f"Allowed: {', '.join(sorted(_EXT_BY_CONTENT_TYPE))}."
            ),
        )

    max_bytes = int(_OCR_MAX_IMAGE_MB * 1024 * 1024)
    # Read one byte past the limit so an oversized upload is caught here,
    # before anything is written to disk, rather than mid-write.
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image exceeds the {_OCR_MAX_IMAGE_MB:.0f}MB limit.",
        )
    if not data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file upload.")

    # Validate the upload before spending a guest query on it.
    owner = _assign_owner(db, request, user, conversation_id,
                          f"Image: {file.filename or 'pasted image'}")

    # A random server-side name (tempfile), not the client-supplied filename —
    # the filename never touches the filesystem path.
    fd, image_path = tempfile.mkstemp(prefix="paper_trail_upload_", suffix=ext)
    with os.fdopen(fd, "wb") as f:
        f.write(data)

    article = Article(text="", url=url, status="processing", stage="queued",
                      kind="image", **owner)
    db.add(article)
    db.commit()
    db.refresh(article)
    log.info("submit_article_image: article_id=%d queued (%d bytes, %s)",
              article.id, len(data), content_type)

    background.add_task(_run_pipeline_image_job, article.id, image_path, url, region)

    return _build_article_out(article)


@router.get(
    "/{article_id}",
    response_model=ArticleOut,
    summary="Retrieve a stored article and its verdict",
)
def get_article(
    article_id: int,
    db: Session = Depends(get_db),
    user: Optional[dict] = Depends(get_optional_user),
) -> ArticleOut:
    """
    Retrieve a previously submitted article by its ID, including the full verdict
    and reasoning trace. Guest submissions are readable without auth; articles
    saved to an account are visible only to that account (and admins).
    """
    article = db.get(Article, article_id)
    if article is None or not _can_read(article, user):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Article {article_id} not found",
        )
    return _build_article_out(article)
