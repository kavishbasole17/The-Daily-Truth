"""
Authentication endpoints.

POST /auth/token     — sign in with username-or-email + password
POST /auth/register  — create a local account (returns a token)
GET  /auth/me        — the signed-in user
POST /auth/google    — sign in with a Google ID token (needs GOOGLE_CLIENT_ID)
GET  /auth/guest     — how many free (signed-out) queries this client has left
"""
from __future__ import annotations

import logging
import re

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from api.auth import (
    _EXPIRE,
    _verify_password,
    find_user,
    get_current_user,
    get_optional_user,
    guest_key,
    guest_limit,
    guest_queries_used,
    hash_password,
    settings,
    token_for,
    user_public,
)
from api.database import get_db
from api.orm import User

log = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["Auth"])

_USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class LoginRequest(BaseModel):
    username: str = Field(..., description="Username or email")
    password: str


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=32)
    email: str = Field(..., max_length=255)
    password: str = Field(..., min_length=8, max_length=128)
    full_name: str | None = Field(None, max_length=120)

    @field_validator("username")
    @classmethod
    def _username(cls, v: str) -> str:
        v = v.strip()
        if not _USERNAME_RE.match(v):
            raise ValueError("Username may use letters, numbers, '.', '_' and '-' (3–32 characters).")
        return v

    @field_validator("email")
    @classmethod
    def _email(cls, v: str) -> str:
        v = v.strip().lower()
        if not _EMAIL_RE.match(v):
            raise ValueError("Enter a valid email address.")
        return v


class GoogleLoginRequest(BaseModel):
    credential: str = Field(..., description="Google ID token from Google Identity Services")


class UserOut(BaseModel):
    id: int
    username: str
    email: str | None = None
    full_name: str | None = None
    role: str
    auth_provider: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_minutes: int
    user: UserOut | None = None


class GuestStatus(BaseModel):
    signed_in: bool
    limit: int
    used: int
    remaining: int


def _token_response(user: User) -> TokenResponse:
    return TokenResponse(
        access_token=token_for(user),
        expires_in_minutes=_EXPIRE,
        user=UserOut(**user_public(user)),
    )


@router.post("/token", response_model=TokenResponse, summary="Sign in and obtain a JWT")
def login(req: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    """
    Exchange username (or email) + password for a JWT bearer token.

    **Seeded accounts:**
    - `testuser` (or `test@dailytruth.app`) / `Test@1234` — regular user
    - `admin` / `admin123`, `analyst` / `analyst123`

    Include the token in subsequent requests as `Authorization: Bearer <access_token>`.
    """
    user = find_user(db, req.username)
    if not user or not _verify_password(req.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    return _token_response(user)


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED,
             summary="Create an account")
def register(req: RegisterRequest, db: Session = Depends(get_db)) -> TokenResponse:
    if find_user(db, req.username):
        raise HTTPException(status.HTTP_409_CONFLICT, detail="That username is taken.")
    if find_user(db, req.email):
        raise HTTPException(status.HTTP_409_CONFLICT, detail="An account with that email already exists.")
    user = User(
        username=req.username,
        email=req.email,
        full_name=(req.full_name or "").strip() or None,
        hashed_password=hash_password(req.password),
        role="user",
        auth_provider="local",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    log.info("auth: registered %s", user.username)
    return _token_response(user)


@router.get("/me", response_model=UserOut, summary="The signed-in user")
def me(user: dict = Depends(get_current_user)) -> UserOut:
    return UserOut(**user)


@router.get("/guest", response_model=GuestStatus, summary="Free queries left without signing in")
def guest_status(
    request: Request,
    user: dict | None = Depends(get_optional_user),
    db: Session = Depends(get_db),
) -> GuestStatus:
    limit = guest_limit()
    if user:
        return GuestStatus(signed_in=True, limit=limit, used=0, remaining=limit)
    used = guest_queries_used(db, guest_key(request))
    return GuestStatus(signed_in=False, limit=limit, used=used, remaining=max(0, limit - used))


def _unique_username(db: Session, base: str) -> str:
    base = re.sub(r"[^A-Za-z0-9_.-]", "", base)[:24] or "user"
    if len(base) < 3:
        base = f"{base}user"
    candidate, n = base, 1
    while find_user(db, candidate):
        n += 1
        candidate = f"{base}{n}"
    return candidate


@router.post("/google", response_model=TokenResponse, summary="Sign in with Google")
def google_login(req: GoogleLoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    """
    Verify a Google ID token (from Google Identity Services on the frontend)
    and sign the user in, creating an account on first use.

    Requires `GOOGLE_CLIENT_ID` in `.env` and the `google-auth` package.
    Until then this returns **501 Not Implemented**.
    """
    client_id = getattr(settings, "GOOGLE_CLIENT_ID", None) if settings else None
    if not client_id:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Google sign-in is not configured yet. Set GOOGLE_CLIENT_ID on the server.",
        )
    try:
        from google.auth.transport import requests as g_requests
        from google.oauth2 import id_token as g_id_token
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Google sign-in needs the 'google-auth' package on the server.",
        )

    try:
        info = g_id_token.verify_oauth2_token(req.credential, g_requests.Request(), client_id)
    except ValueError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail=f"Invalid Google token: {exc}")

    sub = info.get("sub")
    email = (info.get("email") or "").lower() or None
    if not sub:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Google token has no subject.")
    if email and not info.get("email_verified", False):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Google email is not verified.")

    user = db.query(User).filter(User.google_sub == sub).one_or_none()
    if user is None and email:
        # Link to an existing local account with the same verified email.
        user = find_user(db, email)
        if user is not None:
            user.google_sub = sub
    if user is None:
        user = User(
            username=_unique_username(db, (email or "user").split("@")[0]),
            email=email,
            full_name=info.get("name"),
            hashed_password=None,
            role="user",
            auth_provider="google",
            google_sub=sub,
        )
        db.add(user)
    db.commit()
    db.refresh(user)
    return _token_response(user)
