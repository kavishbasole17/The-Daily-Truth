"""
JWT authentication utilities.

Users live in the `users` table (api/orm.py). Three accounts are seeded at
startup so the app is usable out of the box:

  admin    / admin123    — role admin
  analyst  / analyst123  — role analyst
  testuser / Test@1234   — role user  (email test@dailytruth.app)

Anonymous visitors get GUEST_QUERY_LIMIT free verification queries (default
1), counted per client IP in `guest_usage`; after that, submitting requires a
Bearer token.
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import UTC, datetime, timedelta

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from api.database import SessionLocal, get_db
from api.orm import GuestUsage, User

try:
    from config import settings
    _SECRET = settings.JWT_SECRET_KEY
    _ALGO   = settings.JWT_ALGORITHM
    _EXPIRE = settings.JWT_EXPIRE_MINUTES
except ImportError:  # pragma: no cover
    settings = None
    _SECRET = "dev-secret"
    _ALGO   = "HS256"
    _EXPIRE = 60

log = logging.getLogger(__name__)

_pwd = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
_bearer = HTTPBearer(auto_error=False)

# Accounts created on startup if missing. Passwords are hashed before storage.
SEED_USERS: list[dict] = [
    {"username": "admin",    "password": "admin123",   "role": "admin",
     "email": None, "full_name": "Administrator"},
    {"username": "analyst",  "password": "analyst123", "role": "analyst",
     "email": None, "full_name": "Analyst"},
]
TEST_USER: dict = {
    "username": "testuser", "password": "Test@1234", "role": "user",
    "email": "test@dailytruth.app", "full_name": "Test User",
}


# ── Passwords / tokens ──────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    return _pwd.hash(plain)


def _verify_password(plain: str, hashed: str | None) -> bool:
    if not hashed:
        return False  # e.g. a Google-only account has no password
    try:
        return _pwd.verify(plain, hashed)
    except (ValueError, TypeError):
        return False


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    payload = data.copy()
    payload["exp"] = datetime.now(UTC) + (
        expires_delta or timedelta(minutes=_EXPIRE)
    )
    return jwt.encode(payload, _SECRET, algorithm=_ALGO)


def token_for(user: User) -> str:
    return create_access_token({"sub": user.username, "uid": user.id, "role": user.role})


def user_public(user: User) -> dict:
    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "full_name": user.full_name,
        "role": user.role,
        "auth_provider": user.auth_provider,
    }


# ── Lookups ─────────────────────────────────────────────────────────────────────

def find_user(db: Session, identifier: str) -> User | None:
    """Look a user up by username or email (case-insensitive)."""
    ident = (identifier or "").strip().lower()
    if not ident:
        return None
    return db.execute(
        select(User).where(or_(func.lower(User.username) == ident,
                               func.lower(User.email) == ident))
    ).scalar_one_or_none()


def authenticate_user(username: str, password: str, db: Session | None = None) -> dict | None:
    """Return the public user dict on success, None otherwise."""
    own = db is None
    db = db or SessionLocal()
    try:
        user = find_user(db, username)
        if not user or not _verify_password(password, user.hashed_password):
            return None
        return user_public(user)
    finally:
        if own:
            db.close()


def seed_users() -> None:
    """Create the demo + test accounts if they don't exist yet. Idempotent."""
    seeds = list(SEED_USERS)
    if settings is None or getattr(settings, "SEED_TEST_USER", True):
        seeds.append(TEST_USER)
    db = SessionLocal()
    try:
        for s in seeds:
            if find_user(db, s["username"]):
                continue
            db.add(User(
                username=s["username"],
                email=s["email"],
                full_name=s["full_name"],
                hashed_password=hash_password(s["password"]),
                role=s["role"],
                auth_provider="local",
            ))
            log.info("auth: seeded user %s", s["username"])
        db.commit()
    finally:
        db.close()


# ── Dependencies ────────────────────────────────────────────────────────────────

def _decode(credentials: HTTPAuthorizationCredentials) -> dict:
    try:
        payload = jwt.decode(credentials.credentials, _SECRET, algorithms=[_ALGO])
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )
    username = payload.get("sub", "")
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token: missing sub claim",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return payload


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: Session = Depends(get_db),
) -> dict:
    """FastAPI dependency — raises 401 if token is missing or invalid."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated — supply a Bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = _decode(credentials)
    user = find_user(db, payload["sub"])
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Account no longer exists",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user_public(user)


def get_optional_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: Session = Depends(get_db),
) -> dict | None:
    """Like get_current_user, but anonymous callers get None instead of a 401.
    A token that is present but invalid is still rejected."""
    if credentials is None:
        return None
    return get_current_user(credentials, db)


# ── Guest allowance ─────────────────────────────────────────────────────────────

GUEST_LIMIT_CODE = "guest_limit_reached"


def guest_key(request: Request) -> str:
    """Identify an anonymous client. Honours X-Forwarded-For when behind a proxy."""
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()[:128]
    return (request.client.host if request.client else "unknown")[:128]


def guest_limit() -> int:
    return int(getattr(settings, "GUEST_QUERY_LIMIT", 1)) if settings else 1


def guest_queries_used(db: Session, key: str) -> int:
    row = db.execute(select(GuestUsage).where(GuestUsage.client_key == key)).scalar_one_or_none()
    return row.query_count if row else 0


def consume_guest_query(db: Session, key: str) -> None:
    """Spend one of the guest's free queries, or raise 401 if none are left."""
    row = db.execute(select(GuestUsage).where(GuestUsage.client_key == key)).scalar_one_or_none()
    used = row.query_count if row else 0
    if used >= guest_limit():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": GUEST_LIMIT_CODE,
                "message": "You've used your free check. Sign in to keep verifying "
                           "and to save your chat history.",
            },
            headers={"WWW-Authenticate": "Bearer"},
        )
    if row is None:
        db.add(GuestUsage(client_key=key, query_count=1))
    else:
        row.query_count = used + 1
    db.commit()
