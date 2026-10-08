"""
JWT authentication utilities.

Demo users are hard-coded for development.
In production: replace DEMO_USERS with a DB lookup.

Roles:
  admin   — full access
  analyst — same access as admin (kept for existing demo logins)
  public  — read-only (no auth required for GET endpoints)
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from config import settings
    _SECRET = settings.JWT_SECRET_KEY
    _ALGO   = settings.JWT_ALGORITHM
    _EXPIRE = settings.JWT_EXPIRE_MINUTES
except ImportError:
    _SECRET = "dev-secret"
    _ALGO   = "HS256"
    _EXPIRE = 60

_pwd = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
_bearer = HTTPBearer(auto_error=False)

# Demo in-memory user store
DEMO_USERS: dict[str, dict] = {
    "admin": {
        "hashed_password": _pwd.hash("admin123"),
        "role": "admin",
    },
    "analyst": {
        "hashed_password": _pwd.hash("analyst123"),
        "role": "analyst",
    },
}


def _verify_password(plain: str, hashed: str) -> bool:
    return _pwd.verify(plain, hashed)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    payload = data.copy()
    payload["exp"] = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=_EXPIRE)
    )
    return jwt.encode(payload, _SECRET, algorithm=_ALGO)


def authenticate_user(username: str, password: str) -> Optional[dict]:
    user = DEMO_USERS.get(username)
    if not user or not _verify_password(password, user["hashed_password"]):
        return None
    return {"username": username, "role": user["role"]}


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> dict:
    """FastAPI dependency — raises 401 if token is missing or invalid."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated — supply a Bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = jwt.decode(credentials.credentials, _SECRET, algorithms=[_ALGO])
        username: str = payload.get("sub", "")
        role: str = payload.get("role", "public")
        if not username:
            raise JWTError("Missing sub claim")
        return {"username": username, "role": role}
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )
