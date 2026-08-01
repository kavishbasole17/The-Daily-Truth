"""POST /auth/token — obtain a JWT bearer token."""
from __future__ import annotations

from datetime import timedelta

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from api.auth import authenticate_user, create_access_token, _EXPIRE

router = APIRouter(prefix="/auth", tags=["Auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_minutes: int


@router.post("/token", response_model=TokenResponse, summary="Obtain a JWT access token")
def login(req: LoginRequest) -> TokenResponse:
    """
    Exchange username + password for a JWT bearer token.

    **Demo credentials** (change in production):
    - `admin` / `admin123`  — full access
    - `analyst` / `analyst123`  — can resolve review-queue items

    Include the token in subsequent requests:
    ```
    Authorization: Bearer <access_token>
    ```
    """
    user = authenticate_user(req.username, req.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    token = create_access_token(
        {"sub": user["username"], "role": user["role"]},
        expires_delta=timedelta(minutes=_EXPIRE),
    )
    return TokenResponse(access_token=token, expires_in_minutes=_EXPIRE)
