# Backend/auth.py
"""
JWT-based authentication.
Usage:
  - POST /auth/token  with {"api_key": "your-key"} → returns JWT access token
  - Protected routes: add Depends(verify_token) in the endpoint signature
"""
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from Backend.config import JWT_SECRET_KEY, JWT_ALGORITHM, JWT_EXPIRE_MINUTES

logger = logging.getLogger(__name__)

# ── Simple static API key store (replace with DB in production) ───────────────
# Keys are loaded from env: RAG_API_KEYS=key1,key2,key3
_VALID_API_KEYS: set = set(
    k.strip()
    for k in os.getenv("RAG_API_KEYS", "dev-key-change-me").split(",")
    if k.strip()
)

security = HTTPBearer(auto_error=False)


# ─────────────────────────────────────────────────────────────────────────────
# Token models
# ─────────────────────────────────────────────────────────────────────────────

class TokenRequest(BaseModel):
    api_key: str

class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    expires_in:   int  # seconds


# ─────────────────────────────────────────────────────────────────────────────
# Token creation
# ─────────────────────────────────────────────────────────────────────────────

def create_access_token(subject: str, expires_delta: Optional[timedelta] = None) -> str:
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=JWT_EXPIRE_MINUTES)
    )
    payload = {"sub": subject, "exp": expire}
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def verify_api_key(api_key: str) -> bool:
    return api_key in _VALID_API_KEYS


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI dependency — use on protected routes
# ─────────────────────────────────────────────────────────────────────────────

def verify_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> str:
    """
    FastAPI dependency that validates the Bearer JWT token.
    Returns the token subject (e.g. "api_user") on success.
    Raises HTTP 401 on failure.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = jwt.decode(
            credentials.credentials,
            JWT_SECRET_KEY,
            algorithms=[JWT_ALGORITHM],
        )
        subject: str = payload.get("sub")
        if not subject:
            raise JWTError("Missing subject")
        return subject
    except JWTError as e:
        logger.warning("Token verification failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
