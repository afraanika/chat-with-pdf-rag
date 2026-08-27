"""
Password hashing and signed-cookie sessions.

Uses bcrypt directly rather than passlib - passlib's bcrypt backend probes
`bcrypt.__about__.__version__`, which was removed in modern bcrypt releases,
so it's a needless source of warnings/breakage for no benefit here.

Sessions are a signed, timestamped token (itsdangerous) stored in an
HttpOnly cookie - no server-side session table needed. Signing prevents
tampering; there's no revocation list, so "logout" just clears the cookie
client-side and a stolen token remains valid until it expires.
"""

import os

import bcrypt
from fastapi import Depends, HTTPException, Request
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from sqlalchemy.orm import Session

from db import User, get_db_session

SESSION_SECRET = os.environ.get("SESSION_SECRET", "dev-insecure-secret-change-me")
SESSION_COOKIE_NAME = "session"
SESSION_MAX_AGE = 60 * 60 * 24 * 30  # 30 days
COOKIE_SECURE = os.environ.get("COOKIE_SECURE", "false").lower() == "true"

_serializer = URLSafeTimedSerializer(SESSION_SECRET, salt="chat-with-pdf-session")

# bcrypt silently ignores password bytes past 72 - truncate explicitly so
# hashing and verification agree on what was actually checked.
_MAX_PASSWORD_BYTES = 72


def hash_password(password: str) -> str:
    truncated = password.encode("utf-8")[:_MAX_PASSWORD_BYTES]
    return bcrypt.hashpw(truncated, bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    truncated = password.encode("utf-8")[:_MAX_PASSWORD_BYTES]
    return bcrypt.checkpw(truncated, hashed.encode("utf-8"))


def create_session_token(user_id: str) -> str:
    return _serializer.dumps({"user_id": user_id})


def read_session_token(token: str) -> str | None:
    try:
        data = _serializer.loads(token, max_age=SESSION_MAX_AGE)
    except (BadSignature, SignatureExpired):
        return None
    return data.get("user_id")


def get_current_user(request: Request, db: Session = Depends(get_db_session)) -> User:
    token = request.cookies.get(SESSION_COOKIE_NAME)
    user_id = read_session_token(token) if token else None
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated.")

    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    return user
