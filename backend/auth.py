"""
FinSight AI - Authentication Utilities
========================================
Password hashing (bcrypt), JWT token management, and FastAPI dependency
for extracting the current user from the Authorization header.

Author: FinSight AI Team
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
from jose import jwt, JWTError, ExpiredSignatureError
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from authlib.integrations.starlette_client import OAuth

from config import settings

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

JWT_SECRET = settings.JWT_SECRET
# Anyone holding the secret can mint a token for any account, so refuse to run
# on an empty one or the placeholder that used to be committed as the default.
if len(JWT_SECRET) < 32 or JWT_SECRET == "change-me-to-a-strong-random-secret":
    raise RuntimeError(
        "JWT_SECRET is unset or weak (need 32+ chars). Generate one with: "
        'python -c "import secrets; print(secrets.token_urlsafe(48))"'
    )
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 24

# FastAPI security scheme — expects "Authorization: Bearer <token>"
_bearer_scheme = HTTPBearer()

# ---------------------------------------------------------------------------
# Google OAuth (Authlib)
# ---------------------------------------------------------------------------

oauth = OAuth()
oauth.register(
    name="google",
    client_id=settings.GOOGLE_CLIENT_ID,
    client_secret=settings.GOOGLE_CLIENT_SECRET,
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)


# ---------------------------------------------------------------------------
# Password Helpers
# ---------------------------------------------------------------------------

def hash_password(plain_password: str) -> str:
    """Hash a plaintext password with bcrypt."""
    return bcrypt.hashpw(
        plain_password.encode("utf-8"),
        bcrypt.gensalt(),
    ).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Check a plaintext password against a bcrypt hash."""
    # OAuth-only accounts have no password_hash; treat as a failed login,
    # not an AttributeError on .encode().
    if not hashed_password:
        return False
    return bcrypt.checkpw(
        plain_password.encode("utf-8"),
        hashed_password.encode("utf-8"),
    )


# ---------------------------------------------------------------------------
# JWT Helpers
# ---------------------------------------------------------------------------

def create_access_token(user_id: str, email: str, name: str = "") -> str:
    """
    Create a signed JWT token.

    Payload:
        sub   — user_id (canonical identifier for authorization)
        email — for convenience (display, not for authz decisions)
        name  — for convenience
        exp   — expiration timestamp (24h from now)
    """
    payload = {
        "sub": user_id,
        "email": email,
        "name": name,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRY_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict:
    """
    Decode and validate a JWT token.

    Returns the payload dict on success.
    Raises HTTPException(401) on any failure.
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id: Optional[str] = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing subject",
            )
        return payload
    except ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired. Please log in again.",
        )
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token.",
        )


# ---------------------------------------------------------------------------
# FastAPI Dependency
# ---------------------------------------------------------------------------

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
) -> dict:
    """
    FastAPI dependency that extracts and validates the JWT from the
    Authorization header.

    Returns:
        {"user_id": str, "email": str, "name": str}
    """
    payload = decode_access_token(credentials.credentials)
    return {
        "user_id": payload["sub"],
        "email": payload.get("email", ""),
        "name": payload.get("name", ""),
    }
