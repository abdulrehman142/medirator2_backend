"""Google ID token verification, JWT sessions, and local user store."""

from __future__ import annotations

import json
import os
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

import httpx
import jwt
from fastapi import HTTPException, Request, Response
from pydantic import BaseModel, EmailStr

Role = Literal["user"]

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
USERS_PATH = DATA_DIR / "users.json"

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "").strip()
JWT_SECRET = os.getenv("JWT_SECRET", secrets.token_hex(32))
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = int(os.getenv("JWT_EXPIRE_HOURS", "24"))
COOKIE_NAME = "medirator_token"


class GoogleAuthRequest(BaseModel):
    credential: str


class UserPublic(BaseModel):
    id: str
    email: EmailStr
    name: str
    picture: str | None = None
    role: Role


class AuthResponse(BaseModel):
    token: str
    user: UserPublic


def _load_users() -> dict[str, dict[str, Any]]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not USERS_PATH.exists():
        return {}
    with USERS_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def _save_users(users: dict[str, dict[str, Any]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with USERS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(users, handle, indent=2)


def assign_role(_email: str) -> Role:
    return "user"


def verify_google_token(credential: str) -> dict[str, Any]:
    """Verify Google ID token via Google's tokeninfo endpoint."""
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(
            status_code=503,
            detail=(
                "GOOGLE_CLIENT_ID is not configured. "
                "Set it in backend/.env and VITE_GOOGLE_CLIENT_ID in frontend/.env"
            ),
        )

    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(
                "https://oauth2.googleapis.com/tokeninfo",
                params={"id_token": credential},
            )
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to reach Google token verification: {exc}",
        ) from exc

    if response.status_code != 200:
        raise HTTPException(status_code=401, detail="Invalid Google credential.")

    payload = response.json()
    aud = payload.get("aud")
    if aud != GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=401, detail="Google token audience mismatch.")

    if payload.get("email_verified") not in (True, "true"):
        raise HTTPException(status_code=401, detail="Google email is not verified.")

    email = payload.get("email")
    if not email:
        raise HTTPException(status_code=401, detail="Google token missing email.")

    return payload


def upsert_user_from_google(payload: dict[str, Any]) -> dict[str, Any]:
    email = str(payload["email"]).lower()
    users = _load_users()
    existing = users.get(email)

    if existing:
        existing["name"] = payload.get("name") or existing.get("name") or email
        existing["picture"] = payload.get("picture") or existing.get("picture")
        existing["role"] = "user"
        existing["last_login"] = datetime.now(timezone.utc).isoformat()
        users[email] = existing
        _save_users(users)
        return existing

    user = {
        "id": payload.get("sub") or secrets.token_hex(8),
        "email": email,
        "name": payload.get("name") or email.split("@")[0],
        "picture": payload.get("picture"),
        "role": assign_role(email),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "last_login": datetime.now(timezone.utc).isoformat(),
    }
    users[email] = user
    _save_users(users)
    return user


def create_access_token(user: dict[str, Any]) -> str:
    expire = datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS)
    payload = {
        "sub": user["id"],
        "email": user["email"],
        "role": user["role"],
        "name": user["name"],
        "exp": expire,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict[str, Any]:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(status_code=401, detail="Token expired.") from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(status_code=401, detail="Invalid token.") from exc


def set_auth_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        httponly=True,
        secure=False,
        samesite="lax",
        max_age=JWT_EXPIRE_HOURS * 3600,
        path="/",
    )


def clear_auth_cookie(response: Response) -> None:
    response.delete_cookie(COOKIE_NAME, path="/")


def get_token_from_request(request: Request) -> str | None:
    cookie = request.cookies.get(COOKIE_NAME)
    if cookie:
        return cookie
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:].strip()
    return None


def get_current_user(request: Request) -> dict[str, Any]:
    token = get_token_from_request(request)
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated.")

    claims = decode_access_token(token)
    email = str(claims.get("email", "")).lower()
    users = _load_users()
    user = users.get(email)
    if not user:
        raise HTTPException(status_code=401, detail="User not found.")
    return user


def to_public(user: dict[str, Any]) -> UserPublic:
    return UserPublic(
        id=str(user["id"]),
        email=user["email"],
        name=user.get("name") or user["email"],
        picture=user.get("picture"),
        role=user.get("role", "user"),
    )
