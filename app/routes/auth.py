from fastapi import APIRouter, Request, Response

from app.auth import (
    AuthResponse,
    GoogleAuthRequest,
    UserPublic,
    clear_auth_cookie,
    create_access_token,
    get_current_user,
    set_auth_cookie,
    to_public,
    upsert_user_from_google,
    verify_google_token,
)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/google", response_model=AuthResponse)
def google_auth(body: GoogleAuthRequest, response: Response):
    payload = verify_google_token(body.credential)
    user = upsert_user_from_google(payload)
    token = create_access_token(user)
    set_auth_cookie(response, token)
    return AuthResponse(token=token, user=to_public(user))


@router.get("/me", response_model=UserPublic)
def me(request: Request):
    return to_public(get_current_user(request))


@router.post("/logout")
def logout(response: Response):
    clear_auth_cookie(response)
    return {"ok": True}
