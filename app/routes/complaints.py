"""Complaint submission with optional media attachment."""

from __future__ import annotations

import json
import os
import smtplib
import uuid
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from app.auth import get_current_user

router = APIRouter(prefix="/complaints", tags=["complaints"])

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "complaints"
ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".mp4", ".webm"}
ALLOWED_MIME = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "video/mp4",
    "video/webm",
}
MAX_BYTES = 10 * 1024 * 1024
DEFAULT_TO = "abdulrehmantahir142@gmail.com"


def _complaint_to() -> str:
    return os.getenv("COMPLAINT_TO_EMAIL", DEFAULT_TO).strip() or DEFAULT_TO


def _send_via_smtp(
    *,
    sender: str,
    to_email: str,
    subject: str,
    body: str,
    attachment_name: str | None,
    attachment_bytes: bytes | None,
    attachment_mime: str | None,
) -> bool:
    host = os.getenv("SMTP_HOST", "").strip()
    if not host:
        return False

    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER", "").strip()
    password = os.getenv("SMTP_PASSWORD", "").strip()
    from_addr = os.getenv("SMTP_FROM", user or sender).strip()
    use_tls = os.getenv("SMTP_TLS", "true").lower() in {"1", "true", "yes"}

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_email
    msg["Reply-To"] = sender
    msg.set_content(body)

    if attachment_name and attachment_bytes is not None:
        maintype, _, subtype = (attachment_mime or "application/octet-stream").partition(
            "/"
        )
        if not subtype:
            maintype, subtype = "application", "octet-stream"
        msg.add_attachment(
            attachment_bytes,
            maintype=maintype,
            subtype=subtype,
            filename=attachment_name,
        )

    with smtplib.SMTP(host, port, timeout=30) as smtp:
        if use_tls:
            smtp.starttls()
        if user and password:
            smtp.login(user, password)
        smtp.send_message(msg)
    return True


@router.post("")
async def submit_complaint(
    request: Request,
    subject: str = Form(...),
    complaint: str = Form(...),
    to_email: str | None = Form(None),
    attachment: UploadFile | None = File(None),
) -> dict[str, Any]:
    user = get_current_user(request)
    sender = str(user.get("email", "")).strip().lower()
    if not sender:
        raise HTTPException(status_code=401, detail="Not authenticated.")

    to_addr = _complaint_to()
    _ = to_email

    subj = subject.strip()
    body_text = complaint.strip()

    if not subj:
        raise HTTPException(status_code=400, detail="Subject is required.")
    if len(body_text) < 10:
        raise HTTPException(
            status_code=400,
            detail="Complaint should be at least 10 characters.",
        )

    file_bytes: bytes | None = None
    file_name: str | None = None
    file_mime: str | None = None

    if attachment is not None and attachment.filename:
        file_name = Path(attachment.filename).name
        ext = Path(file_name).suffix.lower()
        file_mime = (attachment.content_type or "").lower()
        if ext not in ALLOWED_EXT and file_mime not in ALLOWED_MIME:
            raise HTTPException(
                status_code=400,
                detail="Attachment must be PNG, JPG, MP4, or WEBM.",
            )
        file_bytes = await attachment.read()
        if len(file_bytes) > MAX_BYTES:
            raise HTTPException(
                status_code=400,
                detail="Attachment exceeds the 10MB limit.",
            )

    ticket_id = uuid.uuid4().hex[:12]
    stamp = datetime.now(timezone.utc).isoformat()
    folder = DATA_DIR / ticket_id
    folder.mkdir(parents=True, exist_ok=True)

    saved_attachment: str | None = None
    if file_bytes is not None and file_name:
        dest = folder / file_name
        dest.write_bytes(file_bytes)
        saved_attachment = str(dest.name)

    email_body = (
        "Medirator complaint submission\n"
        f"Ticket: {ticket_id}\n"
        f"Time (UTC): {stamp}\n"
        f"From: {sender}\n"
        f"Name: {user.get('name') or sender}\n"
        f"To: {to_addr}\n"
        f"Subject: {subj}\n\n"
        f"{body_text}\n"
    )

    meta = {
        "id": ticket_id,
        "created_at": stamp,
        "from_email": sender,
        "to_email": to_addr,
        "subject": subj,
        "complaint": body_text,
        "attachment": saved_attachment,
        "user_id": user.get("id"),
        "user_name": user.get("name"),
    }
    (folder / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    mailed = False
    try:
        mailed = _send_via_smtp(
            sender=sender,
            to_email=to_addr,
            subject=f"[Medirator] {subj}",
            body=email_body,
            attachment_name=file_name,
            attachment_bytes=file_bytes,
            attachment_mime=file_mime,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=502,
            detail=f"Failed to send email via SMTP: {exc}",
        ) from exc

    return {
        "ok": True,
        "id": ticket_id,
        "to": to_addr,
        "from_email": sender,
        "from_name": user.get("name") or sender,
        "subject": subj,
        "message_body": email_body,
        "delivered": "smtp" if mailed else "pending_client",
        "message": (
            "Complaint sent successfully"
            if mailed
            else "Complaint saved. Delivering email…"
        ),
    }
