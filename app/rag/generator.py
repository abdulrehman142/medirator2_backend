"""LLM generation (Gemini / Grok) with structured clinical prompts."""

from __future__ import annotations

import json
import os
import re
from typing import Any

import httpx
from fastapi import HTTPException

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").strip().lower() or "gemini"

# Gemini (Google AI Studio) — https://aistudio.google.com/apikey
GEMINI_API_KEY = (
    os.getenv("GEMINI_API_KEY", "").strip()
    or os.getenv("GOOGLE_API_KEY", "").strip()
)
GEMINI_BASE_URL = os.getenv(
    "GEMINI_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta",
).rstrip("/")
GEMINI_MODEL = (
    os.getenv("GEMINI_MODEL", "gemini-1.5-pro").strip() or "gemini-1.5-pro"
)

# Grok (xAI) — optional
GROK_API_KEY = (
    os.getenv("GROK_API_KEY", "").strip()
    or os.getenv("XAI_API_KEY", "").strip()
)
GROK_BASE_URL = os.getenv("GROK_BASE_URL", "https://api.x.ai/v1").rstrip("/")
GROK_MODEL = os.getenv("GROK_MODEL", "grok-3-mini").strip() or "grok-3-mini"

SYSTEM_INSTRUCTION = (
    "You are Medirator 2.0. Reply with valid JSON only. "
    "Use only provided context. No external knowledge."
)


def active_provider() -> str:
    if LLM_PROVIDER in {"gemini", "google"}:
        return "gemini"
    if LLM_PROVIDER in {"grok", "xai"}:
        return "grok"
    # Auto: prefer whatever key is present
    if GEMINI_API_KEY:
        return "gemini"
    if GROK_API_KEY:
        return "grok"
    return LLM_PROVIDER


def llm_configured() -> bool:
    provider = active_provider()
    if provider == "gemini":
        return bool(GEMINI_API_KEY)
    if provider == "grok":
        return bool(GROK_API_KEY)
    return bool(GEMINI_API_KEY or GROK_API_KEY)


def llm_model_name() -> str:
    return GEMINI_MODEL if active_provider() == "gemini" else GROK_MODEL


def llm_reachable() -> bool:
    """Lightweight check that the configured provider accepts the API key."""
    provider = active_provider()
    try:
        with httpx.Client(timeout=10.0) as client:
            if provider == "gemini":
                if not GEMINI_API_KEY:
                    return False
                response = client.get(
                    f"{GEMINI_BASE_URL}/models/{GEMINI_MODEL}",
                    params={"key": GEMINI_API_KEY},
                )
                return response.status_code == 200
            if not GROK_API_KEY:
                return False
            response = client.get(
                f"{GROK_BASE_URL}/models",
                headers={"Authorization": f"Bearer {GROK_API_KEY}"},
            )
            return response.status_code == 200
    except httpx.HTTPError:
        return False


# Back-compat aliases used by older health imports
def grok_configured() -> bool:
    return llm_configured()


def grok_reachable() -> bool:
    return llm_reachable()


def _prompt_for_category(category: str, query: str, context: list[dict]) -> str:
    context_json = json.dumps(context, indent=2)

    if category == "patients":
        return f"""You are Medirator 2.0, a hospital knowledge assistant.
Using ONLY the patient context below, answer the clinician query in SOAP format.

Return STRICT JSON with this shape:
{{
  "subjective": "...",
  "objective": "...",
  "assessment": "...",
  "plan": "...",
  "summary": "one short paragraph"
}}

Context:
{context_json}

Query:
{query}
"""

    if category == "medicines":
        return f"""You are Medirator 2.0, a hospital knowledge assistant.
Using ONLY the medicine context below, answer in structured medicine card format.

Return STRICT JSON with this shape:
{{
  "name": "...",
  "usage": "...",
  "dosage": "...",
  "contraindications": ["..."],
  "summary": "one short paragraph"
}}

Context:
{context_json}

Query:
{query}
"""

    if category == "inventory":
        return f"""You are Medirator 2.0, a hospital knowledge assistant.
Using ONLY the inventory context below, answer as a stock panel.

Return STRICT JSON with this shape:
{{
  "item": "...",
  "quantity": "...",
  "location": "...",
  "status": "...",
  "summary": "one short paragraph"
}}

Context:
{context_json}

Query:
{query}
"""

    return f"""You are Medirator 2.0, a hospital knowledge assistant.
Using ONLY the instrument/equipment context below, answer clearly.

Return STRICT JSON with this shape:
{{
  "name": "...",
  "department": "...",
  "status": "...",
  "location": "...",
  "summary": "one short paragraph"
}}

Context:
{context_json}

Query:
{query}
"""


def _fallback_structured(category: str, context: list[dict]) -> dict[str, Any]:
    if not context:
        return {"summary": "No matching records found in the local knowledge base."}

    record = context[0]
    if category == "patients":
        vitals = record.get("vitals", {})
        return {
            "subjective": record.get("chief_complaint", ""),
            "objective": (
                f"Vitals BP {vitals.get('bp')}, HR {vitals.get('hr')}, "
                f"RR {vitals.get('rr')}, Temp {vitals.get('temp_c')}C, SpO2 {vitals.get('spo2')}%. "
                f"Labs: {', '.join(record.get('labs', []))}."
            ),
            "assessment": record.get("assessment", ""),
            "plan": record.get("plan", ""),
            "summary": (
                f"{record.get('name')} ({record.get('id')}) — "
                f"{record.get('assessment', 'See chart')}."
            ),
        }

    if category == "medicines":
        return {
            "name": record.get("name", ""),
            "usage": record.get("usage", ""),
            "dosage": record.get("dosage", ""),
            "contraindications": record.get("contraindications", []),
            "summary": f"{record.get('name')}: {record.get('usage', '')}",
        }

    if category == "inventory":
        return {
            "item": record.get("item", ""),
            "quantity": f"{record.get('quantity')} {record.get('unit', '')}".strip(),
            "location": record.get("location", ""),
            "status": record.get("status", ""),
            "summary": (
                f"{record.get('item')} — {record.get('status')} "
                f"({record.get('quantity')} {record.get('unit', '')})."
            ),
        }

    return {
        "name": record.get("name", ""),
        "department": record.get("department", ""),
        "status": record.get("status", ""),
        "location": record.get("location", ""),
        "summary": (
            f"{record.get('name')} in {record.get('department')} — "
            f"{record.get('status')}."
        ),
    }


def _extract_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    # Strip markdown fences if the model wraps JSON
    fence = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", text, re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _call_gemini(prompt: str) -> tuple[str, str]:
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY is not configured. Set it in backend/.env",
        )

    url = f"{GEMINI_BASE_URL}/models/{GEMINI_MODEL}:generateContent"
    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            url,
            params={"key": GEMINI_API_KEY},
            headers={"Content-Type": "application/json"},
            json={
                "systemInstruction": {
                    "parts": [{"text": SYSTEM_INSTRUCTION}],
                },
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": prompt}],
                    }
                ],
                "generationConfig": {
                    "temperature": 0.2,
                    "responseMimeType": "application/json",
                },
            },
        )
        if response.status_code >= 400:
            return "", f"gemini-error-{response.status_code}:{response.text[:240]}"
        response.raise_for_status()
        payload = response.json()

    candidates = payload.get("candidates") or []
    parts = (
        ((candidates[0].get("content") or {}).get("parts") or [])
        if candidates
        else []
    )
    text = ""
    for part in parts:
        if isinstance(part, dict) and part.get("text"):
            text += str(part["text"])
    return text.strip(), GEMINI_MODEL


def _call_grok(prompt: str) -> tuple[str, str]:
    if not GROK_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="GROK_API_KEY is not configured. Set it in backend/.env",
        )

    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            f"{GROK_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {GROK_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROK_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_INSTRUCTION},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "temperature": 0.2,
            },
        )
        if response.status_code >= 400:
            return "", f"grok-error-{response.status_code}:{response.text[:240]}"
        response.raise_for_status()
        payload = response.json()

    choices = payload.get("choices") or []
    content = ""
    if choices:
        content = (choices[0].get("message") or {}).get("content") or ""
    return content.strip(), GROK_MODEL


def generate(
    query: str,
    category: str,
    context_records: list[dict[str, Any]],
) -> tuple[str, dict[str, Any] | None, str]:
    """Generate answer via Gemini or Grok; fall back to deterministic structured output."""
    fallback = _fallback_structured(category, context_records)

    if not context_records:
        return fallback["summary"], fallback, "fallback"

    if not llm_configured():
        provider = active_provider()
        key_name = "GEMINI_API_KEY" if provider == "gemini" else "GROK_API_KEY"
        return (
            fallback.get(
                "summary",
                f"{key_name} not configured; showing retrieved record.",
            ),
            fallback,
            "fallback-no-key",
        )

    prompt = _prompt_for_category(category, query, context_records)
    provider = active_provider()

    try:
        if provider == "gemini":
            content, model = _call_gemini(prompt)
        else:
            content, model = _call_grok(prompt)
    except HTTPException:
        return fallback.get("summary", ""), fallback, "fallback-no-model"
    except httpx.HTTPError as exc:
        summary = fallback.get("summary", "")
        return (
            f"{summary}\n\n(Note: LLM request failed: {exc.__class__.__name__})",
            fallback,
            "fallback-error",
        )

    if not content and model.startswith(("gemini-error-", "grok-error-")):
        summary = fallback.get("summary", "")
        code = model.split(":", 1)[0]
        return (
            f"{summary}\n\n(Note: {provider} API failed ({code}). "
            f"Check API key / model / credits.)",
            fallback,
            code,
        )

    structured = _extract_json(content) or fallback
    answer = structured.get("summary") or content or fallback.get("summary", "")
    return answer, structured, model
