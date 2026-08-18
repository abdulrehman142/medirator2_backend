"""Keyword-based retriever (no vector DB)."""

from __future__ import annotations

import re
from typing import Any

from app.rag.storage import CATEGORIES, get_category_records, load_knowledge_base

STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "to",
    "for",
    "in",
    "on",
    "with",
    "is",
    "are",
    "was",
    "were",
    "be",
    "what",
    "who",
    "where",
    "when",
    "how",
    "about",
    "tell",
    "me",
    "show",
    "please",
    "info",
    "information",
    "status",
    "details",
}


CATEGORY_HINTS: dict[str, set[str]] = {
    "patients": {
        "patient",
        "patients",
        "soap",
        "admission",
        "ward",
        "complaint",
        "vitals",
        "diagnosis",
        "case",
    },
    "medicines": {
        "medicine",
        "medicines",
        "drug",
        "drugs",
        "medication",
        "dose",
        "dosage",
        "contraindication",
        "pharmacy",
        "pill",
    },
    "inventory": {
        "inventory",
        "stock",
        "supply",
        "supplies",
        "quantity",
        "reorder",
        "warehouse",
        "ppe",
    },
    "instruments": {
        "instrument",
        "instruments",
        "equipment",
        "device",
        "scanner",
        "machine",
        "tool",
    },
}


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def detect_category(query: str, explicit: str | None = None) -> str:
    if explicit and explicit in CATEGORIES:
        return explicit

    tokens = set(tokenize(query))
    scores = {
        category: len(tokens & hints) for category, hints in CATEGORY_HINTS.items()
    }
    best = max(scores, key=scores.get)
    if scores[best] > 0:
        return best

    # Fallback heuristics from common names / domains
    joined = " ".join(tokens)
    if any(word in joined for word in ("stock", "mask", "glove", "saline", "syringe")):
        return "inventory"
    if any(word in joined for word in ("ultrasound", "defibrillator", "pump", "ecg machine")):
        return "instruments"
    if any(
        word in joined
        for word in (
            "metformin",
            "lisinopril",
            "warfarin",
            "azithromycin",
            "albuterol",
            "aspirin",
            "acetaminophen",
        )
    ):
        return "medicines"
    return "patients"


def _flatten_record(record: dict[str, Any]) -> str:
    parts: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for nested in value.values():
                walk(nested)
        elif isinstance(value, list):
            for nested in value:
                walk(nested)
        else:
            parts.append(str(value))

    walk(record)
    keywords = record.get("keywords") or []
    parts.extend(str(k) for k in keywords)
    return " ".join(parts).lower()


def score_record(query_tokens: list[str], record: dict[str, Any]) -> float:
    if not query_tokens:
        return 0.0

    haystack = _flatten_record(record)
    haystack_tokens = set(tokenize(haystack))
    keywords = {str(k).lower() for k in record.get("keywords", [])}

    hits = 0.0
    for token in query_tokens:
        if token in keywords:
            hits += 2.5
        elif token in haystack_tokens:
            hits += 1.0
        elif token in haystack:
            hits += 0.5

    # Soft boost for multi-token coverage
    coverage = hits / (len(query_tokens) * 2.5)
    return min(hits, 20.0) + coverage


def retrieve(
    query: str,
    category: str | None = None,
    top_k: int = 3,
) -> tuple[str, list[dict[str, Any]], float]:
    """Return (resolved_category, ranked results, confidence)."""
    resolved = detect_category(query, category)
    query_tokens = tokenize(query)

    records = get_category_records(resolved)
    scored: list[tuple[float, dict[str, Any]]] = []
    for record in records:
        score = score_record(query_tokens, record)
        if score > 0:
            scored.append((score, record))

    # If nothing matched in detected category, search all categories
    if not scored:
        for cat in CATEGORIES:
            for record in get_category_records(cat):
                score = score_record(query_tokens, record)
                if score > 0:
                    scored.append((score, {**record, "_category": cat}))
        if scored:
            scored.sort(key=lambda pair: pair[0], reverse=True)
            top = scored[:top_k]
            top_cat = top[0][1].get("_category", resolved)
            results = []
            for score, record in top:
                cat = record.pop("_category", top_cat)
                results.append(
                    {
                        "category": cat,
                        "id": str(record.get("id", "")),
                        "score": round(score, 3),
                        "data": record,
                    }
                )
            confidence = min(1.0, top[0][0] / 8.0)
            return top_cat, results, round(confidence, 3)

    scored.sort(key=lambda pair: pair[0], reverse=True)
    top = scored[:top_k]
    results = [
        {
            "category": resolved,
            "id": str(record.get("id", "")),
            "score": round(score, 3),
            "data": record,
        }
        for score, record in top
    ]
    confidence = round(min(1.0, (top[0][0] / 8.0) if top else 0.0), 3)
    return resolved, results, confidence


def retrieve_all_preview() -> dict[str, list[dict]]:
    return load_knowledge_base()
