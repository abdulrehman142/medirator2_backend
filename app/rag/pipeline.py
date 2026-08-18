"""Medirator 2.0 RAG pipeline orchestration."""

from __future__ import annotations

from typing import Any

from app.rag.generator import generate
from app.rag.retriever import retrieve


def run_retrieve(query: str, category: str | None = None) -> dict[str, Any]:
    resolved, results, confidence = retrieve(query, category=category)
    return {
        "query": query,
        "category": resolved,
        "results": results,
        "confidence": confidence,
    }


def run_generate(
    query: str,
    category: str | None = None,
    context: list[dict] | None = None,
) -> dict[str, Any]:
    if context is None:
        resolved, results, _ = retrieve(query, category=category)
        context_records = [item["data"] for item in results]
        category = resolved
    else:
        resolved = category or "patients"
        context_records = context
        results = []

    answer, structured, model = generate(query, resolved, context_records)
    return {
        "query": query,
        "category": resolved,
        "answer": answer,
        "structured": structured,
        "model": model,
        "retrieved": results,
    }


def run_query(query: str, category: str | None = None) -> dict[str, Any]:
    retrieval = run_retrieve(query, category=category)
    answer, structured, model = generate(
        query,
        retrieval["category"],
        [item["data"] for item in retrieval["results"]],
    )
    return {
        "query": query,
        "category": retrieval["category"],
        "answer": answer,
        "structured": structured,
        "retrieved": retrieval["results"],
        "confidence": retrieval["confidence"],
        "model": model,
        "raw_context": [item["data"] for item in retrieval["results"]],
    }
