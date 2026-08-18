"""JSON knowledge-base loader for Medirator 2.0."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

CATEGORIES = ("patients", "medicines", "inventory", "instruments")


@lru_cache(maxsize=1)
def load_knowledge_base() -> dict[str, list[dict]]:
    kb: dict[str, list[dict]] = {}
    for category in CATEGORIES:
        path = DATA_DIR / f"{category}.json"
        if not path.exists():
            kb[category] = []
            continue
        with path.open(encoding="utf-8") as handle:
            kb[category] = json.load(handle)
    return kb


def reload_knowledge_base() -> dict[str, list[dict]]:
    load_knowledge_base.cache_clear()
    return load_knowledge_base()


def get_category_records(category: str) -> list[dict]:
    kb = load_knowledge_base()
    return kb.get(category, [])


def knowledge_base_stats() -> dict:
    kb = load_knowledge_base()
    counts = {name: len(records) for name, records in kb.items()}
    return {
        "ready": any(count > 0 for count in counts.values()),
        "counts": counts,
        "total_records": sum(counts.values()),
        "source": "local synthetic JSON knowledge base (Houston demo facility)",
        "synthetic": True,
        "pii_masked": True,
    }
