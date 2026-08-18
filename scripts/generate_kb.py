#!/usr/bin/env python3
"""Regenerate synthetic KB. Prefer re-running the documented generator in README/ASSIGNMENT.md.

This wrapper reloads stats after you replace JSON files under backend/data/.
"""
from pathlib import Path
import json
DATA = Path(__file__).resolve().parents[1] / "data"
counts = {}
for name in ("patients", "medicines", "inventory", "instruments"):
    rows = json.loads((DATA / f"{name}.json").read_text())
    counts[name] = len(rows)
print("KB counts:", counts, "total:", sum(counts.values()))
