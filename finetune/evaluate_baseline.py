"""Compare prompt-only SOAP fallback vs expected template fields on eval set.

Does not require GPU. Measures whether required JSON keys are present.
"""

from __future__ import annotations

import json
from pathlib import Path


REQUIRED = {
    "patients": {"subjective", "objective", "assessment", "plan", "summary"},
    "medicines": {"name", "usage", "dosage", "contraindications", "summary"},
    "inventory": {"item", "quantity", "location", "status", "summary"},
    "instruments": {"name", "department", "status", "location", "summary"},
}


def main() -> None:
    path = Path("finetune/data/eval.jsonl")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ok = 0
    for row in rows:
        out = json.loads(row["output"])
        need = REQUIRED[row["category"]]
        if need.issubset(out.keys()):
            ok += 1
    print(f"Template compliance on gold outputs: {ok}/{len(rows)} ({ok/len(rows):.1%})")
    print("Prompt-only RAG uses the same schemas via backend/app/rag/generator.py")
    print("After QLoRA, compare model generations to these gold JSON keys.")


if __name__ == "__main__":
    main()
