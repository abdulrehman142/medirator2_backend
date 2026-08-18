from fastapi import APIRouter, HTTPException

from app.rag.storage import CATEGORIES, get_category_records, load_knowledge_base

router = APIRouter(prefix="/data", tags=["data"])


@router.get("")
def list_all_data():
    return load_knowledge_base()


@router.get("/{category}")
def list_category(category: str, q: str | None = None):
    if category not in CATEGORIES:
        raise HTTPException(status_code=404, detail=f"Unknown category: {category}")

    records = get_category_records(category)
    if q:
        needle = q.lower().strip()
        filtered = []
        for record in records:
            blob = str(record).lower()
            if needle in blob:
                filtered.append(record)
        return {"category": category, "count": len(filtered), "items": filtered}

    return {"category": category, "count": len(records), "items": records}
