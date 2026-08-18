from fastapi import APIRouter, HTTPException

from app.models.schema import (
    GenerateRequest,
    GenerateResponse,
    QueryRequest,
    QueryResponse,
    RetrieveResponse,
    RetrievedItem,
)
from app.rag.pipeline import run_generate, run_query, run_retrieve

router = APIRouter()


def _safe_query_response(result: dict) -> QueryResponse:
    retrieved = []
    for item in result.get("retrieved") or []:
        retrieved.append(
            RetrievedItem(
                category=str(item.get("category", "")),
                id=str(item.get("id", "")),
                score=float(item.get("score") or 0),
                data=item.get("data") if isinstance(item.get("data"), dict) else {},
            )
        )
    answer = result.get("answer")
    if not isinstance(answer, str):
        answer = str(answer or "")
    structured = result.get("structured")
    if structured is not None and not isinstance(structured, dict):
        structured = {"summary": answer}
    return QueryResponse(
        query=str(result.get("query", "")),
        category=str(result.get("category") or "patients"),
        answer=answer,
        structured=structured,
        retrieved=retrieved,
        confidence=float(result.get("confidence") or 0),
        model=str(result.get("model") or "unknown"),
        raw_context=[
            row for row in (result.get("raw_context") or []) if isinstance(row, dict)
        ],
    )


@router.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    q = request.query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        result = run_query(q, category=request.category)
        return _safe_query_response(result)
    except Exception as exc:
        # Never leave the chat UI with a bare 500 if retrieval still works
        try:
            retrieval = run_retrieve(q, category=request.category)
            record = (retrieval["results"][0]["data"] if retrieval["results"] else {})
            summary = (
                f"{record.get('name') or record.get('item') or 'Match found'} — "
                f"showing retrieved record (generator error: {exc.__class__.__name__})."
            )
            return _safe_query_response(
                {
                    "query": q,
                    "category": retrieval["category"],
                    "answer": summary,
                    "structured": {"summary": summary, **({} if not record else {})},
                    "retrieved": retrieval["results"],
                    "confidence": retrieval["confidence"],
                    "model": "fallback-exception",
                    "raw_context": [item["data"] for item in retrieval["results"]],
                }
            )
        except Exception as inner:
            raise HTTPException(
                status_code=500,
                detail=f"Query failed: {exc}; recovery failed: {inner}",
            ) from inner


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve_endpoint(request: QueryRequest):
    q = request.query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    result = run_retrieve(q, category=request.category)
    return RetrieveResponse(
        query=result["query"],
        category=result["category"],
        results=[RetrievedItem(**item) for item in result["results"]],
        confidence=result["confidence"],
    )


@router.post("/generate", response_model=GenerateResponse)
def generate_endpoint(request: GenerateRequest):
    q = request.query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    result = run_generate(q, category=request.category, context=request.context)
    answer = result["answer"]
    if not isinstance(answer, str):
        answer = str(answer or "")
    return GenerateResponse(
        query=result["query"],
        category=result["category"],
        answer=answer,
        structured=result["structured"]
        if isinstance(result.get("structured"), dict)
        else None,
        model=str(result["model"]),
    )
