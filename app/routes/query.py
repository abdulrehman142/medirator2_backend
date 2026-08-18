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


@router.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    q = request.query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        result = run_query(q, category=request.category)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc

    return QueryResponse(
        query=result["query"],
        category=result["category"],
        answer=result["answer"],
        structured=result["structured"],
        retrieved=[RetrievedItem(**item) for item in result["retrieved"]],
        confidence=result["confidence"],
        model=result["model"],
        raw_context=result["raw_context"],
    )


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
    return GenerateResponse(
        query=result["query"],
        category=result["category"],
        answer=result["answer"],
        structured=result["structured"],
        model=result["model"],
    )
