from fastapi import APIRouter

from app.models.schema import HealthResponse
from app.rag.generator import (
    active_provider,
    llm_configured,
    llm_model_name,
    llm_reachable,
)
from app.rag.storage import knowledge_base_stats

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health():
    configured = llm_configured()
    reachable = llm_reachable() if configured else False
    model = llm_model_name() if configured else None
    provider = active_provider()
    return HealthResponse(
        status="ok",
        llm_provider=provider,
        llm_configured=configured,
        llm_reachable=reachable,
        llm_model=model,
        # Backward-compatible aliases for older clients
        ollama_running=reachable,
        ollama_model=model,
        knowledge_base=knowledge_base_stats(),
    )
