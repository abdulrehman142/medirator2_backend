from fastapi import APIRouter

from app.models.schema import HealthResponse
from app.rag.generator import GROK_MODEL, grok_configured, grok_reachable
from app.rag.storage import knowledge_base_stats

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health():
    configured = grok_configured()
    reachable = grok_reachable() if configured else False
    return HealthResponse(
        status="ok",
        llm_provider="grok",
        llm_configured=configured,
        llm_reachable=reachable,
        llm_model=GROK_MODEL if configured else None,
        # Backward-compatible aliases for older clients
        ollama_running=reachable,
        ollama_model=GROK_MODEL if configured else None,
        knowledge_base=knowledge_base_stats(),
    )
