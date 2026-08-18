from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str
    category: str | None = Field(
        default=None,
        description="Optional category: patients | medicines | inventory | instruments",
    )


class RetrievedItem(BaseModel):
    category: str
    id: str
    score: float
    data: dict


class RetrieveResponse(BaseModel):
    query: str
    category: str | None = None
    results: list[RetrievedItem]
    confidence: float


class GenerateRequest(BaseModel):
    query: str
    category: str | None = None
    context: list[dict] | None = None


class GenerateResponse(BaseModel):
    query: str
    category: str
    answer: str
    structured: dict | None = None
    model: str


class QueryResponse(BaseModel):
    query: str
    category: str
    answer: str
    structured: dict | None = None
    retrieved: list[RetrievedItem]
    confidence: float
    model: str
    raw_context: list[dict]


class HealthResponse(BaseModel):
    status: str
    llm_provider: str = "grok"
    llm_configured: bool = False
    llm_reachable: bool = False
    llm_model: str | None = None
    # Deprecated aliases (kept so older frontends keep working)
    ollama_running: bool = False
    ollama_model: str | None = None
    knowledge_base: dict
