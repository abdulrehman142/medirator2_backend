from dotenv import load_dotenv
import os

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import auth, complaints, data, health, query

app = FastAPI(
    title="Medirator 2.0 API",
    description="AI Hospital Knowledge Assistant — keyword RAG + Gemini/Grok",
    version="2.0.0",
)

_default_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://medirator2.netlify.app",
]
_extra = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "").split(",")
    if origin.strip()
]
allow_origins = list(dict.fromkeys([*_default_origins, *_extra]))

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(query.router)
app.include_router(health.router)
app.include_router(data.router)
app.include_router(complaints.router)


@app.get("/")
def root():
    return {
        "name": "Medirator 2.0",
        "message": "AI Hospital Knowledge Assistant API",
        "llm": "gemini",
        "endpoints": {
            "auth_google": "POST /auth/google",
            "auth_me": "GET /auth/me",
            "query": "POST /query",
            "retrieve": "POST /retrieve",
            "generate": "POST /generate",
            "data": "GET /data",
            "health": "GET /health",
            "complaints": "POST /complaints",
        },
    }
