# Medirator 2.0 — Backend

FastAPI keyword RAG API powered by **Grok (xAI)**.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Set GROK_API_KEY, GOOGLE_CLIENT_ID, JWT_SECRET
# For deploy: CORS_ORIGINS=https://your-frontend-url
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Env

| Variable | Purpose |
|----------|---------|
| `GROK_API_KEY` | xAI API key ([console.x.ai](https://console.x.ai/)) |
| `GROK_MODEL` | Default `grok-3-mini` |
| `GOOGLE_CLIENT_ID` | Google OAuth web client |
| `JWT_SECRET` | Session signing secret |
| `CORS_ORIGINS` | Comma-separated frontend URLs |

Frontend companion: https://github.com/abdulrehman142/medirator2_frontend
