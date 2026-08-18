# Medirator 2.0 — Backend

FastAPI keyword RAG API powered by **Gemini** (default) or optional Grok.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Set GEMINI_API_KEY, GOOGLE_CLIENT_ID, JWT_SECRET
# For deploy: CORS_ORIGINS=https://medirator2.netlify.app
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Env

| Variable | Purpose |
|----------|---------|
| `LLM_PROVIDER` | `gemini` (default) or `grok` |
| `GEMINI_API_KEY` | Google AI Studio key ([aistudio.google.com/apikey](https://aistudio.google.com/apikey)) |
| `GEMINI_MODEL` | Default `gemini-3.5-flash` (or `gemini-pro-latest`) |
| `GROK_API_KEY` | Optional xAI key if `LLM_PROVIDER=grok` |
| `GOOGLE_CLIENT_ID` | Google OAuth web client |
| `JWT_SECRET` | Session signing secret |
| `CORS_ORIGINS` | Frontend URL(s), e.g. `https://medirator2.netlify.app` |
| `COOKIE_CROSS_SITE` | `true` for Netlify → Render cookies |

Frontend companion: https://github.com/abdulrehman142/medirator2_frontend
