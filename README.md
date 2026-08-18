# Medirator 2.0 — Backend

FastAPI **keyword RAG** API for a synthetic Houston hospital knowledge base. Retrieves matching JSON records, then generates structured answers with **Gemini** (default) or optional **Grok**.

**Live:** https://medirator2-backend.onrender.com  
**Frontend repo:** https://github.com/abdulrehman142/medirator2_frontend  
**Frontend live:** https://medirator2.netlify.app

No vector DB. Synthetic data only · tokenized PII · not for real clinical use.

---

## What this API does

| Capability | Description |
|------------|-------------|
| **Google auth** | Verify Google ID token → JWT (cookie + token body) |
| **Keyword retrieval** | Score patients / medicines / inventory / instruments JSON |
| **RAG generate** | Prompt Gemini/Grok with retrieved context only → SOAP / cards JSON |
| **Data explore** | `GET /data/{category}` for the KB browser |
| **Health** | LLM + KB status |
| **Complaints** | Multipart store under `data/complaints/` (gitignored) |
| **QLoRA scaffold** | Optional fine-tune scripts under `finetune/` (CUDA) |

---

## Tech stack — what & why

| Technology | Why |
|------------|-----|
| **FastAPI + Uvicorn** | Typed REST API, OpenAPI docs, easy Render deploy |
| **Pydantic** | Request/response validation |
| **httpx** | Call Gemini / Grok / Google tokeninfo |
| **PyJWT** | HTTP-only session cookie + Bearer sessions |
| **python-dotenv** | Secrets and model config |
| **Local JSON KB** | ~1,270 synthetic rows; no Postgres required for demo |
| **Keyword retriever** | Transparent scoring; no embeddings / vector DB |
| **Gemini (default)** | Cloud LLM for deployable generation (`GEMINI_API_KEY`) |
| **Grok (optional)** | Alternate provider via `LLM_PROVIDER=grok` |
| **QLoRA (optional)** | Offline adapter training when a CUDA GPU is available |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│            Frontend (Netlify)                                  │
│  POST /auth/google · POST /query · GET /data · complaints      │
└──────────────────────────────┬───────────────────────────────┘
                               │ HTTPS + CORS
┌──────────────────────────────▼───────────────────────────────┐
│                 FastAPI  (Render :$PORT)                       │
│                                                              │
│  AUTH                         RAG PIPELINE                   │
│  ────                         ────────────                   │
│  routes/auth.py               routes/query.py                │
│       ↓                            ↓                         │
│  verify Google ID token       pipeline.run_query             │
│       ↓                            ↓                         │
│  JWT cookie + token           retriever.py  (keywords)       │
│                                    ↓                         │
│                               generator.py ──────────────┐   │
│                                    ↑                     │   │
│                               storage.py                 │   │
└──────────────────────────────────┼───────────────────────┼───┘
                                   │                       │
                       ┌───────────▼───────────┐   ┌───────▼────────┐
                       │   data/*.json         │   │ Gemini / Grok  │
                       │   1,270 synthetic     │   │ cloud LLM      │
                       │   hospital records    │   └────────────────┘
                       └───────────────────────┘
```

### RAG query flow

```
POST /query { query, category? }
        │
        ├─ detect / force category
        ├─ keyword score records (top 2)
        ├─ confidence = min(1, best_score / 8)
        ├─ prompt: "answer using ONLY this context" + JSON schema
        ├─ Gemini (or Grok) → structured JSON
        └─ response: answer, structured, retrieved, confidence, model
```

**Confidence** is retrieval strength only — not LLM quality and not QLoRA.

---

## Project structure

```
backend/
├── app/
│   ├── main.py                 # FastAPI + CORS
│   ├── auth.py                 # Google verify, JWT, cookies
│   ├── models/schema.py
│   ├── routes/
│   │   ├── auth.py             # /auth/google, /me, /logout
│   │   ├── query.py            # /query, /retrieve, /generate
│   │   ├── data.py             # /data, /data/{category}
│   │   ├── health.py           # /health
│   │   └── complaints.py       # /complaints
│   └── rag/
│       ├── storage.py          # Load JSON KB
│       ├── retriever.py        # Keyword scoring
│       ├── generator.py        # Gemini / Grok prompts
│       └── pipeline.py         # retrieve → generate
├── data/
│   ├── patients.json           # 450
│   ├── medicines.json          # 320
│   ├── inventory.json          # 280
│   ├── instruments.json        # 220
│   └── kb_meta.json
├── finetune/                   # Optional QLoRA
├── scripts/generate_kb.py
├── requirements.txt
└── .env.example
```

---

## Knowledge base

| Category | Count | Used for |
|----------|------:|----------|
| Patients | 450 | SOAP-style answers |
| Medicines | 320 | Dose / contraindications |
| Inventory | 280 | Stock / location |
| Instruments | 220 | Equipment / maintenance |
| **Total** | **1,270** | Cross-linked IDs + keywords |

PII fields are tokens (`mrn_token`, etc.). No database add-on required on Render.

---

## API

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/` | API info |
| `GET` | `/health` | KB + LLM status |
| `POST` | `/auth/google` | `{ credential }` → JWT + user |
| `GET` | `/auth/me` | Current user (Bearer or cookie) |
| `POST` | `/auth/logout` | Clear cookie |
| `POST` | `/query` | Full RAG |
| `POST` | `/retrieve` | Keywords only |
| `POST` | `/generate` | Generate from context |
| `GET` | `/data/{category}` | Browse KB |
| `POST` | `/complaints` | Multipart complaint |

Docs: https://medirator2-backend.onrender.com/docs

---

## Setup (local)

### Prerequisites

- Python 3.11+
- Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)  
  (or Grok key if `LLM_PROVIDER=grok`)
- Google OAuth Web client ID (same as frontend)

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`:

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-gemini-key
GEMINI_MODEL=gemini-3.5-flash

GOOGLE_CLIENT_ID=your-google-client-id.apps.googleusercontent.com
JWT_SECRET=use-a-long-random-secret-at-least-32-chars

CORS_ORIGINS=http://localhost:5173,https://medirator2.netlify.app
COOKIE_CROSS_SITE=true
```

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

API: **http://127.0.0.1:8000** · Docs: **/docs**

---

## Environment variables

| Variable | Purpose |
|----------|---------|
| `LLM_PROVIDER` | `gemini` (default) or `grok` |
| `GEMINI_API_KEY` | Google AI Studio key |
| `GEMINI_MODEL` | Default `gemini-3.5-flash` |
| `GROK_API_KEY` | Optional xAI key |
| `GROK_MODEL` | Default `grok-3-mini` |
| `GOOGLE_CLIENT_ID` | Must match frontend `VITE_GOOGLE_CLIENT_ID` |
| `JWT_SECRET` | Sign session tokens (≥32 bytes recommended) |
| `JWT_EXPIRE_HOURS` | Default `24` |
| `CORS_ORIGINS` | Comma-separated frontend origins |
| `COOKIE_CROSS_SITE` | `true` → `SameSite=None; Secure` for Netlify↔Render |

---

## Deploy (Render Web Service)

| Setting | Value |
|---------|--------|
| Repo | `medirator2_backend` |
| Root directory | _(empty)_ |
| Runtime | Python 3 |
| Build | `pip install -r requirements.txt` |
| Start | `uvicorn app.main:app --host 0.0.0.0 --port $PORT` |

Set the same env vars as local (especially `GEMINI_API_KEY`, `GEMINI_MODEL=gemini-3.5-flash`, `CORS_ORIGINS=https://medirator2.netlify.app`, `COOKIE_CROSS_SITE=true`).

**No database** is required.

---

## Design decisions

**Why keyword RAG?** Auditable demo constraint; no Pinecone/Chroma/FAISS.  
**Why JSON files?** Small synthetic KB; zero DB ops for hosting.  
**Why Gemini by default?** Deployable cloud LLM after Grok credit limits; swap with `LLM_PROVIDER`.  
**Why context-only prompts?** Reduces answers invented outside the KB.  
**Why QLoRA is optional?** Runtime path is prompt + retrieve; fine-tune is for experiments with a GPU.

---

## Example queries

| Category | Query |
|----------|-------|
| Patients | `SOAP for Alex Rivera` |
| Medicines | `Metformin dosage and contraindications` |
| Inventory | `N95 respirator stock status` |
| Instruments | `Portable ultrasound maintenance` |

---

## Companion

| | URL |
|--|-----|
| Frontend repo | https://github.com/abdulrehman142/medirator2_frontend |
| Frontend live | https://medirator2.netlify.app |
| Backend live | https://medirator2-backend.onrender.com |
