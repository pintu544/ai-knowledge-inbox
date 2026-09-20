# AI Knowledge Inbox

A small, production-style web app to **save notes and URLs, then ask questions over them**. Answers are generated with a retrieval-augmented (RAG) pipeline and always cite the source chunks they came from.

- **Backend:** FastAPI (Python), OpenAI for embeddings + chat, in-memory vector store
- **Frontend:** React + TypeScript + Vite + Tailwind
- **Storage:** in-memory and single-user — no auth, no database to set up

---

## Table of contents

- [What it does](#what-it-does)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Local setup](#local-setup)
  - [1. Backend](#1-backend)
  - [2. Frontend](#2-frontend)
- [API reference](#api-reference)
- [Configuration](#configuration)
- [Running the tests](#running-the-tests)
- [Design & tradeoffs](#design--tradeoffs)
- [What breaks at scale](#what-breaks-at-scale)
- [Project layout](#project-layout)

---

## What it does

1. **Ingest content** — save a plain-text note, or submit a URL that the server fetches and reduces to its main article text.
2. **Index it** — content is chunked, embedded with OpenAI, and stored in an in-memory vector store.
3. **Ask questions** — a question is embedded, the most similar chunks are retrieved, and an answer is generated from *only* that context, with inline `[n]` citations mapping to the source snippets.

---

## Architecture

```
┌────────────────────┐        /api/*  (Vite proxy)        ┌─────────────────────────┐
│   React frontend    │  ───────────────────────────────▶ │      FastAPI backend      │
│  (Vite, :5173)      │                                    │        (:8000)            │
│                     │                                    │                           │
│  IngestForm         │   POST /ingest                     │  routes → services        │
│  ItemsList          │   GET  /items                      │    ingestion              │
│  AskPanel/AnswerView│   POST /query                      │    rag                    │
└────────────────────┘                                    │    chunker                │
                                                            │    url_extractor          │
                                                            │  store: in-memory vectors │
                                                            │  clients: OpenAI          │
                                                            └────────────┬──────────────┘
                                                                         │
                                                                 OpenAI API
                                                          (embeddings + chat completions)
```

The backend is layered so responsibilities don't bleed together: **routes** validate and shape HTTP, **services** hold the ingestion and RAG logic, **store** owns vector math and persistence, and **clients** wrap the OpenAI SDK. Every error is returned in one consistent JSON envelope.

---

## Prerequisites

- **Python 3.11+**
- **Node 18+** (built and tested on Node 22)
- An **OpenAI API key** with access to an embedding model and a chat model

---

## Local setup

Clone the repo:

```bash
git clone https://github.com/pintu544/ai-knowledge-inbox.git
cd ai-knowledge-inbox
```

You'll run **two processes**: the backend (port 8000) and the frontend (port 5173).

### 1. Backend

```bash
cd backend

# Create and activate a virtual environment
python -m venv .venv
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure your API key
cp .env.example .env        # Windows: copy .env.example .env
# then edit .env and set OPENAI_API_KEY=sk-...

# Run the server
python -m uvicorn app.main:app --reload --port 8000
```

The API is now at **http://127.0.0.1:8000**, with interactive docs at **http://127.0.0.1:8000/docs**.

Sanity check:

```bash
curl http://127.0.0.1:8000/health
```

> Without `OPENAI_API_KEY`, the server still starts and `/items` works, but `/ingest` and `/query` return `503` until a key is configured.

### 2. Frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173**. The Vite dev server proxies `/api/*` to the backend on port 8000, so there's nothing else to configure. (Port 5173 is also on the backend's CORS allow-list if you'd rather call it directly.)

---

## API reference

All errors share one envelope:

```json
{ "error": { "code": "invalid_input", "message": "…", "details": { "field": "content" } } }
```

### `POST /ingest`

Save a note or URL and index it. Returns `201`.

```jsonc
// request
{ "source_type": "note", "content": "Postgres advisory locks are per-session…", "title": "Advisory locks" }
// or
{ "source_type": "url", "content": "https://fastapi.tiangolo.com/tutorial/first-steps/" }
```

```jsonc
// response
{
  "item": {
    "id": "…", "source_type": "note", "title": "Advisory locks",
    "source_url": null, "created_at": "…", "chunk_count": 1,
    "char_count": 168, "preview": "Postgres advisory locks…"
  },
  "chunk_count": 1
}
```

Errors: `422` (blank text / malformed URL), `429` (provider rate limit), `502` (URL fetch or provider failure), `503` (no API key).

### `GET /items`

List saved items, newest first (metadata + short preview, not full text).

```jsonc
{ "items": [ { "id": "…", "title": "…", "source_type": "url", "preview": "…", "chunk_count": 14, "char_count": 10006, "created_at": "…", "source_url": "…" } ], "count": 1 }
```

### `POST /query`

Ask a question over everything saved.

```jsonc
// request
{ "question": "What did I save about advisory locks?", "top_k": 4 }  // top_k optional (1–20)
```

```jsonc
// response
{
  "question": "…",
  "answer": "Session-scoped locks persist until released… [1]",
  "sources": [
    { "citation": 1, "item_id": "…", "title": "Advisory locks", "source_type": "note",
      "source_url": null, "chunk_index": 0, "score": 0.813, "snippet": "Postgres advisory locks…" }
  ],
  "model": "gpt-4o-mini",
  "retrieved_chunk_count": 4
}
```

The `[n]` markers in `answer` line up with each source's `citation`, so every claim can be traced to the text it came from.

### `GET /health`

Liveness plus the resolved chat/embedding models (handy for confirming your configured model is actually reachable).

---

## Configuration

Backend settings are read from environment variables or a `.env` file (see `backend/.env.example`). Real env vars always win.

| Variable | Default | Notes |
| --- | --- | --- |
| `OPENAI_API_KEY` | — | **Required** for `/ingest` and `/query`. |
| `OPENAI_MODEL` | `gpt-4o-mini` | Chat model. Validated at boot; falls back to the default if unreachable. |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Used for both chunks and questions. |
| `CHUNK_SIZE` | `900` | Target chunk size in characters. |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks (must be < `CHUNK_SIZE`). |
| `RETRIEVAL_TOP_K` | `4` | Default number of chunks retrieved per query. |
| `URL_FETCH_TIMEOUT_SECONDS` | `15` | Timeout for server-side URL fetches. |
| `CORS_ORIGINS` | `localhost:5173,127.0.0.1:5173` | Comma-separated allow-list. |
| `LOG_LEVEL` / `LOG_JSON` | `INFO` / `true` | Structured JSON logging by default. |

Model names are configuration-driven: pointing at a different chat model is a one-line `.env` change, no code change.

The frontend reads `VITE_API_BASE` (default `/api`) if you want to target a backend on another origin — put it in `frontend/.env.local`.

---

## Running the tests

Backend has a full pytest suite (chunker, vector store, URL extractor, routes, model resolution, and the OpenAI client — all mocked, no network calls):

```bash
cd backend
.\.venv\Scripts\Activate.ps1   # or source .venv/bin/activate
pytest
```

Frontend type-check / build:

```bash
cd frontend
npm run build   # tsc -b + vite build
```

---

## Deployment

The app is deployed as two independent services:

- **Backend → [Railway](https://railway.app):** built with Railpack (auto-detects Python via `runtime.txt` + `requirements.txt`) and started from the `Procfile`, which binds uvicorn to `0.0.0.0:$PORT`. Configuration (`OPENAI_API_KEY`, `OPENAI_MODEL`, `CORS_ORIGINS`, …) is set as Railway environment variables — nothing secret is committed.
- **Frontend → [Vercel](https://vercel.com):** a static Vite build (`vercel.json` sets the build command, output dir, and SPA rewrite). `VITE_API_BASE` is provided at build time and points at the Railway backend URL.

Deploy sketch (both CLIs assume you're logged in):

```bash
# Backend
cd backend
railway init --name ai-knowledge-inbox
railway up
railway variables --set "OPENAI_MODEL=gpt-5.5" --set "CORS_ORIGINS=https://<your-frontend>.vercel.app"
echo "$OPENAI_API_KEY" | railway variables set OPENAI_API_KEY --stdin
railway domain            # generate a public URL

# Frontend
cd ../frontend
vercel deploy --prod --build-env VITE_API_BASE=https://<your-backend>.up.railway.app
```

Set `CORS_ORIGINS` on the backend to the frontend's Vercel origin so browser requests are allowed while everything else is blocked.

## Design & tradeoffs

**Chunking — fixed-size windows snapped to sentence boundaries.**
~900 characters (~200 tokens) is roughly a paragraph: big enough to carry an idea, small enough that a match stays precise. A ~150-character overlap keeps a fact that straddles a boundary intact in one of the two neighbouring chunks. Cuts are snapped back to the nearest sentence end (then any whitespace) so chunks don't begin or end mid-word, which embeds poorly and reads badly as a citation. This is deliberately *not* token-accurate or structure-aware — see below.

**Vector store — in-memory numpy with an exhaustive cosine scan.**
For a single user, "all my saved chunks" is small. Embeddings are L2-normalised on insert, so cosine similarity is a single matrix-vector dot product, and top-k selection uses `argpartition` (O(n)). A linear scan over a few thousand 1536-dim vectors is sub-millisecond and avoids the operational weight of a real index. The store is thread-safe and intentionally ephemeral — restarting the server clears everything.

**Retrieval-only answers.**
The LLM is instructed to answer from the retrieved context and cite it. This keeps answers grounded and traceable rather than free-associating.

**One error envelope.**
Every failure — validation, upstream fetch, provider error — comes back as `{ "error": { "code", "message", "details? } }`, so the frontend has exactly one shape to parse and can surface field-level validation inline.

**Model resolution at boot.**
The configured chat model is validated once at startup. If it isn't reachable for the given key, the app logs a warning and falls back to a known-good default instead of failing every request. `GET /health` reports what's actually in use.

---

## What breaks at scale

This is a single-user, in-memory app by design. Moving toward production would mean:

- **Persistence** — swap the in-memory store for SQLite + a vector extension (e.g. `sqlite-vec`), pgvector, or a dedicated vector DB. Content currently vanishes on restart.
- **Indexing** — the exhaustive cosine scan is fine for thousands of chunks; beyond ~100k, switch to an approximate index (HNSW/IVF) to keep queries fast.
- **Token-accurate, structure-aware chunking** — count real tokens and respect headings, lists, and code blocks instead of raw character windows.
- **Multi-user** — add auth and scope items/vectors per user.
- **Async ingestion** — large URLs and batch embedding should move to a background queue so requests don't block.
- **Caching & rate limiting** — cache embeddings, and guard the OpenAI spend behind per-user limits.

---

## Project layout

```
.
├── backend/
│   ├── app/
│   │   ├── clients/        OpenAI SDK wrapper + error taxonomy
│   │   ├── routes/         /health, /ingest, /items, /query
│   │   ├── services/       chunker, ingestion, rag, url_extractor
│   │   ├── store/          in-memory vector store
│   │   ├── config.py       env-driven settings
│   │   ├── errors.py       single JSON error envelope
│   │   ├── models.py       Pydantic request/response/domain models
│   │   └── main.py         app factory + startup wiring
│   ├── tests/              pytest suite (fully mocked)
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── api/            typed client + contract types
│   │   ├── components/     IngestForm, ItemsList, AskPanel, AnswerView, ui
│   │   ├── hooks/          useItems
│   │   └── App.tsx
│   └── package.json
└── readme.md
```
