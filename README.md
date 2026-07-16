# wisq — Document Q&A

Upload `.docx` documents, index them for retrieval, and chat against them via an
LLM. FastAPI backend, React (Vite/TS) frontend, Postgres + Qdrant for storage.

> The vector store (`QdrantVectorStore`) is still an unimplemented stub — the app
> always retrieves against an in-memory fake regardless of config (see
> `CLAUDE.md`'s "Stub status" section). The LLM provider, however, is real and
> configurable: set `LLM_PROVIDER=openai` + `OPENAI_API_KEY` in `.env` to use a
> real model, or leave it at the default `fake` for a deterministic, no-network,
> no-cost provider.

## Prerequisites

- Python 3.11+
- Node 18+
- Docker + Docker Compose (only needed for the full-stack / Postgres+Qdrant setup)

## Quickest way to run it: full stack via Docker Compose

This runs Postgres, Qdrant, the backend, and the frontend together.

```bash
cp .env.example .env
docker-compose up --build
```

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000

Edit `.env` first if you want to change credentials/ports — see
`.env.example` for every variable and its default.

## Running backend and frontend separately (local dev)

Useful for iterating on one side without rebuilding containers.

### Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Needs a reachable Postgres + Qdrant. Either:
#   docker-compose up postgres qdrant   (from the repo root, in another terminal)
# or point DATABASE_URL/QDRANT_URL at your own instances.
export DATABASE_URL=postgresql+asyncpg://wisq:wisq@localhost:5432/wisq
export QDRANT_URL=http://localhost:6333
export UPLOAD_DIR=/tmp/wisq-uploads

uvicorn app.main:app --reload --port 8000
```

The API is then at http://localhost:8000, with interactive docs at
http://localhost:8000/docs.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Opens on http://localhost:5173 by default and proxies `/api` to
`http://localhost:8000` (see `vite.config.ts`) — the backend must already be
running.

## Running tests

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate   # if not already set up
pip install -r requirements.txt

pytest -q                                                          # full suite
pytest tests/test_documents.py::test_upload_docx_gets_indexed -q   # single test
```

Tests run against sqlite (in-memory) and the fake RAG/LLM implementations —
**no Postgres, Qdrant, or Docker required.**

Frontend has no test suite yet; `npm run build` (`tsc -b && vite build`) is the
closest thing to a check and will fail on type errors.

## Configuration reference

All settings are environment variables, listed with defaults in `.env.example`:

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | Postgres connection string (async driver, `postgresql+asyncpg://...`) |
| `QDRANT_URL` / `QDRANT_COLLECTION` | Qdrant connection (not yet consumed — see stub status) |
| `UPLOAD_DIR` | Where uploaded `.docx` files are stored on disk |
| `LLM_PROVIDER` | `fake` (default, no credentials/network calls) or `openai`. `anthropic` is a placeholder — not implemented yet |
| `OPENAI_API_KEY` / `OPENAI_MODEL` | Used when `LLM_PROVIDER=openai` |
| `CORS_ORIGINS` | Allowed origin(s) for the frontend dev server |

## More detail

`CLAUDE.md` has the full architecture writeup (composition-root DI, ingestion
pipeline stages, the agentic retrieval/tool-calling design, hybrid search, DB
schema) — worth reading before making non-trivial changes.
