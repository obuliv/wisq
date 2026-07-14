from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import chat, documents
from app.config import get_settings
from app.db.models import Base
from app.db.session import engine
from app.ingestion import loaders  # noqa: F401 - registers DocxLoader etc. into LOADER_REGISTRY


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Skeleton-only schema bootstrap; replace with Alembic migrations before this
    # goes anywhere near production data.
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield


app = FastAPI(title="Wisq Q&A", lifespan=lifespan)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)
app.include_router(chat.router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
