from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://wisq:wisq@localhost:5432/wisq"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "documents"

    upload_dir: str = "/data/uploads"

    # Which LLMClient dependencies.py::get_llm_client() wires up. "fake" (default)
    # needs no external calls/credentials -- tests always force this regardless of
    # .env (see tests/conftest.py) so the suite stays hermetic. Switch to "openai"
    # (or, once implemented, "anthropic") via .env to use a real provider.
    llm_provider: str = "fake"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    # Not implemented yet -- see app/llm/openai_client.py's docstring for the
    # swap-point pattern a future app/llm/anthropic_client.py would follow.
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-5"

    # Which Embedder dependencies.py::get_embedder() wires up. "fake" (default)
    # needs no external calls/credentials -- tests always force this regardless
    # of .env (see tests/conftest.py) so the suite stays hermetic. Switch to
    # "openai" via .env to use a real embedding model (reuses openai_api_key above).
    embedding_provider: str = "fake"
    openai_embedding_model: str = "text-embedding-3-small"

    # Which SparseEmbedder dependencies.py::get_sparse_embedder() wires up.
    # "fake" (default) needs no external calls/downloads -- tests always force
    # this regardless of .env (see tests/conftest.py) so the suite stays
    # hermetic. Switch to "fastembed" for a real, local BM25 model (no API key,
    # just a one-time model download).
    sparse_embedding_provider: str = "fake"
    bm25_model: str = "Qdrant/bm25"

    cors_origins: str = "http://localhost:3000"

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
