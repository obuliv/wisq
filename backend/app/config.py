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
    # gpt-5.6-luna measured noticeably more reliable than gpt-4o-mini on
    # multi-document precedence reasoning (e.g. applying "the more generous
    # benefit applies" across two conflicting figures) in live testing -- see
    # CLAUDE.md's Retrieval section. It's a reasoning model, which is why
    # openai_reasoning_effort defaults to "none" alongside it (see below).
    openai_model: str = "gpt-5.6-luna"
    # Reasoning models (e.g. gpt-5.x, including the gpt-5.6-luna default above)
    # reject tool calls on chat.completions unless reasoning_effort is set
    # explicitly; non-reasoning models (e.g. gpt-4o-mini) reject the param
    # outright if it's sent at all ("Unrecognized request argument"). Clear
    # this to "" in .env if you swap openai_model to a non-reasoning model.
    openai_reasoning_effort: str = "none"
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
