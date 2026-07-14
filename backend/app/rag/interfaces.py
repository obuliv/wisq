from dataclasses import dataclass
from typing import Protocol

from app.ingestion.chunking import Chunk

Vector = list[float]

# Payload filter shape mirrors what Qdrant's filter DSL needs, kept provider-agnostic
# here: {"field": value} for exact match, {"field": {"gte": ..., "lte": ...}} for range.
Filters = dict[str, object]


@dataclass
class EmbeddedChunk:
    chunk: Chunk
    vector: Vector


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


class Embedder(Protocol):
    """Turns text into vectors. Provider (OpenAI, local sentence-transformers, ...)
    is a RAG deep-dive decision — deferred."""

    def embed(self, texts: list[str]) -> list[Vector]: ...


class VectorStore(Protocol):
    """Persists and searches chunk embeddings + payload (doc_id, locator, metadata
    like effective_date/location for filtering). Qdrant-backed in production
    (see qdrant_store.py); an in-memory fake backs local dev/tests."""

    def upsert(self, doc_id: str, chunks: list[EmbeddedChunk]) -> None: ...

    def search(
        self, query_vector: Vector, top_k: int = 5, filters: Filters | None = None
    ) -> list[ScoredChunk]: ...


class Retriever(Protocol):
    """RAG entrypoint used by chat. Default impl just embeds the query and calls
    VectorStore.search; a later design may add reranking or hybrid (dense+sparse)
    fusion behind this same interface without changing callers."""

    def retrieve(
        self, query: str, top_k: int = 5, filters: Filters | None = None
    ) -> list[ScoredChunk]: ...
