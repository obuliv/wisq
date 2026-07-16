from dataclasses import dataclass
from typing import Protocol

from app.ingestion.chunking import Chunk

Vector = list[float]

# Payload filter shape mirrors what Qdrant's filter DSL needs, kept provider-agnostic:
#   {"field": value}                   exact match      -> MatchValue
#   {"field": {"gte": .., "lte": ..}}  range             -> Range/DatetimeRange
#   {"field": {"any": [...]}}          list overlap      -> MatchAny
#   {"field": {"not_any": [...]}}      list non-overlap  -> MatchAny + must_not
#   {"field": {"any_or_empty": [...]}} overlap or empty  -> should[MatchAny, IsEmptyCondition]
Filters = dict[str, object]


@dataclass
class SparseVector:
    """BM25-style sparse embedding: parallel index/value arrays, matching
    Qdrant's sparse vector wire format directly."""

    indices: list[int]
    values: list[float]


class SparseEmbedder(Protocol):
    """Sparse (lexical/BM25-style) counterpart to Embedder, used for hybrid
    search. `embed` is for indexing document/chunk text; `embed_query` is for
    the search query -- these are genuinely asymmetric for real BM25 (document
    side is term-frequency-weighted with length normalization; query side is
    just term presence), unlike a dense Embedder where the same method suits
    both. See FakeSparseEmbedder for the deterministic stand-in (which doesn't
    need the asymmetry) and FastEmbedSparseEmbedder for the real one."""

    def embed(self, texts: list[str]) -> list[SparseVector]: ...

    def embed_query(self, texts: list[str]) -> list[SparseVector]: ...


@dataclass
class EmbeddedChunk:
    chunk: Chunk
    vector: Vector
    sparse_vector: SparseVector | None = None


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
    (see qdrant_store.py); an in-memory fake backs local dev/tests.

    When query_sparse_vector is omitted, search is dense-only (unchanged
    behavior). When provided, dense and sparse candidate rankings are fused via
    Reciprocal Rank Fusion (see rag/fusion.py) -- in Qdrant this maps to a Query
    API call with Prefetch(using="dense")/Prefetch(using="sparse") +
    FusionQuery(fusion=Fusion.RRF) instead of a single vector search.
    """

    def upsert(self, doc_id: str, chunks: list[EmbeddedChunk]) -> None: ...

    def update_metadata(self, doc_id: str, updates: dict) -> None:
        """Patches every existing chunk's metadata for `doc_id` in place -- no
        re-embedding, since only denormalized Document fields change (e.g.
        `is_latest` flipping when a newer version supersedes this document),
        not the chunk text/vector. Maps to Qdrant's set_payload (a cheap,
        standard payload-only update) once qdrant_store.py is implemented."""
        ...

    def search(
        self,
        query_vector: Vector,
        query_sparse_vector: SparseVector | None = None,
        top_k: int = 5,
        filters: Filters | None = None,
    ) -> list[ScoredChunk]: ...


class Retriever(Protocol):
    """RAG entrypoint used by chat. Default impl embeds the query (dense, and
    sparse when a SparseEmbedder is wired) and calls VectorStore.search, which
    owns any dense/sparse fusion -- callers stay agnostic to whether hybrid
    search is happening at all."""

    def retrieve(
        self, query: str, top_k: int = 5, filters: Filters | None = None
    ) -> list[ScoredChunk]: ...
