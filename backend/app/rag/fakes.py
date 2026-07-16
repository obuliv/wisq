import hashlib
import math

from app.ingestion.chunking import Chunk
from app.rag.fusion import reciprocal_rank_fusion
from app.rag.interfaces import EmbeddedChunk, Filters, ScoredChunk, SparseVector, Vector

_DIM = 32
_SPARSE_BUCKETS = 2**16
_PREFETCH_LIMIT = 50


class FakeEmbedder:
    """Deterministic hash-based embedding, good enough for the fake vector store to
    return *something* plausible in dev/tests. Not a stand-in for a real embedding
    model's semantic quality — swap for a real Embedder when the RAG design lands."""

    def embed(self, texts: list[str]) -> list[Vector]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> Vector:
        vector = [0.0] * _DIM
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode()).digest()
            idx = digest[0] % _DIM
            vector[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


class FakeSparseEmbedder:
    """Deterministic hashed term-frequency sparse embedding -- a stand-in for a
    real BM25/SPLADE model (e.g. fastembed), same "Stub status" convention as
    FakeEmbedder. Not IDF-weighted, so it's not a real BM25 approximation, just
    enough lexical signal to prove the hybrid fusion path end-to-end."""

    def embed(self, texts: list[str]) -> list[SparseVector]:
        return [self._embed_one(text) for text in texts]

    def embed_query(self, texts: list[str]) -> list[SparseVector]:
        # No document/query asymmetry needed for this stand-in -- unlike real
        # BM25 (see SparseEmbedder's docstring / FastEmbedSparseEmbedder).
        return self.embed(texts)

    def _embed_one(self, text: str) -> SparseVector:
        counts: dict[int, float] = {}
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode()).digest()
            idx = int.from_bytes(digest[:4], "big") % _SPARSE_BUCKETS
            counts[idx] = counts.get(idx, 0.0) + 1.0
        indices = sorted(counts)
        return SparseVector(indices=indices, values=[counts[i] for i in indices])


def _cosine(a: Vector, b: Vector) -> float:
    return sum(x * y for x, y in zip(a, b, strict=False))


def _sparse_dot(a: SparseVector, b: SparseVector) -> float:
    b_map = dict(zip(b.indices, b.values, strict=True))
    return sum(v * b_map.get(i, 0.0) for i, v in zip(a.indices, a.values, strict=True))


def _flatten(chunk: Chunk) -> dict:
    """Mirrors the real Qdrant payload shape documented in qdrant_store.py's
    TODOs: doc_id/id/locator are top-level Chunk fields, not inside
    chunk.metadata, but filters need to be able to target them too."""
    return {**chunk.metadata, "doc_id": chunk.doc_id, "id": chunk.id, "locator": chunk.locator}


def _matches(chunk: Chunk, filters: Filters) -> bool:
    payload = _flatten(chunk)
    for field, expected in filters.items():
        value = payload.get(field)
        if isinstance(expected, dict):
            if "gte" in expected and (value is None or value < expected["gte"]):
                return False
            if "lte" in expected and (value is None or value > expected["lte"]):
                return False
            if "any" in expected:
                # List-field overlap -- Qdrant MatchAny. An empty/missing field
                # never matches (unlike any_or_empty below).
                if not set(value or []) & set(expected["any"]):
                    return False
            if "not_any" in expected:
                # List-field must NOT overlap -- Qdrant MatchAny + must_not.
                if set(value or []) & set(expected["not_any"]):
                    return False
            if "any_or_empty" in expected:
                # Overlaps, OR the field is empty/missing -- e.g. an empty
                # regions_included list means "applies everywhere", so it
                # should match any requested geography rather than none.
                values = value or []
                if values and not (set(values) & set(expected["any_or_empty"])):
                    return False
        elif value != expected:
            return False
    return True


class InMemoryVectorStore:
    """Volatile, single-process VectorStore for local dev/tests. Mirrors the filter
    semantics (exact match + gte/lte range, plus any/not_any/any_or_empty) that
    Qdrant supports, so swapping in QdrantVectorStore later doesn't change
    caller behavior. Also mirrors Qdrant's hybrid search shape client-side:
    dense and sparse candidates are each capped at prefetch_limit, then fused
    via Reciprocal Rank Fusion -- see rag/fusion.py."""

    def __init__(self, prefetch_limit: int = _PREFETCH_LIMIT) -> None:
        self._items: list[EmbeddedChunk] = []
        self._prefetch_limit = prefetch_limit

    def upsert(self, doc_id: str, chunks: list[EmbeddedChunk]) -> None:
        self._items = [c for c in self._items if c.chunk.doc_id != doc_id]
        self._items.extend(chunks)

    def update_metadata(self, doc_id: str, updates: dict) -> None:
        for item in self._items:
            if item.chunk.doc_id == doc_id:
                item.chunk.metadata.update(updates)

    def search(
        self,
        query_vector: Vector,
        query_sparse_vector: SparseVector | None = None,
        top_k: int = 5,
        filters: Filters | None = None,
    ) -> list[ScoredChunk]:
        candidates = self._items
        if filters:
            candidates = [c for c in candidates if _matches(c.chunk, filters)]

        if query_sparse_vector is None:
            # Dense-only: unchanged behavior from before hybrid search existed.
            scored = [
                ScoredChunk(chunk=c.chunk, score=_cosine(query_vector, c.vector)) for c in candidates
            ]
            scored.sort(key=lambda s: s.score, reverse=True)
            return scored[:top_k]

        by_id = {c.chunk.id: c for c in candidates}
        dense_ranked = sorted(candidates, key=lambda c: (-_cosine(query_vector, c.vector), c.chunk.id))
        dense_top = [c.chunk.id for c in dense_ranked[: self._prefetch_limit]]

        sparse_candidates = [c for c in candidates if c.sparse_vector is not None]
        sparse_ranked = sorted(
            (c for c in sparse_candidates if _sparse_dot(query_sparse_vector, c.sparse_vector) > 0),
            key=lambda c: (-_sparse_dot(query_sparse_vector, c.sparse_vector), c.chunk.id),
        )
        sparse_top = [c.chunk.id for c in sparse_ranked[: self._prefetch_limit]]

        fused = reciprocal_rank_fusion([dense_top, sparse_top])
        ranked_ids = sorted(fused, key=lambda cid: (-fused[cid], cid))
        return [ScoredChunk(chunk=by_id[cid].chunk, score=fused[cid]) for cid in ranked_ids[:top_k]]
