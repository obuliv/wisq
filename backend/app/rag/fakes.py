import hashlib
import math

from app.rag.interfaces import EmbeddedChunk, Filters, ScoredChunk, Vector

_DIM = 32


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


def _cosine(a: Vector, b: Vector) -> float:
    return sum(x * y for x, y in zip(a, b, strict=False))


def _matches(metadata: dict, filters: Filters) -> bool:
    for field, expected in filters.items():
        value = metadata.get(field)
        if isinstance(expected, dict):
            if "gte" in expected and (value is None or value < expected["gte"]):
                return False
            if "lte" in expected and (value is None or value > expected["lte"]):
                return False
        elif value != expected:
            return False
    return True


class InMemoryVectorStore:
    """Volatile, single-process VectorStore for local dev/tests. Mirrors the filter
    semantics (exact match + gte/lte range) that Qdrant supports, so swapping in
    QdrantVectorStore later doesn't change caller behavior."""

    def __init__(self) -> None:
        self._items: list[EmbeddedChunk] = []

    def upsert(self, doc_id: str, chunks: list[EmbeddedChunk]) -> None:
        self._items = [c for c in self._items if c.chunk.doc_id != doc_id]
        self._items.extend(chunks)

    def search(
        self, query_vector: Vector, top_k: int = 5, filters: Filters | None = None
    ) -> list[ScoredChunk]:
        candidates = self._items
        if filters:
            candidates = [c for c in candidates if _matches(c.chunk.metadata, filters)]
        scored = [
            ScoredChunk(chunk=c.chunk, score=_cosine(query_vector, c.vector)) for c in candidates
        ]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]
