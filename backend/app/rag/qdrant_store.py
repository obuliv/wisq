"""Qdrant-backed VectorStore -- SKELETON ONLY, fill in per the RAG deep-dive.

Matches the VectorStore protocol in interfaces.py so it's a drop-in replacement for
InMemoryVectorStore in dependencies.py once implemented. Left unimplemented on
purpose per the RAG design doc: embedding model / chunking / hybrid-search choices
need to land first and will shape exactly what payload fields get indexed.
"""

from qdrant_client import QdrantClient

from app.rag.interfaces import EmbeddedChunk, Filters, ScoredChunk, Vector


class QdrantVectorStore:
    def __init__(self, url: str, collection: str) -> None:
        self._client = QdrantClient(url=url)
        self._collection = collection

    def upsert(self, doc_id: str, chunks: list[EmbeddedChunk]) -> None:
        # TODO: ensure_collection (vector size from embedder, distance=Cosine),
        # then self._client.upsert(collection_name=self._collection, points=[...]),
        # with payload = {"doc_id": doc_id, "text": chunk.text, "locator": ...,
        # **chunk.metadata} so effective_date/location filters (see design doc) work.
        raise NotImplementedError

    def search(
        self, query_vector: Vector, top_k: int = 5, filters: Filters | None = None
    ) -> list[ScoredChunk]:
        # TODO: translate `filters` into qdrant_client.models.Filter (MatchValue /
        # Range / DatetimeRange) and call self._client.search(...); map hits back to
        # ScoredChunk. Consider models.Filter + query_filter, and a payload index on
        # frequently-filtered fields (effective_date, location) for performance.
        raise NotImplementedError
