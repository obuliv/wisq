from app.rag.interfaces import Embedder, Filters, ScoredChunk, VectorStore


class SimpleRetriever:
    """Default Retriever: embed the query, search the vector store. Just wiring —
    reranking / hybrid (dense+sparse) fusion is a RAG deep-dive addition that would
    slot in here behind the same Retriever interface."""

    def __init__(self, embedder: Embedder, vector_store: VectorStore) -> None:
        self._embedder = embedder
        self._vector_store = vector_store

    def retrieve(
        self, query: str, top_k: int = 5, filters: Filters | None = None
    ) -> list[ScoredChunk]:
        (query_vector,) = self._embedder.embed([query])
        return self._vector_store.search(query_vector, top_k=top_k, filters=filters)
