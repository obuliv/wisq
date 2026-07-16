from app.rag.interfaces import Embedder, Filters, ScoredChunk, SparseEmbedder, VectorStore


class SimpleRetriever:
    """Default Retriever: embed the query (dense, and sparse when a
    SparseEmbedder is wired), search the vector store. Just wiring -- fusion of
    dense+sparse rankings lives inside VectorStore.search (see rag/fakes.py /
    rag/fusion.py), so this class stays agnostic to whether hybrid search is
    happening at all. Reranking would be a further RAG deep-dive addition that
    would slot in here behind the same Retriever interface."""

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        sparse_embedder: SparseEmbedder | None = None,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._sparse_embedder = sparse_embedder

    def retrieve(
        self, query: str, top_k: int = 5, filters: Filters | None = None
    ) -> list[ScoredChunk]:
        (query_vector,) = self._embedder.embed([query])
        query_sparse_vector = None
        if self._sparse_embedder is not None:
            (query_sparse_vector,) = self._sparse_embedder.embed_query([query])
        return self._vector_store.search(
            query_vector, query_sparse_vector=query_sparse_vector, top_k=top_k, filters=filters
        )
