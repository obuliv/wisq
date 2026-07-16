from fastembed import SparseTextEmbedding

from app.rag.interfaces import SparseVector


class FastEmbedSparseEmbedder:
    """Real BM25 sparse embedder via FastEmbed (Qdrant's own open-source
    library) -- runs locally via ONNX, no API key/per-call network cost, just a
    one-time model download on first use. Swap point referenced in
    dependencies.py::get_sparse_embedder(), selected via
    SPARSE_EMBEDDING_PROVIDER=fastembed in .env.

    Uses `embed` (document/passage side: term-frequency-weighted, length
    normalized) for indexing and `query_embed` (term-presence only) for search
    queries -- these are genuinely different for real BM25, unlike
    FakeSparseEmbedder which doesn't need the distinction.
    """

    def __init__(self, model_name: str = "Qdrant/bm25") -> None:
        self._model = SparseTextEmbedding(model_name=model_name)

    def embed(self, texts: list[str]) -> list[SparseVector]:
        return [
            SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
            for e in self._model.embed(texts)
        ]

    def embed_query(self, texts: list[str]) -> list[SparseVector]:
        return [
            SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
            for e in self._model.query_embed(texts)
        ]
