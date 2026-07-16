from app.ingestion.chunking import Chunk
from app.rag.fakes import InMemoryVectorStore
from app.rag.interfaces import EmbeddedChunk, SparseVector


def _store_with(*embedded: EmbeddedChunk) -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    for e in embedded:
        store.upsert(e.chunk.doc_id, [e])
    return store


def test_dense_only_search_ignores_sparse_vectors_and_is_unchanged():
    chunk_a = EmbeddedChunk(chunk=Chunk(doc_id="d1", text="a"), vector=[0.9, 0.1])
    chunk_b = EmbeddedChunk(
        chunk=Chunk(doc_id="d2", text="b"),
        vector=[0.1, 0.9],
        sparse_vector=SparseVector(indices=[0], values=[5.0]),
    )
    store = _store_with(chunk_a, chunk_b)

    # No query_sparse_vector -> dense-only, same ranking as before hybrid search existed.
    results = store.search([1.0, 0.0], top_k=5)

    assert [r.chunk.id for r in results] == [chunk_a.chunk.id, chunk_b.chunk.id]
    assert results[0].score == 0.9
    assert results[1].score == 0.1


def test_hybrid_fusion_can_flip_the_dense_only_winner():
    # A: strong dense match, zero sparse overlap (excluded from sparse ranking).
    # B: weak dense match, strong sparse match -- RRF should surface B above A
    # even though a dense-only search would rank A first.
    chunk_a = EmbeddedChunk(
        chunk=Chunk(doc_id="d1", text="a"),
        vector=[0.9, 0.1],
        sparse_vector=SparseVector(indices=[1], values=[1.0]),
    )
    chunk_b = EmbeddedChunk(
        chunk=Chunk(doc_id="d2", text="b"),
        vector=[0.1, 0.9],
        sparse_vector=SparseVector(indices=[0], values=[5.0]),
    )
    store = _store_with(chunk_a, chunk_b)
    query_vector = [1.0, 0.0]
    query_sparse = SparseVector(indices=[0], values=[1.0])

    dense_only = store.search(query_vector, top_k=5)
    assert [r.chunk.id for r in dense_only] == [chunk_a.chunk.id, chunk_b.chunk.id]

    hybrid = store.search(query_vector, query_sparse_vector=query_sparse, top_k=5)
    assert [r.chunk.id for r in hybrid] == [chunk_b.chunk.id, chunk_a.chunk.id]


def test_chunk_with_no_sparse_vector_still_surfaces_via_dense_in_hybrid_query():
    chunk_dense_only = EmbeddedChunk(chunk=Chunk(doc_id="d1", text="c"), vector=[0.5, 0.5])
    store = _store_with(chunk_dense_only)

    results = store.search(
        [1.0, 0.0], query_sparse_vector=SparseVector(indices=[0], values=[1.0]), top_k=5
    )

    assert len(results) == 1
    assert results[0].chunk.id == chunk_dense_only.chunk.id
    assert results[0].score > 0
