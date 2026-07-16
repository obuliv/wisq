from app.ingestion.chunking import Chunk
from app.rag.fakes import FakeEmbedder, InMemoryVectorStore
from app.rag.interfaces import EmbeddedChunk


def _upsert(store: InMemoryVectorStore, chunk: Chunk) -> None:
    (vector,) = FakeEmbedder().embed([chunk.text])
    store.upsert(chunk.doc_id, [EmbeddedChunk(chunk=chunk, vector=vector)])


def test_doc_id_filter_matches_chunk():
    # Regression test: doc_id/id live on Chunk, not chunk.metadata, so a naive
    # metadata-only filter engine would silently match nothing here.
    store = InMemoryVectorStore()
    chunk = Chunk(doc_id="doc-1", text="hello world")
    _upsert(store, chunk)

    (query_vector,) = FakeEmbedder().embed(["hello"])
    results = store.search(query_vector, top_k=5, filters={"doc_id": "doc-1"})

    assert len(results) == 1
    assert results[0].chunk.id == chunk.id

    no_match = store.search(query_vector, top_k=5, filters={"doc_id": "doc-2"})
    assert no_match == []


def test_any_or_empty_matches_globally_applicable_chunk():
    # An empty regions_included list means "applies everywhere" -- a naive
    # list-overlap-only predicate would wrongly exclude it.
    store = InMemoryVectorStore()
    global_chunk = Chunk(
        doc_id="doc-1",
        text="global policy",
        metadata={"regions_included": [], "regions_excluded": []},
    )
    scoped_chunk = Chunk(
        doc_id="doc-2",
        text="regional policy",
        metadata={"regions_included": ["Malaysia"], "regions_excluded": []},
    )
    _upsert(store, global_chunk)
    _upsert(store, scoped_chunk)

    (query_vector,) = FakeEmbedder().embed(["policy"])
    results = store.search(
        query_vector,
        top_k=5,
        filters={"regions_included": {"any_or_empty": ["Singapore"]}},
    )

    result_ids = {r.chunk.id for r in results}
    assert global_chunk.id in result_ids
    assert scoped_chunk.id not in result_ids


def test_not_any_vetoes_excluded_region_even_when_included_is_empty():
    store = InMemoryVectorStore()
    excluded_chunk = Chunk(
        doc_id="doc-1",
        text="policy with carve-out",
        metadata={"regions_included": [], "regions_excluded": ["Japan"]},
    )
    _upsert(store, excluded_chunk)

    (query_vector,) = FakeEmbedder().embed(["policy"])
    results = store.search(
        query_vector,
        top_k=5,
        filters={"regions_excluded": {"not_any": ["Japan"]}},
    )

    assert results == []


def test_gte_lte_range_filter_unchanged():
    store = InMemoryVectorStore()
    early = Chunk(doc_id="doc-1", text="early doc", metadata={"effective_date": "2024-01-01"})
    late = Chunk(doc_id="doc-2", text="late doc", metadata={"effective_date": "2026-01-01"})
    _upsert(store, early)
    _upsert(store, late)

    (query_vector,) = FakeEmbedder().embed(["doc"])
    results = store.search(
        query_vector,
        top_k=5,
        filters={"effective_date": {"gte": "2025-01-01"}},
    )

    result_ids = {r.chunk.id for r in results}
    assert result_ids == {late.id}
