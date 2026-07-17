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


def test_any_or_empty_fuzzy_matches_formal_region_name():
    # Regression test for a real bug found via end-to-end testing: ingestion
    # extracted "People's Republic of China" (the document's own formal name)
    # while a real query naturally asks for "China" -- an exact-match predicate
    # silently excludes this chunk even though it's exactly what's being asked
    # about.
    store = InMemoryVectorStore()
    apac_chunk = Chunk(
        doc_id="doc-1",
        text="APAC benefits policy",
        metadata={
            "regions_included": ["People's Republic of China", "Japan", "Taiwan"],
            "regions_excluded": [],
        },
    )
    _upsert(store, apac_chunk)

    (query_vector,) = FakeEmbedder().embed(["policy"])
    results = store.search(
        query_vector, top_k=5, filters={"regions_included": {"any_or_empty": ["China"]}}
    )

    assert {r.chunk.id for r in results} == {apac_chunk.id}


def test_any_or_empty_fuzzy_match_does_not_over_match_unrelated_regions():
    store = InMemoryVectorStore()
    apac_chunk = Chunk(
        doc_id="doc-1",
        text="APAC benefits policy",
        metadata={"regions_included": ["Japan"], "regions_excluded": []},
    )
    _upsert(store, apac_chunk)

    (query_vector,) = FakeEmbedder().embed(["policy"])
    results = store.search(
        query_vector, top_k=5, filters={"regions_included": {"any_or_empty": ["China"]}}
    )

    assert results == []


def test_doc_type_scalar_filter_fuzzy_matches():
    store = InMemoryVectorStore()
    chunk = Chunk(doc_id="doc-1", text="handbook text", metadata={"doc_type": "Acme Employee Handbook"})
    _upsert(store, chunk)

    (query_vector,) = FakeEmbedder().embed(["handbook"])
    results = store.search(
        query_vector, top_k=5, filters={"doc_type": "Employee Handbook"}
    )

    assert {r.chunk.id for r in results} == {chunk.id}


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
