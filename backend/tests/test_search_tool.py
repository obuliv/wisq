from app.chat.search_tool import build_search_filters, format_search_results
from app.ingestion.chunking import Chunk
from app.rag.interfaces import ScoredChunk


def test_no_year_defaults_to_latest_only():
    filters = build_search_filters({"query": "PTO policy"})
    assert filters == {"is_latest": True}


def test_year_drops_the_latest_only_default():
    # Regression test: previously is_latest=True was always forced on, which
    # meant a query for a specific past year could never match a since-
    # superseded document -- the model tried exactly this and got a silent,
    # unexplained empty result (see search_agent.py's docstring/design notes).
    filters = build_search_filters({"query": "PTO policy", "year": 2025})
    assert "is_latest" not in filters
    assert filters["effective_date"] == {"gte": "2025-01-01", "lte": "2025-12-31"}


def test_year_combines_with_other_filters():
    filters = build_search_filters(
        {"query": "PTO policy", "year": 2025, "geography": "Singapore", "doc_type": "Employee Handbook"}
    )
    assert "is_latest" not in filters
    assert filters["effective_date"] == {"gte": "2025-01-01", "lte": "2025-12-31"}
    assert filters["regions_included"] == {"any_or_empty": ["Singapore"]}
    assert filters["regions_excluded"] == {"not_any": ["Singapore"]}
    assert filters["doc_type"] == "Employee Handbook"


def test_format_search_results_empty():
    assert format_search_results([]) == "No matching passages found."


def test_format_search_results_surfaces_document_scope_and_groups_by_doc():
    apac_chunk_1 = Chunk(
        doc_id="apac-doc",
        text="Contractors are not covered.",
        metadata={
            "document_title": "APAC Benefits Handbook",
            "is_latest": True,
            "regions_included": ["China", "Japan", "Taiwan"],
            "regions_excluded": [],
            "personnel_included": ["employees"],
            "personnel_excluded": ["contractors"],
            "heading_path": ["SCOPE"],
        },
    )
    apac_chunk_2 = Chunk(
        doc_id="apac-doc",
        text="Gym reimbursement is $30/month.",
        metadata={
            "document_title": "APAC Benefits Handbook",
            "is_latest": True,
            "regions_included": ["China", "Japan", "Taiwan"],
            "regions_excluded": [],
            "personnel_included": ["employees"],
            "personnel_excluded": ["contractors"],
            "heading_path": ["REGIONAL BENEFITS"],
        },
    )
    global_chunk = Chunk(
        doc_id="global-doc",
        text="Gym reimbursement is $50/month.",
        metadata={"document_title": "Acme Employee Handbook", "is_latest": True},
    )

    result = format_search_results(
        [
            ScoredChunk(chunk=apac_chunk_1, score=0.9),
            ScoredChunk(chunk=global_chunk, score=0.8),
            ScoredChunk(chunk=apac_chunk_2, score=0.7),
        ]
    )

    # One scope line per document, not per chunk -- chunks from the same doc grouped together.
    assert result.count("doc_id=apac-doc") == 1
    assert result.count("doc_id=global-doc") == 1
    assert 'personnel(included=[\'employees\'], excluded=[\'contractors\'])' in result
    assert "Contractors are not covered." in result
    assert "Gym reimbursement is $30/month." in result
    assert "Gym reimbursement is $50/month." in result
    # global_chunk has no regions/personnel metadata -- shouldn't render an empty scope clause.
    assert 'title="Acme Employee Handbook"' in result
