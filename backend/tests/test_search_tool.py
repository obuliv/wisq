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


def test_format_search_results_surfaces_effective_date():
    # Regression test for a real hallucination found via manual testing: the
    # model stated the current (2026) PTO figure as fact for a nonexistent
    # "2021" query, since nothing in the tool result flagged which year the
    # retrieved passage was actually dated. effective_date is now a hard,
    # explicit signal right next to the retrieved text.
    chunk = Chunk(
        doc_id="doc-2026",
        text="The standard PTO entitlement is 15 days per year.",
        metadata={
            "document_title": "Acme Employee Handbook",
            "is_latest": True,
            "effective_date": "2026-01-01",
        },
    )

    result = format_search_results([ScoredChunk(chunk=chunk, score=0.9)])

    assert "effective_date=2026-01-01" in result


def test_format_search_results_omits_effective_date_when_absent():
    chunk = Chunk(
        doc_id="doc-1",
        text="Some text.",
        metadata={"document_title": "APAC Benefits Handbook", "is_latest": True},
    )

    result = format_search_results([ScoredChunk(chunk=chunk, score=0.9)])

    assert "effective_date" not in result


def test_format_search_results_surfaces_related_documents_hint():
    # Regression test for a real gap found via manual testing: a plain query
    # ("gym benefits for Taiwan") retrieved only the REGIONAL BENEFITS section,
    # never the CONFLICTS AND PRECEDENCE section that states "for all other
    # benefits, refer to the global handbook" -- so the model never learned it
    # should check the global handbook, and answered with the regional-only
    # figure. related_documents is baked onto every chunk of the document (see
    # test_pipeline.py), so it shows up regardless of which specific chunk hit.
    chunk = Chunk(
        doc_id="apac-doc",
        text="Gym reimbursement is $30/month.",
        metadata={
            "document_title": "APAC Benefits Handbook",
            "is_latest": True,
            "related_documents": [
                {"relation_type": "reference", "topic": "other benefits", "target": "global Acme Employee Handbook"},
                {"relation_type": "precedence", "topic": "PTO", "target": "global Acme Employee Handbook"},
            ],
        },
    )

    result = format_search_results([ScoredChunk(chunk=chunk, score=0.9)])

    assert 'reference/"other benefits"->"global Acme Employee Handbook"' in result
    assert 'precedence/"PTO"->"global Acme Employee Handbook"' in result
    assert "get_related_documents(doc_id=apac-doc)" in result


def test_format_search_results_omits_related_documents_when_absent():
    chunk = Chunk(
        doc_id="doc-1",
        text="Some text.",
        metadata={"document_title": "Acme Employee Handbook", "is_latest": True},
    )

    result = format_search_results([ScoredChunk(chunk=chunk, score=0.9)])

    assert "related_documents" not in result


def test_format_search_results_surfaces_default_precedence_rule():
    # Regression test for the "gym benefits for Taiwan" investigation: even
    # once related_documents gets the model to check the global handbook, the
    # global handbook's OWN general conflict-resolution rule is only reliably
    # visible this same way -- baked onto its own chunks, not dependent on
    # having retrieved its specific CONFLICTS AND PRECEDENCE section.
    chunk = Chunk(
        doc_id="global-doc",
        text="Gym reimbursement is $50/month.",
        metadata={
            "document_title": "Acme Employee Handbook",
            "is_latest": True,
            "default_precedence_rule": "The more generous perk or benefit applies.",
        },
    )

    result = format_search_results([ScoredChunk(chunk=chunk, score=0.9)])

    assert 'default_precedence_rule="The more generous perk or benefit applies."' in result


def test_format_search_results_omits_default_precedence_rule_when_absent():
    chunk = Chunk(
        doc_id="doc-1",
        text="Some text.",
        metadata={"document_title": "APAC Benefits Handbook", "is_latest": True},
    )

    result = format_search_results([ScoredChunk(chunk=chunk, score=0.9)])

    assert "default_precedence_rule" not in result
