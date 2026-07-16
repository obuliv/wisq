from app.chat.search_tool import build_search_filters


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
