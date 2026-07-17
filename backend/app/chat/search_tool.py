from app.chat.formatting import format_document_scope
from app.llm.interfaces import ToolDefinition
from app.rag.interfaces import Filters, ScoredChunk

SEARCH_TOOL = ToolDefinition(
    name="search_documents",
    description=(
        "Search the indexed document corpus for passages relevant to a query. "
        "You may call this more than once in the same turn -- e.g. first scoped "
        "to a specific geography and/or year, then again with fewer or no filters "
        "if the scoped search doesn't turn up an answer. Only answer once you "
        "have enough context; say you don't know if the documents don't contain "
        "the answer."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language search query.",
            },
            "geography": {
                "type": "string",
                "description": (
                    "A SPECIFIC country or named region to scope the search to, e.g. "
                    "'Singapore' or 'Taiwan' -- only set this when the user named a "
                    "specific country/region. Do NOT set this to a broad/continent-level "
                    "term like 'Asia' or 'Europe': documents are scoped by country, not "
                    "continent, so a continent-level filter silently excludes every "
                    "country-specific document and hides the fact that different "
                    "countries in that continent may have different policies. For a "
                    "continent-level or otherwise vague location, omit this filter "
                    "entirely so all region-specific documents are visible."
                ),
            },
            "year": {
                "type": "integer",
                "description": (
                    "Effective year to scope the search to. Omit to search only the "
                    "current version of each document. Providing a year also searches "
                    "superseded/older versions effective in that year -- use this to "
                    "answer questions about a specific past year or a named prior version."
                ),
            },
            "doc_type": {
                "type": "string",
                "description": "Document type to scope to, e.g. 'Employee Handbook'. Omit for any type.",
            },
        },
        "required": ["query"],
    },
)


_BROAD_GEOGRAPHY_TERMS = {
    "asia",
    "apac",
    "europe",
    "emea",
    "africa",
    "north america",
    "south america",
    "latin america",
    "oceania",
    "middle east",
    "worldwide",
    "global",
}


def build_search_filters(arguments: dict) -> Filters:
    """Maps search_documents tool call arguments to a Filters dict.

    is_latest=True only defaults on when no year is given -- an explicit year
    already disambiguates which version is wanted, so it shouldn't ALSO be
    required to be the current one. Without a year, default to "current only"
    so a plain query can't silently mix a superseded document's content in.
    (Previously is_latest was always forced on regardless of year, which meant
    a query like year=2025 against a since-superseded 2025 document could never
    match anything -- a real bug found via end-to-end testing: the model tried
    exactly this and got a silent, unexplained empty result.)
    """
    year = arguments.get("year")
    filters: Filters = {} if year else {"is_latest": True}

    geography = arguments.get("geography")
    if geography and geography.strip().lower() not in _BROAD_GEOGRAPHY_TERMS:
        # Documents are scoped by country, not continent -- despite the tool
        # description telling the model to omit continent-level terms like
        # "Asia", live testing showed it still passes them sometimes. Silently
        # dropping the filter for a known broad term (instead of trusting the
        # model to comply) guarantees country-specific documents in that
        # continent stay visible in results rather than being fuzzy-matched
        # away against a term ("Asia") that shares little text with the
        # specific country names ("China", "Japan", "Taiwan") they list.
        filters["regions_included"] = {"any_or_empty": [geography]}
        filters["regions_excluded"] = {"not_any": [geography]}

    if year:
        filters["effective_date"] = {"gte": f"{year}-01-01", "lte": f"{year}-12-31"}

    doc_type = arguments.get("doc_type")
    if doc_type:
        filters["doc_type"] = doc_type

    return filters


def format_search_results(scored: list[ScoredChunk]) -> str:
    """Groups matching chunks by document and prepends each document's
    structured scope (regions/personnel/doc_type/is_latest) once per document,
    via format_document_scope -- so the model sees whether a retrieved
    document actually covers the situation asked about directly in the tool
    result, rather than having to infer it from whichever prose chunk
    happened to rank highly."""
    if not scored:
        return "No matching passages found."

    by_doc: dict[str, list[ScoredChunk]] = {}
    order: list[str] = []
    for s in scored:
        if s.chunk.doc_id not in by_doc:
            by_doc[s.chunk.doc_id] = []
            order.append(s.chunk.doc_id)
        by_doc[s.chunk.doc_id].append(s)

    blocks = []
    for doc_id in order:
        chunks = by_doc[doc_id]
        scope_line = format_document_scope(doc_id, chunks[0].chunk.metadata)
        chunk_blocks = []
        for s in chunks:
            heading = " > ".join(s.chunk.metadata.get("heading_path", [])) or None
            chunk_blocks.append(f"(locator={s.chunk.locator} heading={heading})\n{s.chunk.text}")
        blocks.append(scope_line + "\n" + "\n\n".join(chunk_blocks))
    return "\n\n---\n\n".join(blocks)
