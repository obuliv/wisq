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
                "description": "Region or country to scope the search to, e.g. 'Singapore'. Omit to search all regions.",
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
    if geography:
        filters["regions_included"] = {"any_or_empty": [geography]}
        filters["regions_excluded"] = {"not_any": [geography]}

    if year:
        filters["effective_date"] = {"gte": f"{year}-01-01", "lte": f"{year}-12-31"}

    doc_type = arguments.get("doc_type")
    if doc_type:
        filters["doc_type"] = doc_type

    return filters


def format_search_results(scored: list[ScoredChunk]) -> str:
    if not scored:
        return "No matching passages found."
    blocks = []
    for s in scored:
        heading = " > ".join(s.chunk.metadata.get("heading_path", [])) or None
        header = f"[doc_id={s.chunk.doc_id} locator={s.chunk.locator} heading={heading}]"
        blocks.append(f"{header}\n{s.chunk.text}")
    return "\n\n".join(blocks)
