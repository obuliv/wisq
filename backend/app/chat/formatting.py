def format_document_scope(doc_id: str, metadata: dict) -> str:
    """One-line summary of a document's structured scope (region/personnel
    coverage, doc_type, is_latest), surfaced directly in tool results so the
    LLM can judge whether a retrieved document actually covers the situation
    asked about -- e.g. "does this document cover contractors?" -- without
    depending on having also retrieved the exact prose sentence that states it
    via semantic similarity, which vector search can easily miss.
    """
    title = metadata.get("document_title") or doc_id
    parts = [f'doc_id={doc_id}', f'title="{title}"']

    doc_type = metadata.get("doc_type")
    if doc_type:
        parts.append(f"doc_type={doc_type}")

    if "is_latest" in metadata:
        parts.append(f"is_latest={metadata['is_latest']}")

    regions_included = metadata.get("regions_included") or []
    regions_excluded = metadata.get("regions_excluded") or []
    if regions_included or regions_excluded:
        parts.append(
            f"regions(included={regions_included or 'any'}, excluded={regions_excluded or 'none'})"
        )

    personnel_included = metadata.get("personnel_included") or []
    personnel_excluded = metadata.get("personnel_excluded") or []
    if personnel_included or personnel_excluded:
        parts.append(
            f"personnel(included={personnel_included or 'any'}, excluded={personnel_excluded or 'none'})"
        )

    return "[" + " ".join(parts) + "]"
