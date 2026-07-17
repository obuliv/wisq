def format_document_scope(doc_id: str, metadata: dict) -> str:
    """One-line summary of a document's structured scope (region/personnel
    coverage, doc_type, is_latest, effective_date), surfaced directly in tool
    results so the LLM can judge whether a retrieved document actually covers
    the situation asked about -- e.g. "does this document cover contractors?"
    or "is this passage actually dated for the year being asked about?" --
    without depending on having also retrieved the exact prose sentence that
    states it via semantic similarity, which vector search can easily miss.
    `effective_date` in particular is a hard guardrail against presenting a
    current/general figure as if confirmed for an unconfirmed year -- a real
    hallucination found via manual testing (the model stated the current 15-day
    PTO figure as fact for a nonexistent "2021" document with no such date
    anywhere in the corpus).
    """
    title = metadata.get("document_title") or doc_id
    parts = [f'doc_id={doc_id}', f'title="{title}"']

    doc_type = metadata.get("doc_type")
    if doc_type:
        parts.append(f"doc_type={doc_type}")

    effective_date = metadata.get("effective_date")
    if effective_date:
        parts.append(f"effective_date={effective_date}")

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

    related_documents = metadata.get("related_documents") or []
    if related_documents:
        rendered = ", ".join(
            f'{r["relation_type"]}/"{r["topic"]}"->"{r["target"]}"' if r.get("topic") else
            f'{r["relation_type"]}->"{r["target"]}"'
            for r in related_documents
        )
        parts.append(
            f"related_documents=[{rendered}] (call get_related_documents(doc_id={doc_id}) "
            "if relevant to the question)"
        )

    return "[" + " ".join(parts) + "]"
