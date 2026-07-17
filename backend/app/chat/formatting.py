from app.ingestion.field_registry import PLAIN_FIELDS, SCOPE_PREFIXES, render_scope


def format_document_scope(doc_id: str, metadata: dict) -> str:
    """One-line summary of a document's structured scope (region/personnel
    coverage, doc_type, is_latest, effective_date, related_documents,
    default_precedence_rule), surfaced directly in tool results so the LLM can
    judge whether a retrieved document actually covers the situation asked
    about -- e.g. "does this document cover contractors?" or "is this passage
    actually dated for the year being asked about?" -- without depending on
    having also retrieved the exact prose sentence that states it via semantic
    similarity, which vector search can easily miss. `effective_date` in
    particular is a hard guardrail against presenting a current/general figure
    as if confirmed for an unconfirmed year -- a real hallucination found via
    manual testing (the model stated the current 15-day PTO figure as fact for
    a nonexistent "2021" document with no such date anywhere in the corpus).
    `default_precedence_rule` (e.g. "the more generous benefit applies") is the
    same "bake it onto every chunk" fix applied one level deeper: even once
    `related_documents` gets the model to check a connected document, the
    connected document's own general conflict-resolution rule is itself only
    reliably visible this way, not by hoping the model also retrieves that
    document's own CONFLICTS AND PRECEDENCE section.
    """
    title = metadata.get("document_title") or doc_id
    parts = [f'doc_id={doc_id}', f'title="{title}"']

    for field in PLAIN_FIELDS:
        rendered = field.render(field.chunk_key, metadata)
        if rendered is not None:
            parts.append(rendered)

    for prefix in SCOPE_PREFIXES:
        rendered = render_scope(prefix, metadata)
        if rendered is not None:
            parts.append(rendered)

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
