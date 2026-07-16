from dataclasses import dataclass

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.formatting import format_document_scope
from app.db.models import Document, DocumentRelationship
from app.llm.interfaces import ToolDefinition
from app.rag.interfaces import Retriever, ScoredChunk

RELATED_DOCS_TOOL = ToolDefinition(
    name="get_related_documents",
    description=(
        "Look up documents connected to one you've already found via search_documents "
        "(pass the doc_id shown in a search_documents result). Covers both cross-document "
        "precedence/conflict rules (e.g. a regional handbook's PTO section overriding a "
        "global handbook's) and older/newer versions of the same document. Call this when "
        "a passage references another document by name, states a rule about which document "
        "governs a topic, or you're unsure whether the passage you're citing is the current "
        "version."
    ),
    parameters={
        "type": "object",
        "properties": {
            "doc_id": {
                "type": "string",
                "description": "The doc_id of a document already found via search_documents.",
            },
            "topic": {
                "type": "string",
                "description": "Optional topic to help pick which connected content to fetch, e.g. 'PTO'.",
            },
        },
        "required": ["doc_id"],
    },
)


async def find_related_documents(db: AsyncSession, doc_id: str) -> list[DocumentRelationship]:
    result = await db.execute(
        select(DocumentRelationship).where(
            or_(
                DocumentRelationship.source_doc_id == doc_id,
                DocumentRelationship.target_doc_id == doc_id,
            )
        )
    )
    return list(result.scalars().all())


@dataclass
class RelatedDocContext:
    relationship: DocumentRelationship
    doc_is_source: bool
    other_doc_id: str | None  # None if unresolved (target_doc_id is None)
    other_doc_title: str | None
    chunks: list[ScoredChunk]  # empty if unresolved


async def load_related_context(
    db: AsyncSession,
    retriever: Retriever,
    doc_id: str,
    topic: str | None,
    fallback_query: str,
    related_doc_top_k: int,
) -> list[RelatedDocContext]:
    relationships = await find_related_documents(db, doc_id)
    contexts: list[RelatedDocContext] = []

    for rel in relationships:
        doc_is_source = rel.source_doc_id == doc_id
        other_doc_id = rel.target_doc_id if doc_is_source else rel.source_doc_id

        other_doc_title: str | None = None
        chunks: list[ScoredChunk] = []
        if other_doc_id is not None:
            other_doc = await db.get(Document, other_doc_id)
            if other_doc is not None:
                other_doc_title = other_doc.title or other_doc.filename
            query_text = topic or rel.topic or fallback_query
            chunks = retriever.retrieve(
                query_text, top_k=related_doc_top_k, filters={"doc_id": other_doc_id}
            )

        contexts.append(
            RelatedDocContext(
                relationship=rel,
                doc_is_source=doc_is_source,
                other_doc_id=other_doc_id,
                other_doc_title=other_doc_title,
                chunks=chunks,
            )
        )

    return contexts


def _format_chunks(chunks: list[ScoredChunk], tag: str) -> str:
    if not chunks:
        return ""
    blocks = [format_document_scope(chunks[0].chunk.doc_id, chunks[0].chunk.metadata)]
    for s in chunks:
        heading = " > ".join(s.chunk.metadata.get("heading_path", [])) or None
        header = f"[{tag} locator={s.chunk.locator} heading={heading}]"
        blocks.append(f"{header}\n{s.chunk.text}")
    return "\n\n".join(blocks)


def format_related_documents(doc_id: str, contexts: list[RelatedDocContext]) -> str:
    if not contexts:
        return f"No related documents found for doc_id={doc_id}."

    sections: list[str] = []
    for ctx in contexts:
        rel = ctx.relationship

        if ctx.other_doc_id is None:
            topic_part = f", topic={rel.topic}" if rel.topic else ""
            sections.append(
                f"[relationship unresolved] This document references another document, "
                f'"{rel.target_doc_ref}", which has not been found in the uploaded corpus '
                f'(relation_type={rel.relation_type}{topic_part}). Quoted text: "{rel.source_text}"'
            )
            continue

        other_label = ctx.other_doc_title or ctx.other_doc_id

        if rel.relation_type == "supersedes":
            if ctx.doc_is_source:
                note = (
                    f"This document (doc_id={doc_id}) supersedes an older version, "
                    f'"{other_label}" (doc_id={ctx.other_doc_id}). Prefer this document\'s '
                    "content over the older version unless the user specifically asks about "
                    "a prior/previous version."
                )
                tag = f"older version doc_id={ctx.other_doc_id}"
            else:
                note = (
                    f"This document (doc_id={doc_id}) has been superseded by a newer version, "
                    f'"{other_label}" (doc_id={ctx.other_doc_id}). Prefer the newer version\'s '
                    "content unless the user specifically asks about this older version."
                )
                tag = f"newer version doc_id={ctx.other_doc_id}"
        else:
            topic_note = f" for topic '{rel.topic}'" if rel.topic else ""
            if rel.precedence == "source_over_target":
                winner, loser = (
                    (doc_id, ctx.other_doc_id) if ctx.doc_is_source else (ctx.other_doc_id, doc_id)
                )
                note = f"Document {winner} takes precedence over document {loser}{topic_note}."
            elif rel.precedence == "target_over_source":
                winner, loser = (
                    (ctx.other_doc_id, doc_id) if ctx.doc_is_source else (doc_id, ctx.other_doc_id)
                )
                note = f"Document {winner} takes precedence over document {loser}{topic_note}."
            else:
                note = (
                    f'Document {doc_id} is related to document {ctx.other_doc_id} ("{other_label}") '
                    f'(relation_type={rel.relation_type}{topic_note}). Quoted text: "{rel.source_text}".'
                )
            tag = f"related doc_id={ctx.other_doc_id}"

        section = note
        if ctx.chunks:
            section += "\n\n" + _format_chunks(ctx.chunks, tag)
        sections.append(section)

    return "\n\n---\n\n".join(sections)
