from app.chat.relationship_tool import (
    RelatedDocContext,
    find_related_documents,
    format_related_documents,
)
from app.db.models import Document, DocumentRelationship
from app.db.session import async_session_factory


async def test_find_related_documents_matches_from_either_side():
    async with async_session_factory() as db:
        db.add(Document(id="doc-a", filename="a.docx", content_type="x", storage_path="x"))
        db.add(Document(id="doc-b", filename="b.docx", content_type="x", storage_path="x"))
        db.add(
            DocumentRelationship(
                source_doc_id="doc-a",
                target_doc_id="doc-b",
                target_doc_ref="Document B",
                relation_type="precedence",
                topic="PTO",
                precedence="source_over_target",
                source_text="A takes precedence over B",
            )
        )
        await db.commit()

        from_source = await find_related_documents(db, "doc-a")
        from_target = await find_related_documents(db, "doc-b")

    assert len(from_source) == 1
    assert len(from_target) == 1
    assert from_source[0].id == from_target[0].id


def test_format_related_documents_no_relationships():
    assert format_related_documents("doc-a", []) == "No related documents found for doc_id=doc-a."


def test_format_related_documents_precedence_source_over_target_direction():
    rel = DocumentRelationship(
        source_doc_id="doc-a",
        target_doc_id="doc-b",
        target_doc_ref="Document B",
        relation_type="precedence",
        topic="PTO",
        precedence="source_over_target",
        source_text="A takes precedence over B",
    )

    # Queried from the source's side: doc-a should win.
    ctx_from_source = RelatedDocContext(
        relationship=rel, doc_is_source=True, other_doc_id="doc-b", other_doc_title="Doc B", chunks=[]
    )
    text = format_related_documents("doc-a", [ctx_from_source])
    assert "Document doc-a takes precedence over document doc-b" in text

    # Queried from the target's side: doc-a still wins (direction must not flip incorrectly).
    ctx_from_target = RelatedDocContext(
        relationship=rel, doc_is_source=False, other_doc_id="doc-a", other_doc_title="Doc A", chunks=[]
    )
    text2 = format_related_documents("doc-b", [ctx_from_target])
    assert "Document doc-a takes precedence over document doc-b" in text2


def test_format_related_documents_supersedes_direction():
    rel = DocumentRelationship(
        source_doc_id="doc-new",
        target_doc_id="doc-old",
        target_doc_ref="Old Handbook",
        relation_type="supersedes",
        source_text="Inferred from versioning",
    )

    ctx_newer_queried = RelatedDocContext(
        relationship=rel, doc_is_source=True, other_doc_id="doc-old", other_doc_title="Old Handbook", chunks=[]
    )
    text = format_related_documents("doc-new", [ctx_newer_queried])
    assert "supersedes an older version" in text

    ctx_older_queried = RelatedDocContext(
        relationship=rel, doc_is_source=False, other_doc_id="doc-new", other_doc_title="New Handbook", chunks=[]
    )
    text2 = format_related_documents("doc-old", [ctx_older_queried])
    assert "has been superseded by a newer version" in text2


def test_format_related_documents_unresolved_reference():
    rel = DocumentRelationship(
        source_doc_id="doc-a",
        target_doc_id=None,
        target_doc_ref="Global Handbook",
        relation_type="reference",
        source_text="refer to the global handbook",
    )
    ctx = RelatedDocContext(
        relationship=rel, doc_is_source=True, other_doc_id=None, other_doc_title=None, chunks=[]
    )

    text = format_related_documents("doc-a", [ctx])

    assert "has not been found in the uploaded corpus" in text
    assert "Global Handbook" in text
