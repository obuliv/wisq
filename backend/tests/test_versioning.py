from sqlalchemy import select

from app.db.models import Document, DocumentRelationship
from app.db.session import async_session_factory
from app.ingestion.versioning import parse_version, resolve_latest


def test_parse_version():
    assert parse_version("2026") == (2026,)
    assert parse_version("2.1") == (2, 1)
    assert parse_version("2") == (2,)
    assert parse_version(None) is None
    assert parse_version("not-a-version") is None


async def _make_document(db, *, filename: str, title: str, version: str | None) -> Document:
    document = Document(
        filename=filename,
        content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        storage_path=f"/tmp/{filename}",
        title=title,
        version=version,
    )
    db.add(document)
    await db.flush()
    return document


async def test_resolve_latest_in_order():
    async with async_session_factory() as db:
        v2025 = await _make_document(db, filename="h2025.docx", title="Employee Handbook", version="2025")
        changed = await resolve_latest(db, v2025)
        await db.commit()
        assert v2025.is_latest is True
        assert changed == []  # first upload of this title, no siblings to flip

        v2026 = await _make_document(db, filename="h2026.docx", title="Employee Handbook", version="2026")
        changed = await resolve_latest(db, v2026)
        await db.commit()

        await db.refresh(v2025)
        assert v2025.is_latest is False
        assert v2026.is_latest is True
        assert v2025.document_group_key == v2026.document_group_key
        # The caller (IngestionPipeline._resolve_version) uses this to patch
        # v2025's already-indexed chunks' stale is_latest metadata.
        assert [d.id for d in changed] == [v2025.id]

        result = await db.execute(
            select(DocumentRelationship).where(
                DocumentRelationship.relation_type == "supersedes",
                DocumentRelationship.source_doc_id == v2026.id,
            )
        )
        relationship = result.scalar_one()
        assert relationship.target_doc_id == v2025.id


async def test_resolve_latest_out_of_order():
    async with async_session_factory() as db:
        v2026 = await _make_document(
            db, filename="oo-h2026.docx", title="OOO Handbook", version="2026"
        )
        await resolve_latest(db, v2026)
        await db.commit()

        # An older version uploaded *after* a newer one should not become latest.
        v2025 = await _make_document(
            db, filename="oo-h2025.docx", title="OOO Handbook", version="2025"
        )
        changed = await resolve_latest(db, v2025)
        await db.commit()

        await db.refresh(v2026)
        assert v2026.is_latest is True
        assert v2025.is_latest is False
        # v2026's is_latest value never changes here, so no chunk patch is needed.
        assert changed == []
