from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Document, DocumentRelationship
from app.ingestion.metadata import normalize_title


def parse_version(version: str | None) -> tuple[int, ...] | None:
    """Parses a bare year ("2026"), bare int ("2"), or dotted semver-like string
    ("2.1", "2.1.0") into a comparable tuple of ints. Returns None when the
    version is missing or not in one of those shapes -- callers should then fall
    back to upload-order (created_at) instead."""
    if not version:
        return None
    try:
        return tuple(int(part) for part in version.split("."))
    except ValueError:
        return None


def _sort_key(document: Document) -> tuple:
    parsed = parse_version(document.version)
    return (parsed is not None, parsed or (), document.created_at)


async def resolve_latest(db: AsyncSession, document: Document) -> None:
    """Groups `document` with any prior uploads sharing the same normalized title,
    marks the newest as `is_latest=True` and flips the rest to False -- handles
    out-of-order uploads (an older version uploaded after a newer one is already
    marked non-latest on arrival). Records a `supersedes` DocumentRelationship
    when this upload displaces a previous latest version, as an audit trail."""
    if not document.title:
        # No canonical title extracted -- nothing to group against; leave the
        # default is_latest=True from the column default in place.
        return

    document.document_group_key = normalize_title(document.title)

    result = await db.execute(
        select(Document).where(
            Document.document_group_key == document.document_group_key,
            Document.id != document.id,
        )
    )
    siblings = list(result.scalars().all())
    if not siblings:
        document.is_latest = True
        return

    candidates = [*siblings, document]
    newest = max(candidates, key=_sort_key)
    for candidate in candidates:
        candidate.is_latest = candidate is newest

    if newest is document:
        previous_latest = max(siblings, key=_sort_key)
        db.add(
            DocumentRelationship(
                source_doc_id=document.id,
                target_doc_id=previous_latest.id,
                target_doc_ref=previous_latest.title or previous_latest.filename,
                relation_type="supersedes",
                source_text=(
                    f"Inferred from versioning: '{document.title}' "
                    f"(version {document.version}) supersedes version {previous_latest.version}."
                ),
            )
        )
