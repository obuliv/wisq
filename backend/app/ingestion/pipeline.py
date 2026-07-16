import logging
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Document, DocumentStatus
from app.ingestion.chunking import Chunk, Chunker
from app.ingestion.loaders.base import Element, get_loader
from app.ingestion.metadata import DocumentMetadataExtractor, GeographicScope
from app.ingestion.relationships import SectionAnnotationExtractor, extract_and_store_relationships
from app.ingestion.versioning import resolve_latest
from app.rag.interfaces import EmbeddedChunk, Embedder, SparseEmbedder, VectorStore

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Orchestrates the full ingestion sequence for one document, updating its
    status row as it goes:

        load -> [metadata extraction -> version resolution -> relationship
        extraction] -> chunk -> embed -> upsert

    The bracketed enrichment stages are best-effort: a failure there is logged
    and skipped, not raised, since they enrich the document rather than being
    required to index it. Only a load/chunk/embed/upsert failure marks the
    document FAILED. Which Embedder/VectorStore/Chunker/extractors are plugged
    in is a RAG deep-dive concern; this module only sequences the steps.
    """

    def __init__(
        self,
        chunker: Chunker,
        embedder: Embedder,
        vector_store: VectorStore,
        metadata_extractor: DocumentMetadataExtractor,
        relationship_extractor: SectionAnnotationExtractor,
        sparse_embedder: SparseEmbedder | None = None,
    ) -> None:
        self._chunker = chunker
        self._embedder = embedder
        self._vector_store = vector_store
        self._metadata_extractor = metadata_extractor
        self._relationship_extractor = relationship_extractor
        self._sparse_embedder = sparse_embedder

    async def run(self, db: AsyncSession, doc_id: str, file_path: Path) -> None:
        document = await db.get(Document, doc_id)
        if document is None:
            return

        try:
            document.status = DocumentStatus.PROCESSING
            await db.commit()

            loader = get_loader(file_path.suffix)
            elements = loader.load(file_path)

            self._apply_metadata(document, file_path.name, elements)
            await self._resolve_version(db, document)
            geo_overrides = await self._extract_relationships(db, document, elements)

            chunks = self._chunker.split(doc_id, elements)
            self._enrich_chunk_metadata(chunks, document, geo_overrides)

            vectors = self._embedder.embed([c.text for c in chunks])
            sparse_vectors = (
                self._sparse_embedder.embed([c.text for c in chunks])
                if self._sparse_embedder is not None
                else [None] * len(chunks)
            )
            embedded = [
                EmbeddedChunk(chunk=c, vector=v, sparse_vector=sv)
                for c, v, sv in zip(chunks, vectors, sparse_vectors, strict=True)
            ]
            self._vector_store.upsert(doc_id, embedded)

            document.status = DocumentStatus.READY
            document.error_message = None
        except Exception as exc:  # noqa: BLE001 - surface any failure on the document row
            document.status = DocumentStatus.FAILED
            document.error_message = str(exc)
        finally:
            await db.commit()

    def _apply_metadata(self, document: Document, filename: str, elements: list[Element]) -> None:
        try:
            metadata = self._metadata_extractor.extract(filename, elements)
        except Exception:
            logger.exception("Metadata extraction failed for document %s", document.id)
            return

        document.doc_type = metadata.doc_type or document.doc_type
        document.title = metadata.title or document.title
        document.version = metadata.version or document.version
        document.effective_date = metadata.effective_date or document.effective_date
        if metadata.applicable_regions is not None:
            document.applicable_regions = metadata.applicable_regions.model_dump()

    async def _resolve_version(self, db: AsyncSession, document: Document) -> None:
        try:
            changed_siblings = await resolve_latest(db, document)
            for sibling in changed_siblings:
                # sibling's chunks were already embedded/indexed under the old
                # is_latest value -- that's a metadata snapshot, not a live view
                # of the Document row, so it needs an explicit patch or it stays
                # stale forever (search_documents' is_latest filter would keep
                # matching a superseded document's chunks otherwise).
                self._vector_store.update_metadata(sibling.id, {"is_latest": sibling.is_latest})
        except Exception:
            logger.exception("Version resolution failed for document %s", document.id)

    async def _extract_relationships(
        self, db: AsyncSession, document: Document, elements: list[Element]
    ) -> dict[tuple[str, ...], GeographicScope]:
        try:
            return await extract_and_store_relationships(
                db, document, elements, self._relationship_extractor
            )
        except Exception:
            logger.exception("Relationship extraction failed for document %s", document.id)
            return {}

    def _enrich_chunk_metadata(
        self,
        chunks: list[Chunk],
        document: Document,
        geo_overrides: dict[tuple[str, ...], GeographicScope],
    ) -> None:
        default_scope = (
            GeographicScope(**document.applicable_regions) if document.applicable_regions else None
        )
        for chunk in chunks:
            heading_path = tuple(chunk.metadata.get("heading_path", []))
            scope = self._resolve_geo_scope(heading_path, geo_overrides, default_scope)
            chunk.metadata.update(
                {
                    "doc_type": document.doc_type,
                    "version": document.version,
                    # Human-readable name for citations -- once an answer can cite
                    # chunks from more than one document (see get_related_documents),
                    # a bare doc_id UUID isn't enough to tell them apart in the UI.
                    "document_title": document.title or document.filename,
                    "effective_date": document.effective_date.isoformat()
                    if document.effective_date
                    else None,
                    "is_latest": document.is_latest,
                    "applicable_regions": scope.model_dump() if scope else None,
                    # Flat lists alongside the nested applicable_regions dict above --
                    # the any/not_any/any_or_empty Filters predicates (rag/fakes.py)
                    # need list-shaped payload fields to query, since the nested dict
                    # form can't be targeted by the generic exact-match/range engine.
                    "regions_included": scope.included if scope else [],
                    "regions_excluded": scope.excluded if scope else [],
                }
            )

    @staticmethod
    def _resolve_geo_scope(
        heading_path: tuple[str, ...],
        overrides: dict[tuple[str, ...], GeographicScope],
        default: GeographicScope | None,
    ) -> GeographicScope | None:
        best_key: tuple[str, ...] | None = None
        for key in overrides:
            if heading_path[: len(key)] == key and (best_key is None or len(key) > len(best_key)):
                best_key = key
        return overrides[best_key] if best_key is not None else default
