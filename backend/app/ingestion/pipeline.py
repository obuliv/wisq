from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Document, DocumentStatus
from app.ingestion.chunking import Chunker
from app.ingestion.loaders.base import get_loader
from app.rag.interfaces import EmbeddedChunk, Embedder, VectorStore


class IngestionPipeline:
    """Orchestrates load -> chunk -> embed -> upsert for one document, updating its
    status row as it goes. Which Embedder/VectorStore/Chunker are plugged in is a
    RAG deep-dive concern; this module only sequences the steps."""

    def __init__(self, chunker: Chunker, embedder: Embedder, vector_store: VectorStore) -> None:
        self._chunker = chunker
        self._embedder = embedder
        self._vector_store = vector_store

    async def run(self, db: AsyncSession, doc_id: str, file_path: Path) -> None:
        document = await db.get(Document, doc_id)
        if document is None:
            return

        try:
            document.status = DocumentStatus.PROCESSING
            await db.commit()

            loader = get_loader(file_path.suffix)
            sections = loader.load(file_path)
            chunks = self._chunker.split(doc_id, sections)

            vectors = self._embedder.embed([c.text for c in chunks])
            embedded = [EmbeddedChunk(chunk=c, vector=v) for c, v in zip(chunks, vectors, strict=True)]
            self._vector_store.upsert(doc_id, embedded)

            document.status = DocumentStatus.READY
            document.error_message = None
        except Exception as exc:  # noqa: BLE001 - surface any failure on the document row
            document.status = DocumentStatus.FAILED
            document.error_message = str(exc)
        finally:
            await db.commit()
