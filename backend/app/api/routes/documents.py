from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Document, DocumentStatus
from app.db.session import async_session_factory, get_db
from app.dependencies import get_document_store, get_ingestion_pipeline
from app.ingestion.loaders.base import LOADER_REGISTRY
from app.ingestion.pipeline import IngestionPipeline
from app.schemas import DocumentOut
from app.storage.interfaces import DocumentStore

router = APIRouter(prefix="/api/documents", tags=["documents"])


async def _run_ingestion(pipeline: IngestionPipeline, doc_id: str, file_path: Path) -> None:
    async with async_session_factory() as db:
        await pipeline.run(db, doc_id, file_path)


@router.post("", response_model=DocumentOut, status_code=201)
async def upload_document(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    store: DocumentStore = Depends(get_document_store),
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
) -> Document:
    extension = Path(file.filename or "").suffix.lower()
    if extension not in LOADER_REGISTRY:
        supported = ", ".join(sorted(LOADER_REGISTRY)) or "(none registered)"
        raise HTTPException(400, f"Unsupported file type {extension!r}. Supported: {supported}")

    content = await file.read()
    document = Document(
        filename=file.filename or "unnamed",
        content_type=file.content_type or "application/octet-stream",
        storage_path="",
        status=DocumentStatus.QUEUED,
    )
    db.add(document)
    await db.flush()

    storage_path = store.save(document.id, document.filename, content)
    document.storage_path = storage_path
    await db.commit()
    await db.refresh(document)

    background_tasks.add_task(_run_ingestion, pipeline, document.id, Path(storage_path))
    return document


@router.get("", response_model=list[DocumentOut])
async def list_documents(db: AsyncSession = Depends(get_db)) -> list[Document]:
    result = await db.execute(select(Document).order_by(Document.created_at.desc()))
    return list(result.scalars().all())


@router.get("/{document_id}", response_model=DocumentOut)
async def get_document(document_id: str, db: AsyncSession = Depends(get_db)) -> Document:
    document = await db.get(Document, document_id)
    if document is None:
        raise HTTPException(404, "Document not found")
    return document
