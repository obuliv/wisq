import uuid
from datetime import date, datetime
from enum import StrEnum

from sqlalchemy import JSON, Boolean, Date, DateTime, Float, ForeignKey, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


def _uuid() -> str:
    return str(uuid.uuid4())


class DocumentStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class Document(Base):
    """Source-of-truth record for an uploaded document. Chunk content/embeddings
    live in the vector store; this table tracks the file itself and ingestion state."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    content_type: Mapped[str] = mapped_column(String, nullable=False)
    storage_path: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, default=DocumentStatus.QUEUED, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Extracted document-level metadata (filename parsing + LLM extraction over
    # the document body). See app/ingestion/metadata.py.
    doc_type: Mapped[str | None] = mapped_column(String, nullable=True)
    version: Mapped[str | None] = mapped_column(String, nullable=True)
    title: Mapped[str | None] = mapped_column(String, nullable=True)  # canonical display title
    effective_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    # {"included": [...], "excluded": [...]} -- whole-document default scope.
    applicable_regions: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # {"included": [...], "excluded": [...]} -- which personnel categories
    # (employees, contractors, ...) this document applies to, same shape.
    applicable_personnel: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Versioning (app/ingestion/versioning.py): documents sharing the same
    # normalized title are grouped, and only the newest is is_latest=True.
    document_group_key: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    is_latest: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Extensible catch-all for anything not promoted to its own column.
    doc_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class DocumentRelationship(Base):
    """A cross-document relationship stated in a document's text (e.g. a regional
    handbook declaring its PTO section takes precedence over the global handbook's).
    Also doubles as the audit trail for version supersession (relation_type
    "supersedes"). See app/ingestion/relationships.py and versioning.py."""

    __tablename__ = "document_relationships"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    source_doc_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), nullable=False)
    # Nullable + raw ref string: the mentioned document may not be uploaded yet.
    target_doc_id: Mapped[str | None] = mapped_column(ForeignKey("documents.id"), nullable=True)
    target_doc_ref: Mapped[str] = mapped_column(String, nullable=False)
    relation_type: Mapped[str] = mapped_column(String, nullable=False)  # precedence | reference | supplement | supersedes
    topic: Mapped[str | None] = mapped_column(String, nullable=True)  # e.g. "PTO"
    precedence: Mapped[str | None] = mapped_column(String, nullable=True)  # source_over_target | target_over_source
    source_text: Mapped[str] = mapped_column(Text, nullable=False)
    source_locator: Mapped[str | None] = mapped_column(String, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    messages: Mapped[list["ChatMessage"]] = relationship(
        back_populates="session", order_by="ChatMessage.created_at"
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(ForeignKey("chat_sessions.id"), nullable=False)
    role: Mapped[str] = mapped_column(String, nullable=False)  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # Chunks used to ground an assistant reply, for citation display.
    sources: Mapped[list[dict] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    session: Mapped[ChatSession] = relationship(back_populates="messages")
