from datetime import datetime

from pydantic import BaseModel


class DocumentOut(BaseModel):
    id: str
    filename: str
    content_type: str
    status: str
    error_message: str | None = None
    doc_metadata: dict | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str
    top_k: int = 5
    filters: dict | None = None


class SourceOut(BaseModel):
    doc_id: str
    locator: str | None = None
    text: str
    score: float


class ChatMessageOut(BaseModel):
    id: str
    role: str
    content: str
    sources: list[dict] | None = None
    created_at: datetime

    model_config = {"from_attributes": True}
