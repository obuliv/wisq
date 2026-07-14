from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.orchestrator import ChatOrchestrator
from app.db.models import ChatMessage, ChatSession
from app.db.session import get_db
from app.dependencies import get_chat_orchestrator
from app.schemas import ChatMessageOut, ChatRequest

router = APIRouter(prefix="/api/chat", tags=["chat"])


def _sse(event: str, data: str) -> str:
    # SSE frames are newline-delimited; escape embedded newlines per the spec.
    payload = data.replace("\n", "\\n")
    return f"event: {event}\ndata: {payload}\n\n"


async def _stream(
    orchestrator: ChatOrchestrator,
    db: AsyncSession,
    session_id: str,
    request: ChatRequest,
) -> AsyncIterator[str]:
    yield _sse("session", session_id)
    async for delta in orchestrator.answer(
        db, session_id, request.message, top_k=request.top_k, filters=request.filters
    ):
        yield _sse("token", delta)
    yield _sse("done", "")


@router.post("")
async def chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    orchestrator: ChatOrchestrator = Depends(get_chat_orchestrator),
) -> StreamingResponse:
    session_id = request.session_id
    if session_id is None:
        session = ChatSession()
        db.add(session)
        await db.commit()
        session_id = session.id
    elif await db.get(ChatSession, session_id) is None:
        raise HTTPException(404, "Chat session not found")

    return StreamingResponse(
        _stream(orchestrator, db, session_id, request), media_type="text/event-stream"
    )


@router.get("/sessions/{session_id}/messages", response_model=list[ChatMessageOut])
async def list_messages(session_id: str, db: AsyncSession = Depends(get_db)) -> list[ChatMessage]:
    if await db.get(ChatSession, session_id) is None:
        raise HTTPException(404, "Chat session not found")
    result = await db.execute(
        select(ChatMessage).where(ChatMessage.session_id == session_id).order_by(ChatMessage.created_at)
    )
    return list(result.scalars().all())
