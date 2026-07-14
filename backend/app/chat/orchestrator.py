from collections.abc import AsyncIterator

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ChatMessage
from app.llm.interfaces import LLMClient, Message
from app.rag.interfaces import Filters, Retriever, ScoredChunk

_SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions using only the provided context. "
    "If the context doesn't contain the answer, say you don't know."
)


class ChatOrchestrator:
    """Wires Retriever (RAG) + LLMClient together for a single chat turn and persists
    the exchange. Prompt construction here is intentionally minimal -- the LLM
    deep-dive owns prompt engineering, history truncation, etc."""

    def __init__(self, retriever: Retriever, llm_client: LLMClient) -> None:
        self._retriever = retriever
        self._llm_client = llm_client

    async def answer(
        self,
        db: AsyncSession,
        session_id: str,
        user_message: str,
        top_k: int = 5,
        filters: Filters | None = None,
    ) -> AsyncIterator[str]:
        history = await self._load_history(db, session_id)
        sources = self._retriever.retrieve(user_message, top_k=top_k, filters=filters)
        messages = self._build_messages(history, sources, user_message)

        db.add(ChatMessage(session_id=session_id, role="user", content=user_message))

        chunks: list[str] = []
        for delta in self._llm_client.generate(messages, stream=True):
            chunks.append(delta)
            yield delta

        db.add(
            ChatMessage(
                session_id=session_id,
                role="assistant",
                content="".join(chunks),
                sources=[self._source_payload(s) for s in sources],
            )
        )
        await db.commit()

    async def _load_history(self, db: AsyncSession, session_id: str) -> list[Message]:
        result = await db.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at)
        )
        return [Message(role=m.role, content=m.content) for m in result.scalars().all()]

    def _build_messages(
        self, history: list[Message], sources: list[ScoredChunk], user_message: str
    ) -> list[Message]:
        context = "\n\n".join(f"- {s.chunk.text}" for s in sources) or "(no matching context found)"
        system = Message(role="system", content=f"{_SYSTEM_PROMPT}\n\nContext:\n{context}")
        return [system, *history, Message(role="user", content=user_message)]

    @staticmethod
    def _source_payload(scored: ScoredChunk) -> dict:
        return {
            "doc_id": scored.chunk.doc_id,
            "locator": scored.chunk.locator,
            "text": scored.chunk.text,
            "score": scored.score,
        }
