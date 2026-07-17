from collections.abc import AsyncIterator

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.search_agent import SearchAgent
from app.db.models import ChatMessage
from app.llm.interfaces import Message
from app.rag.interfaces import Filters, ScoredChunk

_SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions using only the provided context. "
    "Use the search_documents tool to find relevant passages before answering -- call "
    "it more than once with different filters (e.g. a specific geography/year first, "
    "then broader) if the first search doesn't have the answer. A second tool, "
    "get_related_documents, looks up other documents connected to one you've already "
    "found (by the doc_id shown in a search_documents result) -- including cross-document "
    "precedence/conflict rules and older/newer versions of the same document. Each search "
    "result states the source document's scope: which regions and personnel categories "
    "(e.g. employees vs. contractors) it applies to, its effective_date, and a "
    "related_documents list when it has stated relationships with other documents (topic + "
    "relation_type, e.g. a regional handbook referencing the global handbook for topics it "
    "doesn't itself govern). Call get_related_documents whenever related_documents lists a "
    "topic connected to the question, a passage mentions another document by name, states a "
    "rule about which document governs a topic, or you're unsure whether the passage you're "
    "citing is the current version -- don't assume the single retrieved document has the "
    "full picture just because it answers the question on its face. Also use effective_date "
    "and scope to judge whether a retrieved document actually covers the situation asked "
    "about (region, personnel category, year), and search again (different filters, or a "
    "different/related document) if it doesn't. If the documents don't contain the answer, "
    "say you don't know -- never state a figure as applying to a year, region, or version "
    "that isn't confirmed by a retrieved passage."
)


class ChatOrchestrator:
    """Wires SearchAgent (LLM-driven retrieval) + persistence together for a
    single chat turn. Prompt construction here is intentionally minimal -- the
    LLM deep-dive owns prompt engineering, history truncation, etc."""

    def __init__(self, search_agent: SearchAgent) -> None:
        self._search_agent = search_agent

    async def answer(
        self,
        db: AsyncSession,
        session_id: str,
        user_message: str,
        top_k: int = 5,
        filters: Filters | None = None,
    ) -> AsyncIterator[str]:
        history = await self._load_history(db, session_id)
        messages = [
            Message(role="system", content=_SYSTEM_PROMPT),
            *history,
            Message(role="user", content=user_message),
        ]

        db.add(ChatMessage(session_id=session_id, role="user", content=user_message))

        result = await self._search_agent.run(messages, top_k=top_k, filters=filters, db=db)

        chunks: list[str] = []
        for delta in self._chunk_words(result.final_text):
            chunks.append(delta)
            yield delta

        db.add(
            ChatMessage(
                session_id=session_id,
                role="assistant",
                content="".join(chunks),
                sources=[self._source_payload(s) for s in result.sources],
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

    @staticmethod
    def _chunk_words(text: str) -> list[str]:
        # SearchAgent.run materializes the full answer (see its docstring/design
        # notes on the streaming-vs-tool-calls tradeoff) -- word-chunk it here so
        # the SSE contract (token-by-token) is unchanged for callers. TODO: once a
        # real streaming provider is wired in, revisit whether a true incremental
        # stream is worth a second generation call for turns with no tool calls.
        if not text:
            return []
        words = text.split(" ")
        return [word + (" " if i < len(words) - 1 else "") for i, word in enumerate(words)]

    @staticmethod
    def _source_payload(scored: ScoredChunk) -> dict:
        return {
            "id": scored.chunk.id,
            "doc_id": scored.chunk.doc_id,
            "document_title": scored.chunk.metadata.get("document_title"),
            "locator": scored.chunk.locator,
            "text": scored.chunk.text,
            "score": scored.score,
        }
