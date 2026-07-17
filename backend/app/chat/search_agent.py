import logging
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.relationship_tool import RELATED_DOCS_TOOL, format_related_documents, load_related_context
from app.chat.search_tool import SEARCH_TOOL, build_search_filters, format_search_results
from app.llm.interfaces import LLMClient, Message, ToolCall
from app.rag.interfaces import Filters, Retriever, ScoredChunk

logger = logging.getLogger(__name__)


@dataclass
class AgenticSearchResult:
    final_text: str
    sources: list[ScoredChunk]


class SearchAgent:
    """Bounded LLM-driven tool-call loop over exactly two tools: search_documents
    and get_related_documents. NOT a generic multi-tool agent framework --
    deliberately narrow, matching this codebase's "no orchestration framework"
    stance; generalize only if/when a third tool is actually needed.

    Pure LLM+Retriever(+optional DB) logic, kept out of ChatOrchestrator so it's
    independently unit-testable without AsyncSession/ChatMessage plumbing.
    """

    def __init__(
        self,
        retriever: Retriever,
        llm_client: LLMClient,
        max_iterations: int = 4,
        max_sources: int = 10,
        related_doc_top_k: int = 3,
    ) -> None:
        self._retriever = retriever
        self._llm_client = llm_client
        self._max_iterations = max_iterations
        self._max_sources = max_sources
        self._related_doc_top_k = related_doc_top_k

    async def run(
        self,
        messages: list[Message],
        top_k: int = 5,
        filters: Filters | None = None,
        db: AsyncSession | None = None,
    ) -> AgenticSearchResult:
        working_messages = list(messages)
        merged: dict[str, ScoredChunk] = {}
        original_question = next(
            (m.content for m in reversed(messages) if m.role == "user"), ""
        )

        for iteration in range(self._max_iterations):
            is_last_iteration = iteration == self._max_iterations - 1
            tools = [] if is_last_iteration else [SEARCH_TOOL, RELATED_DOCS_TOOL]
            try:
                result = self._llm_client.generate_with_tools(working_messages, tools=tools)
            except Exception:  # noqa: BLE001 - an LLM outage shouldn't kill the chat turn/SSE stream
                logger.exception("LLM call failed during agentic search loop")
                return self._finish(
                    "Sorry, I couldn't reach the language model to answer this question. "
                    "Please try again shortly.",
                    merged,
                )

            if not result.tool_calls:
                return self._finish(result.text or "", merged)

            working_messages.append(
                Message(role="assistant", content="", tool_calls=result.tool_calls)
            )
            for call in result.tool_calls:
                content = await self._execute(call, merged, top_k, filters, db, original_question)
                working_messages.append(
                    Message(role="tool", content=content, tool_call_id=call.id)
                )

        # Exhausted max_iterations without a final answer (only reachable if the
        # model keeps requesting tool calls even on the forced tools=[] turn) --
        # terminate rather than loop forever, with whatever context was gathered.
        return self._finish("", merged)

    async def _execute(
        self,
        call: ToolCall,
        merged: dict[str, ScoredChunk],
        top_k: int,
        caller_filters: Filters | None,
        db: AsyncSession | None,
        original_question: str,
    ) -> str:
        if call.name == SEARCH_TOOL.name:
            return await self._execute_search(call, merged, top_k, caller_filters, db, original_question)
        if call.name == RELATED_DOCS_TOOL.name:
            return await self._execute_related(call, merged, db, original_question)
        return f"Tool call failed: unknown tool '{call.name}'."

    async def _execute_search(
        self,
        call: ToolCall,
        merged: dict[str, ScoredChunk],
        top_k: int,
        caller_filters: Filters | None,
        db: AsyncSession | None,
        original_question: str,
    ) -> str:
        try:
            call_filters = build_search_filters(call.arguments)
            if caller_filters:
                call_filters = {**call_filters, **caller_filters}
            query = call.arguments.get("query", "")
            scored = self._retriever.retrieve(query, top_k=top_k, filters=call_filters)
        except Exception as exc:  # noqa: BLE001 - a failed search shouldn't kill the chat turn
            logger.exception("search_documents tool call failed")
            return f"Search failed: {exc}"

        self._merge(scored, merged)
        result_text = format_search_results(scored)

        if db is not None:
            expansion = await self._auto_expand_related(scored, merged, db, query or original_question)
            if expansion:
                result_text += "\n\n" + expansion

        return result_text

    async def _auto_expand_related(
        self,
        scored: list[ScoredChunk],
        merged: dict[str, ScoredChunk],
        db: AsyncSession,
        fallback_query: str,
    ) -> str:
        """Whenever a retrieved chunk's document has a stated non-version
        relationship (precedence/reference/supplement), automatically pull in
        the connected document's content in this SAME tool result -- rather
        than relying on the model to notice the related_documents hint and
        choose to make a separate get_related_documents call, which real
        testing showed was correct only ~50% of the time (the model doesn't
        reliably choose the extra step). `supersedes` is deliberately excluded:
        auto-including a superseded document's content by default would
        reintroduce the stale-version contamination bug fixed earlier --
        reaching an older version should stay an explicit, LLM-decided
        get_related_documents call.
        """
        seen_doc_ids: set[str] = set()
        sections: list[str] = []
        for s in scored:
            doc_id = s.chunk.doc_id
            if doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)

            related = s.chunk.metadata.get("related_documents") or []
            if not any(r.get("relation_type") != "supersedes" for r in related):
                continue

            try:
                contexts = await load_related_context(
                    db, self._retriever, doc_id, None, fallback_query, self._related_doc_top_k
                )
            except Exception:  # noqa: BLE001 - a failed auto-expansion shouldn't kill the chat turn
                logger.exception("Auto-expanding related documents failed for doc_id=%s", doc_id)
                continue

            contexts = [c for c in contexts if c.relationship.relation_type != "supersedes"]
            if not contexts:
                continue

            for ctx in contexts:
                self._merge(ctx.chunks, merged)
            sections.append(format_related_documents(doc_id, contexts))

        if not sections:
            return ""
        return (
            "Automatically included related document context (the retrieved "
            "document above has a stated relationship with another document):"
            "\n\n" + "\n\n---\n\n".join(sections)
        )

    async def _execute_related(
        self,
        call: ToolCall,
        merged: dict[str, ScoredChunk],
        db: AsyncSession | None,
        original_question: str,
    ) -> str:
        if db is None:
            return "Related-document lookup is unavailable in this context."
        doc_id = call.arguments.get("doc_id", "")
        try:
            contexts = await load_related_context(
                db,
                self._retriever,
                doc_id,
                call.arguments.get("topic"),
                original_question,
                self._related_doc_top_k,
            )
        except Exception as exc:  # noqa: BLE001 - a failed lookup shouldn't kill the chat turn
            logger.exception("get_related_documents tool call failed")
            return f"Related-document lookup failed: {exc}"

        for ctx in contexts:
            self._merge(ctx.chunks, merged)
        return format_related_documents(doc_id, contexts)

    @staticmethod
    def _merge(scored: list[ScoredChunk], merged: dict[str, ScoredChunk]) -> None:
        for s in scored:
            existing = merged.get(s.chunk.id)
            if existing is None or s.score > existing.score:
                merged[s.chunk.id] = s

    def _finish(self, final_text: str, merged: dict[str, ScoredChunk]) -> AgenticSearchResult:
        sources = sorted(merged.values(), key=lambda s: s.score, reverse=True)[: self._max_sources]
        return AgenticSearchResult(final_text=final_text, sources=sources)
