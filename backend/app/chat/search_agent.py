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
            return self._execute_search(call, merged, top_k, caller_filters)
        if call.name == RELATED_DOCS_TOOL.name:
            return await self._execute_related(call, merged, db, original_question)
        return f"Tool call failed: unknown tool '{call.name}'."

    def _execute_search(
        self, call: ToolCall, merged: dict[str, ScoredChunk], top_k: int, caller_filters: Filters | None
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
        return format_search_results(scored)

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
