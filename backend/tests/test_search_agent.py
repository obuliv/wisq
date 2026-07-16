from app.chat.search_agent import SearchAgent
from app.db.models import Document, DocumentRelationship
from app.db.session import async_session_factory
from app.ingestion.chunking import Chunk
from app.llm.interfaces import GenerationResult, Message, ToolCall
from app.rag.fakes import FakeEmbedder, InMemoryVectorStore
from app.rag.interfaces import EmbeddedChunk, Filters, ScoredChunk
from app.rag.retriever import SimpleRetriever


class ScriptedToolLLMClient:
    """Returns canned GenerationResults in sequence -- same ad hoc Scripted/Stub
    convention used for LLMClient fakes elsewhere in this test suite. Records
    every call's messages so tests can inspect what tool-result content was fed
    back to a later call."""

    def __init__(self, responses: list[GenerationResult]) -> None:
        self._responses = list(responses)
        self._call_count = 0
        self.calls: list[list[Message]] = []

    def generate_with_tools(self, messages: list[Message], tools: list) -> GenerationResult:
        self.calls.append(list(messages))
        response = self._responses[self._call_count]
        self._call_count += 1
        return response


class AlwaysToolCallLLMClient:
    """Ignores `tools` (even when empty) and always requests another tool call --
    used to exercise the max_iterations forced-termination path."""

    def generate_with_tools(self, messages: list[Message], tools: list) -> GenerationResult:
        return GenerationResult(
            tool_calls=[ToolCall(id="x", name="search_documents", arguments={"query": "q"})]
        )


class RaisingRetriever:
    def retrieve(self, query: str, top_k: int = 5, filters: Filters | None = None) -> list[ScoredChunk]:
        raise RuntimeError("boom")


def _make_retriever_with_chunks(*chunks: Chunk) -> SimpleRetriever:
    store = InMemoryVectorStore()
    for chunk in chunks:
        (vector,) = FakeEmbedder().embed([chunk.text])
        store.upsert(chunk.doc_id, [EmbeddedChunk(chunk=chunk, vector=vector)])
    return SimpleRetriever(embedder=FakeEmbedder(), vector_store=store)


def _make_retriever_with_chunk(chunk: Chunk) -> SimpleRetriever:
    return _make_retriever_with_chunks(chunk)


async def test_single_tool_call_then_final_answer():
    chunk = Chunk(
        doc_id="doc-1",
        text="PTO policy applies to all employees.",
        metadata={"is_latest": True},
    )
    retriever = _make_retriever_with_chunk(chunk)
    scripted = ScriptedToolLLMClient(
        [
            GenerationResult(
                tool_calls=[ToolCall(id="1", name="search_documents", arguments={"query": "PTO policy"})]
            ),
            GenerationResult(text="Final answer here."),
        ]
    )

    agent = SearchAgent(retriever=retriever, llm_client=scripted)
    result = await agent.run([Message(role="user", content="What is the PTO policy?")])

    assert result.final_text == "Final answer here."
    assert len(result.sources) == 1
    assert result.sources[0].chunk.id == chunk.id


async def test_overlapping_tool_calls_dedupe_by_chunk_id_keeping_max_score():
    chunk = Chunk(
        doc_id="doc-1",
        text="PTO policy applies to all employees.",
        metadata={"is_latest": True, "regions_included": [], "regions_excluded": []},
    )
    retriever = _make_retriever_with_chunk(chunk)

    scripted = ScriptedToolLLMClient(
        [
            GenerationResult(
                tool_calls=[
                    ToolCall(
                        id="1",
                        name="search_documents",
                        arguments={"query": "PTO Singapore", "geography": "Singapore"},
                    )
                ]
            ),
            GenerationResult(
                tool_calls=[
                    ToolCall(id="2", name="search_documents", arguments={"query": "PTO policy"})
                ]
            ),
            GenerationResult(text="answer"),
        ]
    )

    agent = SearchAgent(retriever=retriever, llm_client=scripted)
    result = await agent.run([Message(role="user", content="What is the PTO policy?")])

    assert result.final_text == "answer"
    assert len(result.sources) == 1  # deduped despite two overlapping tool calls
    assert result.sources[0].chunk.id == chunk.id

    expected_max_score = max(
        retriever.retrieve("PTO Singapore", filters={"is_latest": True})[0].score,
        retriever.retrieve("PTO policy", filters={"is_latest": True})[0].score,
    )
    assert result.sources[0].score == expected_max_score


async def test_max_iterations_forces_termination_without_looping_forever():
    agent = SearchAgent(
        retriever=_make_retriever_with_chunk(Chunk(doc_id="doc-1", text="irrelevant")),
        llm_client=AlwaysToolCallLLMClient(),
        max_iterations=3,
    )

    result = await agent.run([Message(role="user", content="anything")])

    assert result.final_text == ""


async def test_search_failure_is_reported_to_llm_instead_of_raising():
    scripted = ScriptedToolLLMClient(
        [
            GenerationResult(
                tool_calls=[ToolCall(id="1", name="search_documents", arguments={"query": "q"})]
            ),
            GenerationResult(text="fallback answer"),
        ]
    )

    agent = SearchAgent(retriever=RaisingRetriever(), llm_client=scripted)
    result = await agent.run([Message(role="user", content="q")])

    assert result.final_text == "fallback answer"
    assert result.sources == []


async def test_related_documents_tool_pulls_in_connected_document_and_merges_sources():
    regional_chunk = Chunk(
        doc_id="doc-regional",
        text="Local PTO policy takes precedence for APAC employees.",
        metadata={"is_latest": True},
    )
    global_chunk = Chunk(
        doc_id="doc-global",
        text="Global PTO policy applies to all employees.",
        metadata={"is_latest": True},
    )
    retriever = _make_retriever_with_chunks(regional_chunk, global_chunk)

    scripted = ScriptedToolLLMClient(
        [
            GenerationResult(
                tool_calls=[
                    ToolCall(id="1", name="search_documents", arguments={"query": "PTO policy"})
                ]
            ),
            GenerationResult(
                tool_calls=[
                    ToolCall(
                        id="2",
                        name="get_related_documents",
                        arguments={"doc_id": "doc-regional", "topic": "PTO"},
                    )
                ]
            ),
            GenerationResult(text="Regional PTO policy takes precedence."),
        ]
    )

    async with async_session_factory() as db:
        db.add(Document(id="doc-regional", filename="regional.docx", content_type="x", storage_path="x", title="APAC Benefits Handbook"))
        db.add(Document(id="doc-global", filename="global.docx", content_type="x", storage_path="x", title="Global Employee Handbook"))
        db.add(
            DocumentRelationship(
                source_doc_id="doc-regional",
                target_doc_id="doc-global",
                target_doc_ref="Global Employee Handbook",
                relation_type="precedence",
                topic="PTO",
                precedence="source_over_target",
                source_text="TAKES PRECEDENCE over any conflicting PTO provision",
            )
        )
        await db.commit()

        agent = SearchAgent(retriever=retriever, llm_client=scripted)
        result = await agent.run(
            [Message(role="user", content="What is the PTO policy for APAC?")], db=db
        )

    assert result.final_text == "Regional PTO policy takes precedence."
    source_doc_ids = {s.chunk.doc_id for s in result.sources}
    assert source_doc_ids == {"doc-regional", "doc-global"}

    final_call_messages = scripted.calls[-1]
    tool_results = [m.content for m in final_call_messages if m.role == "tool"]
    assert any("takes precedence" in content for content in tool_results)


async def test_related_documents_tool_degrades_gracefully_without_db():
    chunk = Chunk(doc_id="doc-regional", text="Local PTO policy.", metadata={"is_latest": True})
    retriever = _make_retriever_with_chunk(chunk)

    scripted = ScriptedToolLLMClient(
        [
            GenerationResult(
                tool_calls=[
                    ToolCall(id="1", name="search_documents", arguments={"query": "PTO policy"})
                ]
            ),
            GenerationResult(
                tool_calls=[
                    ToolCall(id="2", name="get_related_documents", arguments={"doc_id": "doc-regional"})
                ]
            ),
            GenerationResult(text="answer without related docs"),
        ]
    )

    agent = SearchAgent(retriever=retriever, llm_client=scripted)
    result = await agent.run([Message(role="user", content="q")])  # db omitted

    assert result.final_text == "answer without related docs"
    final_call_messages = scripted.calls[-1]
    tool_results = [m.content for m in final_call_messages if m.role == "tool"]
    assert "Related-document lookup is unavailable in this context." in tool_results
