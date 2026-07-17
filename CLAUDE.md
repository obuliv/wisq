# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A document Q&A system: upload `.docx` files, index them into a vector store for RAG,
chat against them via an LLM. API, UI, DB models, document ingestion (parsing,
metadata/relationship extraction, section-aware chunking), and chat plumbing are
real and tested; the vector store and LLM provider are intentionally unimplemented
behind interfaces (see "Stub status" below). The full design rationale lives in two
places if you need the "why" behind a decision — ask the user if you can't find them:
the original high-level design doc (tech stack choices, why no LangChain, Qdrant
capability analysis) and the ingestion/indexing deep-dive (why `unstructured`,
document metadata/versioning/relationship extraction design).

## Commands

### Backend (`backend/`, Python 3.11+, FastAPI)
```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest -q                                              # full suite, no external services needed (uses sqlite)
pytest tests/test_documents.py::test_upload_docx_gets_indexed -q   # single test

uvicorn app.main:app --reload --port 8000              # dev server; needs DATABASE_URL/QDRANT_URL reachable
```
Tests set `DATABASE_URL`/`UPLOAD_DIR` env vars in `tests/conftest.py` *before*
importing `app.main` (settings are read at import time via a module-level
`get_settings()` call in `app/db/session.py`) — keep that ordering if you add
fixtures. Tests run against sqlite+aiosqlite and the fake RAG/LLM implementations,
so they don't need postgres/qdrant/docker running. `pyproject.toml` sets
`asyncio_mode = "auto"`, so plain `async def test_...` functions work without a
`@pytest.mark.asyncio` decorator; a session-scoped autouse fixture in
`conftest.py` creates tables up front so tests that talk to the DB directly (via
`app.db.session.async_session_factory`, not the `client` fixture) don't need one.

### Frontend (`frontend/`, Vite + React + TS)
```
npm install
npm run dev       # vite dev server, proxies /api -> http://localhost:8000 (see vite.config.ts)
npm run build     # tsc -b && vite build
```

### Full stack
```
cp .env.example .env
docker-compose up --build     # postgres + qdrant + backend + frontend
docker compose config         # validate compose file without building
```

## Architecture

**Composition root:** `backend/app/dependencies.py` is the only place that wires
concrete implementations to interfaces (`Embedder`, `VectorStore`, `Retriever`,
`LLMClient`, `Chunker`, `DocumentStore`). Everything else — routers, pipeline,
orchestrator — depends only on the Protocol types (`app/rag/interfaces.py`,
`app/llm/interfaces.py`, `app/storage/interfaces.py`, `app/ingestion/chunking.py`).
To swap in a real implementation, change the one `@lru_cache`-decorated factory
function in `dependencies.py` — no other file should need to change.

**Stub status (what's real vs. placeholder):**
- Real: FastAPI app, DB models/session, `LocalDocumentStore`, `DocxLoader` (built
  on `unstructured`), `SectionAwareChunker`, `CompositeMetadataExtractor`,
  `SectionAnnotationExtractor`, versioning/reconciliation logic, API routes, SSE
  chat streaming, `ChatOrchestrator`, `SearchAgent` (the agentic tool-call loop —
  see "Retrieval" below), the RRF fusion algorithm (`app/rag/fusion.py`).
- The LLM-backed extractors (`CompositeMetadataExtractor`'s LLM half,
  `SectionAnnotationExtractor`) and `SearchAgent`'s tool-calling loop are real
  orchestration code, not stubs — they just produce mostly-empty/single-shot
  results against the current `FakeLLMClient`, since it never returns valid JSON
  and its `generate_with_tools` never requests a tool call. They're designed to
  degrade gracefully (log + skip, or terminate in one iteration) rather than fail
  when that happens; don't "fix" that by making them raise or loop. Quality of
  doc_type/effective_date/applicable_regions/relationship extraction and of the
  agentic search behavior is entirely gated on which `LLMClient` is wired.
- Placeholder-but-runnable: `InMemoryVectorStore` (`app/rag/fakes.py`) —
  deterministic, in-memory, no external API keys needed. `FakeEmbedder` and
  `FakeSparseEmbedder` (`app/rag/fakes.py`) are the default dense/sparse
  embedders but are both now swappable for real ones (see below), same pattern
  as `FakeLLMClient`.
- Not implemented yet: `app/rag/qdrant_store.py` (`QdrantVectorStore` — methods
  raise `NotImplementedError`, TODOs describe the payload/filter/fusion shape
  needed for Qdrant, including named dense+sparse vectors) — the vector store is
  always the in-memory fake regardless of provider config elsewhere.
- `app/llm/openai_client.py` (`OpenAIClient`) is a real, working `LLMClient`
  (both `generate` and `generate_with_tools`, via OpenAI's chat completions +
  function-calling API). `get_llm_client()` (`dependencies.py`) selects it via
  `LLM_PROVIDER=openai` in `.env` (needs `OPENAI_API_KEY`); default is `fake`
  (`FakeLLMClient`, no credentials/network). `tests/conftest.py` force-sets
  `LLM_PROVIDER=fake` before importing `app.main` so the suite never picks up a
  real provider from a developer's `.env` — never remove that line. `anthropic`
  is a recognized but unimplemented provider value (`get_llm_client()` raises
  `NotImplementedError` with a pointer to follow `OpenAIClient`'s pattern).
  `OPENAI_MODEL` defaults to `gpt-5.6-luna` (measured noticeably more reliable
  than `gpt-4o-mini` on multi-document precedence reasoning — see the
  Retrieval section below). It's a reasoning model, so `OpenAIClient` also
  takes `reasoning_effort` (`OPENAI_REASONING_EFFORT` in `.env`, defaults to
  `"none"`), included in every `chat.completions` call only when non-empty —
  reasoning models reject tool calls without it, but non-reasoning models like
  `gpt-4o-mini` reject the param outright if it's sent at all ("Unrecognized
  request argument"), so clear it to `""` if you switch back to one.
- `app/rag/openai_embedder.py` (`OpenAIEmbedder`) is a real, working dense
  `Embedder` (OpenAI's embeddings API, default model `text-embedding-3-small`,
  defensively re-normalized since `InMemoryVectorStore`'s scoring is a raw dot
  product). `get_embedder()` (`dependencies.py`) selects it via
  `EMBEDDING_PROVIDER=openai` in `.env` (reuses `OPENAI_API_KEY`); default is
  `fake`. `tests/conftest.py` force-sets `EMBEDDING_PROVIDER=fake` alongside
  `LLM_PROVIDER=fake` — same hermeticity rule, never remove that line either.
- `app/rag/fastembed_sparse_embedder.py` (`FastEmbedSparseEmbedder`) is a real,
  working `SparseEmbedder` — real BM25 via FastEmbed (Qdrant's own open-source
  library, runs locally via ONNX, no API key, just a one-time model download).
  `get_sparse_embedder()` (`dependencies.py`) selects it via
  `SPARSE_EMBEDDING_PROVIDER=fastembed` in `.env`; default is `fake`.
  `tests/conftest.py` force-sets `SPARSE_EMBEDDING_PROVIDER=fake` too. Note
  `SparseEmbedder` has two methods, not one: `embed` (document/passage side —
  term-frequency-weighted, length-normalized) and `embed_query` (query side —
  term presence only) — these are genuinely asymmetric for real BM25, unlike a
  dense `Embedder` where the same method suits both; `FakeSparseEmbedder`
  doesn't need the distinction and just delegates `embed_query` to `embed`.

**Two async flows, each with its own DB session lifecycle:**
- *Ingestion*: `POST /api/documents` saves the file via `DocumentStore`, creates a
  `Document` row (`status=queued`), then schedules `IngestionPipeline.run` as a
  FastAPI `BackgroundTask`. That background task opens its **own** DB session via
  `async_session_factory` directly (`app/api/routes/documents.py:_run_ingestion`)
  rather than reusing the request-scoped session, since the request session is
  closed by the time the background task runs. Pipeline sequence (`app/ingestion/pipeline.py`):
  `DocumentLoader` (by extension, via `LOADER_REGISTRY`) → metadata extraction →
  version resolution → relationship extraction → `Chunker` → `Embedder` →
  `VectorStore.upsert`. The three middle stages are wrapped to catch and log
  exceptions rather than raise — only a load/chunk/embed/upsert failure sets
  `Document.status=FAILED`; enrichment failures are logged and skipped.
- *Chat*: `POST /api/chat` returns a `StreamingResponse` of a custom SSE format
  (`event: session|token|done`, see `_sse()` in `app/api/routes/chat.py`) — **not**
  browser `EventSource`-compatible since it needs a POST body; the frontend parses
  it manually in `frontend/src/api/client.ts:streamChat`. `ChatOrchestrator.answer`
  loads history, delegates to `SearchAgent.run` (see "Retrieval" below) to get a
  final answer + merged sources, then word-chunks the answer for the SSE stream
  and persists both the user and assistant `ChatMessage` rows (with sources) after
  the stream is fully consumed. No new SSE event type was added for tool calls —
  they're invisible to the client, resolved entirely inside `SearchAgent.run`
  before any tokens are streamed.

**Document format extensibility:** adding a new format is "write a loader
implementing `DocumentLoader.load() -> list[Element]`, register it in
`app/ingestion/loaders/__init__.py`'s `LOADER_REGISTRY` by extension" — metadata
extraction/chunking/embedding/storage all operate on the loader's normalized
`Element` output (`app/ingestion/loaders/base.py`: `text`, `category`,
`heading_path`, `locator`), not the source format. `DocxLoader` gets this shape
from `unstructured.partition.docx.partition_docx` and builds `heading_path` by
walking a depth-keyed stack of `category == "Title"` elements — a future
`PdfLoader` would reuse that exact stack-walking logic against `partition_pdf`'s
output, since `unstructured` normalizes both to the same element shape.
`unstructured` types never leak past the loader — same boundary principle as not
adopting LangChain for orchestration (see the high-level design doc).

**Document metadata, versioning, and cross-document relationships**
(`app/ingestion/metadata.py`, `versioning.py`, `relationships.py`):
- `Document` carries extracted `doc_type`/`title`/`version`/`effective_date`/
  `applicable_regions`/`applicable_personnel` (filename regex extraction + one
  LLM call over the intro, composited so filename wins for title/version).
  `applicable_personnel` (`{"included": [...], "excluded": [...]}`, same shape
  as `applicable_regions`) captures which personnel categories (employees,
  contractors, ...) a document applies to as a structured fact at ingestion
  time, copied down to `regions_included`/`personnel_included`/etc. on every
  chunk and surfaced directly in `search_documents`/`get_related_documents`
  tool results (`format_document_scope`, `app/chat/formatting.py`) — so a
  question like "are contractors covered?" doesn't depend on the LLM having
  also retrieved the exact prose sentence that states it via semantic
  similarity, which vector search can easily miss.
- `document_group_key` (normalized title) + `is_latest` track which upload is the
  current version of a document family; `resolve_latest()` in `versioning.py`
  handles out-of-order uploads and records a `supersedes` `DocumentRelationship`.
- `DocumentRelationship` also models in-text cross-document references (e.g. "this
  regional handbook's PTO section takes precedence over the global handbook's") —
  extracted per flagged section (`find_candidate_sections`, keyword-gated to avoid
  whole-document LLM calls) via `SectionAnnotationExtractor`, then fuzzy-matched
  (`rapidfuzz`, via `normalize_title`) against existing `Document.title`s.
  Unresolved references (target not uploaded yet) get backfilled by
  `reconcile_relationships()` whenever a new document's title matches later.
  `_resolve_target` compares candidates by `(score, is_latest)`, not score
  alone — multiple versions of the same document family share a title and
  score identically against a reference, so a plain score comparison left the
  tie-break to whichever row the DB query happened to return first (a real bug
  found via live testing: relationships kept resolving to a superseded
  document instead of the current one). `reconcile_relationships` doesn't have
  the same issue (it only claims already-unresolved rows, never re-evaluates
  an already-resolved one against a better/newer candidate that arrives later).
- The same section-annotation pass also extracts section-scoped
  `geographic_scope` overrides (narrower than the document default), which
  `IngestionPipeline._enrich_chunk_metadata` applies to chunks by `heading_path`
  prefix match.
- `extract_and_store_relationships` also returns a `RelationshipHint` per
  outgoing relationship (relation_type/topic/target), which
  `_enrich_chunk_metadata` bakes onto **every** chunk of the source document as
  `related_documents` (not just chunks from the section the relationship was
  extracted from) — surfaced via `format_document_scope`. Fixes a real gap
  found via manual testing: a plain query like "gym benefits for Taiwan"
  retrieves the REGIONAL BENEFITS section but not the separate CONFLICTS AND
  PRECEDENCE section that says "for all other benefits, refer to the global
  handbook," so the model never learned to call `get_related_documents` and
  answered with the regional-only figure instead of resolving via the global
  handbook's "more generous benefit applies" rule. Only covers the *source*
  side (a document knows its own outgoing references at its own ingestion
  time) — an existing document later becoming the *target* of a new one's
  relationship isn't hinted retroactively; that would need the same
  `update_metadata` staleness-patching mechanism used for `is_latest`.
- `Document.default_precedence_rule` (free text, e.g. "the more generous
  benefit applies") is the same "bake onto every chunk" fix applied one level
  deeper than `related_documents`: even once a connected document gets
  checked, *that* document's own general conflict-resolution rule is only
  reliably visible this way too, not by hoping the model also retrieves its
  own CONFLICTS AND PRECEDENCE section. Extracted in the same
  `SectionAnnotationExtractor` pass as relationships/geographic_scope (not
  `LLMDocumentMetadataExtractor`, which only reads the document intro and would
  usually miss this section), set directly on `document` as a side effect of
  `extract_and_store_relationships` (same pattern as its `db.add(DocumentRelationship(...))`
  calls) rather than threaded through as a third return value.
- **`app/ingestion/field_registry.py`** is a declarative registry that drives
  the two genuinely duplicated fan-out sites for a document-level field:
  `IngestionPipeline._enrich_chunk_metadata`'s per-chunk metadata copy and
  `format_document_scope`'s LLM-visible rendering. Before this existed, adding
  a field like `default_precedence_rule` above required hand-editing ~11-13
  sites across 8 files with no shared abstraction. A new *plain* field (single
  value, no section override, e.g. `doc_type`) now needs one `PlainField(...)`
  entry plus the `Document`/`DocumentMetadata`/`schemas.py`/`client.ts`
  mirrors (4 sites); a new *scope* field (included/excluded lists, e.g.
  `applicable_regions`) needs one `SCOPE_PREFIXES` entry plus a
  `scope_chunk_fields(...)` call in `_enrich_chunk_metadata` (formatting.py
  needs no change, it already loops `SCOPE_PREFIXES` generically).
  Relationship-derived fields (`related_documents`, sourced from
  `DocumentRelationship` rather than a `Document` column) stay hand-written
  one-offs — only one example exists, so it isn't generalized. See the
  module's own top-of-file checklist docstring for the exact steps.
- `is_latest` is consumed at retrieval time via `build_search_filters`
  (`app/chat/search_tool.py`), which always bakes in `{"is_latest": True}`, not
  exposed as a tool argument. `DocumentRelationship` (cross-document precedence
  *and* version supersession — see `get_related_documents` below) is now also
  consumed at chat time, closing the original "let the LLM see both documents to
  resolve the conflict" requirement from the initial design doc.
- **Auto-expansion of related documents** (`SearchAgent._auto_expand_related`):
  relying on the model to *notice* a `related_documents` hint and *choose* to
  call `get_related_documents` was measured at only ~50% reliability in live
  testing (e.g. "gym benefits for Taiwan" — model has both figures available
  but doesn't always fetch the global handbook). Fixed by making
  `search_documents` itself auto-fetch and append connected-document content in
  the *same* tool result whenever a retrieved chunk's `related_documents` hint
  has a non-`supersedes` relation_type — removing the dependence on the model
  choosing an extra step. `supersedes` is deliberately excluded from
  auto-expansion (stays an explicit `get_related_documents` call) to avoid
  reintroducing stale-version contamination.
- **Broad/continent-level geography terms** (e.g. "Asia", "Europe") are
  stripped server-side in `build_search_filters` (`_BROAD_GEOGRAPHY_TERMS`)
  rather than trusted to the model, even though the tool description also
  tells it to omit them. Documents are scoped by country, not continent, so a
  continent-level term fuzzy-matches poorly against country lists (e.g. "Asia"
  vs. `["China", "Japan", "Taiwan"]`) and would otherwise silently exclude
  every country-specific document from results.
- **Residual limitation (as measured with `gpt-4o-mini`)**: with
  retrieval-completeness now structurally guaranteed (verified via logs — both
  documents' content is present in the tool result regardless of model
  choice), the remaining failure mode was the model's own reasoning: applying
  a `default_precedence_rule` (e.g. "the more generous benefit applies") to
  two now-visible figures was correct only ~50% of the time with
  `gpt-4o-mini`, even with an explicit worked example in the system prompt —
  a small-model reasoning-consistency ceiling rather than a missing-context
  problem. Confirmed by switching the default model to `gpt-5.6-luna` (see
  above): the same fixes/prompt, same test questions, but the precedence
  reasoning was correct 5/5 repeat runs instead of ~50%, and the
  ambiguous-geography hedge (Q8-style "Asia" question) came out cleanly
  broken down per-country instead of guessing one figure. Keep this in mind
  if `gpt-4o-mini` (or another small/non-reasoning model) is ever swapped back
  in — the retrieval side is solid, but expect the same reasoning variance.

**Retrieval: agentic search + hybrid vector search** (`app/chat/search_agent.py`,
`search_tool.py`, `relationship_tool.py`, `app/rag/fusion.py`, `app/rag/fakes.py`):
- Retrieval is not a fixed pipeline step — it's two tools the LLM invokes via
  `LLMClient.generate_with_tools` (additive alongside the original streaming
  `generate` — `extract_json` and the extractors above only ever use `generate`,
  so this didn't touch them):
  - `search_documents` (`search_tool.py`): vector search with optional
    `geography`/`year`/`doc_type` filters, `is_latest` always baked in.
  - `get_related_documents` (`relationship_tool.py`): given a `doc_id` the model
    saw in a prior `search_documents` result, queries `DocumentRelationship` for
    that document and pulls a few chunks from whatever it's connected to —
    covering both cross-document precedence/reference rules and
    `relation_type="supersedes"` (older/newer version) rows, since both live in
    the same table. `format_related_documents` produces direction- and
    relation-type-aware wording (which side wins precedence, "prefer the newer
    version" vs. "this is an older version," or an explicit "not uploaded yet"
    note for an unresolved reference) so the model can actually resolve a
    conflict rather than just seeing two documents' text with no relationship
    stated. This tool needs a DB session — `SearchAgent.run`/`_execute` are
    `async def` and take `db: AsyncSession | None` as a **call-scoped parameter**
    (not a constructor dependency, since `SearchAgent` is an `@lru_cache`
    singleton in `dependencies.py` shared across requests); if `db` is omitted
    the tool degrades gracefully with an "unavailable" tool-result string rather
    than raising, same convention as a failed `search_documents` call.
  - `geography`/`doc_type` are free-text strings extracted by an LLM at
    ingestion time (e.g. `"People's Republic of China"`), which won't always
    exact-match what a query later asks for (e.g. `"China"`) — confirmed as a
    real bug via end-to-end testing (a `geography="China"` search silently
    excluded the one document that actually covers China). Fixed via
    `rapidfuzz.fuzz.token_set_ratio` (`_FUZZY_MATCH_THRESHOLD = 80.0`, same
    algorithm/threshold as `relationships.py`'s document-title matching) in
    `_matches`/`_fuzzy_overlap`/`_fuzzy_in` (`app/rag/fakes.py`) — applies to the
    `any`/`not_any`/`any_or_empty` list predicates and to the `doc_type` scalar
    field specifically (`_FUZZY_SCALAR_FIELDS`; deliberately not every string
    field — `doc_id`/`locator`/`id` must stay exact). Qdrant's native
    `MatchAny`/`MatchValue` have no fuzzy option server-side, so
    `qdrant_store.py`'s TODOs document two real alternatives for when that's
    implemented: canonicalize these strings at ingestion time, or resolve the
    query-time term against the corpus's distinct values before building the
    Qdrant filter (same pattern as `_resolve_target`'s document-title
    resolution) — don't just port the in-memory fuzzy loop as-is.
  `SearchAgent.run` (`search_agent.py`) is a bounded loop (`max_iterations`,
  default 4): call the LLM with both tools offered, execute any tool calls,
  feed results back as `role="tool"` messages, repeat until the model answers
  with plain text (or the iteration cap forces a final answer with no tools
  offered). It's deliberately two-tool, not a generic tool-registry/agent
  framework — same no-orchestration-framework stance as the rest of this codebase.
- The model can call either tool more than once per turn (e.g. `search_documents`
  scoped to a `geography`/`year` first, then broader; or `get_related_documents`
  for more than one connected document). `SearchAgent` dedupes results across
  every call by `Chunk.id` (a stable uuid4, added specifically for this — see
  below), keeping the highest score on collision, and caps the merged source
  list at `max_sources`.
- Citations can now legitimately come from more than one document, so
  `_enrich_chunk_metadata` (`app/ingestion/pipeline.py`) also copies
  `document_title` (`document.title or document.filename`) onto every chunk's
  metadata, threaded through `_source_payload` → `SourceOut` → the frontend
  sources type → `ChatWindow.tsx`'s citations — otherwise a citation from a
  connected document would just be an unreadable UUID.
- **Hybrid dense+sparse search**: `EmbeddedChunk.sparse_vector` (a `SparseVector`
  of parallel index/value arrays) rides alongside the dense `vector`.
  `VectorStore.search` takes an optional `query_sparse_vector`; when omitted,
  behavior is unchanged (dense-only, cosine ranking). When provided,
  `InMemoryVectorStore` ranks dense and sparse candidates separately (each capped
  at `prefetch_limit`) and fuses them via Reciprocal Rank Fusion
  (`reciprocal_rank_fusion` in `fusion.py`, `k=60`) — this is the client-side
  mirror of what `QdrantVectorStore` should eventually do server-side via its
  Query API (`Prefetch` + `FusionQuery(fusion=Fusion.RRF)`, see the TODOs in
  `qdrant_store.py`). `FakeSparseEmbedder` (hashed term-frequency, no real
  IDF weighting) is the placeholder sparse embedder, same stub convention as
  `FakeEmbedder` — swap both when a real dense/sparse model is chosen.
- `Chunk` (`app/ingestion/chunking.py`) now carries a stable `id: str` (uuid4).
  This fixes a real gap (Qdrant requires a unique point id per upserted point,
  and nothing generated one before) and doubles as `SearchAgent`'s dedup key.
- Fused/RRF scores are not bounded like cosine similarity (`~0` to `~0.033` for
  two fused lists at `k=60`) — don't assume `ScoredChunk.score` is in `[0, 1]`
  downstream of a hybrid search call.
- Streaming tradeoff: once `SearchAgent` gets a final answer, `ChatOrchestrator`
  does not make a second real streaming LLM call to reproduce it — it word-chunks
  the already-materialized text instead, to avoid paying for every answer twice.
  Revisit this once a real streaming provider is wired in (see the TODO in
  `orchestrator.py`).

**DB schema bootstrap:** `Base.metadata.create_all` runs in `app/main.py`'s
`lifespan` on startup — there is no Alembic/migrations setup yet. If you add one,
remove the `create_all` call.

**Vector-store filter shape:** `Filters` (`app/rag/interfaces.py`) is
`{"field": value}` for exact match, `{"field": {"gte": ..., "lte": ...}}` for
range, `{"field": {"any": [...]}}` for list-field overlap, `{"field": {"not_any": [...]}}`
for list-field non-overlap, and `{"field": {"any_or_empty": [...]}}` for overlap-or-empty
(needed because an empty `regions_included` list means "applies everywhere", not
"applies nowhere" — a plain overlap check would wrongly exclude it). These map
directly onto Qdrant's filter DSL (`MatchValue`/`Range`/`DatetimeRange`/`MatchAny`/
`IsEmptyCondition`, see the TODOs in `qdrant_store.py` for the exact mapping).
`_matches`/`_flatten` in `app/rag/fakes.py` implement this same semantics against
a flattened view of `Chunk` (metadata plus `doc_id`/`id`/`locator`, mirroring the
real Qdrant payload shape) so behavior doesn't change when `QdrantVectorStore` is
filled in. Note `doc_id`/`id`/`locator` are top-level `Chunk` fields, not inside
`chunk.metadata` — filtering on them only works because `_flatten` merges both.
