"""Qdrant-backed VectorStore -- SKELETON ONLY, fill in per the RAG deep-dive.

Matches the VectorStore protocol in interfaces.py so it's a drop-in replacement for
InMemoryVectorStore in dependencies.py once implemented. Left unimplemented on
purpose per the RAG design doc: embedding model / chunking / hybrid-search choices
need to land first and will shape exactly what payload fields get indexed.
"""

from qdrant_client import QdrantClient

from app.rag.interfaces import EmbeddedChunk, Filters, ScoredChunk, SparseVector, Vector


class QdrantVectorStore:
    def __init__(self, url: str, collection: str) -> None:
        self._client = QdrantClient(url=url)
        self._collection = collection

    def upsert(self, doc_id: str, chunks: list[EmbeddedChunk]) -> None:
        # TODO: ensure_collection (named vectors: "dense" sized from the dense
        # Embedder with distance=Cosine, "sparse" as a sparse vector config once
        # hybrid search lands), then delete-by-filter(doc_id=doc_id) followed by
        # self._client.upsert(collection_name=self._collection, points=[...]),
        # with point id = chunk.id (uuid4, stable per ingestion run -- not stable
        # across re-ingestion, hence delete-then-insert rather than a merge-upsert)
        # and payload = {"doc_id": chunk.doc_id, "id": chunk.id, "text": chunk.text,
        # "locator": chunk.locator, **chunk.metadata} so effective_date/doc_type/
        # regions_included/regions_excluded filters (see design doc) work. Add a
        # payload index on frequently-filtered fields (effective_date, doc_type,
        # regions_included, regions_excluded, is_latest) for performance.
        raise NotImplementedError

    def update_metadata(self, doc_id: str, updates: dict) -> None:
        # TODO: self._client.set_payload(collection_name=self._collection,
        # payload=updates, points=Filter(must=[FieldCondition(key="doc_id",
        # match=MatchValue(value=doc_id))])) -- a payload-only update, no
        # re-embedding/re-upsert needed. Used by IngestionPipeline._resolve_version
        # to correct a superseded sibling's already-indexed chunks' is_latest
        # flag (see versioning.py's resolve_latest returning changed siblings).
        raise NotImplementedError

    def search(
        self,
        query_vector: Vector,
        query_sparse_vector: SparseVector | None = None,
        top_k: int = 5,
        filters: Filters | None = None,
    ) -> list[ScoredChunk]:
        # TODO: translate `filters` into qdrant_client.models.Filter:
        #   {"field": scalar}                  -> FieldCondition(match=MatchValue(value=scalar))          [must]
        #   {"field": {"gte": x, "lte": y}}     -> FieldCondition(range=Range(gte=x, lte=y))                [must]
        #   {"field": {"any": [...]}}           -> FieldCondition(match=MatchAny(any=[...]))                [must]
        #   {"field": {"not_any": [...]}}       -> FieldCondition(match=MatchAny(any=[...]))                [must_not]
        #   {"field": {"any_or_empty": [...]}}  -> Filter(should=[FieldCondition(match=MatchAny(any=[...])),
        #                                                          IsEmptyCondition(is_empty=PayloadField(key=field))])
        #                                          nested inside the outer `must` list (Qdrant permits a Filter
        #                                          object anywhere a condition is expected).
        # Then call self._client.search(...); map hits back to ScoredChunk via
        # point.id (== chunk.id) and payload. Once hybrid search lands, this
        # becomes a Query API call with Prefetch(using="dense")/Prefetch(using="sparse")
        # + FusionQuery(fusion=Fusion.RRF) instead of a single vector search.
        #
        # IMPORTANT DIVERGENCE from InMemoryVectorStore (rag/fakes.py): region/
        # personnel/doc_type values are free text an LLM extracted at ingestion
        # time (e.g. "People's Republic of China"), which won't always exact-
        # match what a query later asks for (e.g. "China") -- the in-memory fake
        # handles this with rapidfuzz fuzzy matching (token_set_ratio) inside
        # _matches, but Qdrant's native MatchAny/MatchValue are exact-only, no
        # fuzzy option server-side. A real implementation needs one of:
        #   (a) canonicalize these strings into a fixed form at ingestion time
        #       (a small alias/gazetteer table, e.g. "China" -> canonical key),
        #       so exact MatchAny works because both sides already agree; or
        #   (b) resolve the query-time term against the distinct values actually
        #       present in the corpus via fuzzy matching BEFORE constructing the
        #       Qdrant filter (same pattern as relationships.py's _resolve_target
        #       resolving a free-text document reference against Document.title
        #       rows), then pass the resolved canonical value(s) into MatchAny.
        # Don't just port the in-memory fuzzy-matching loop as-is -- it doesn't
        # translate to a single Qdrant filter condition.
        raise NotImplementedError
