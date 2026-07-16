"""Composition root: wires concrete implementations behind each interface.

Everything below the "swap point" comments defaults to an in-memory fake so the API
is runnable end-to-end without a real embedding model, Qdrant collection, or LLM
provider. Replace those lines (and fill in rag/qdrant_store.py + a real LLMClient)
when the RAG/LLM deep-dive implementations are ready -- nothing else in this file
or the routers needs to change.
"""

from functools import lru_cache

from app.chat.orchestrator import ChatOrchestrator
from app.chat.search_agent import SearchAgent
from app.config import get_settings
from app.ingestion.chunking import Chunker, SectionAwareChunker
from app.ingestion.metadata import CompositeMetadataExtractor, DocumentMetadataExtractor
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.relationships import SectionAnnotationExtractor
from app.llm.fakes import FakeLLMClient
from app.llm.interfaces import LLMClient
from app.llm.openai_client import OpenAIClient
from app.rag.fakes import FakeEmbedder, FakeSparseEmbedder, InMemoryVectorStore
from app.rag.fastembed_sparse_embedder import FastEmbedSparseEmbedder
from app.rag.interfaces import Embedder, Retriever, SparseEmbedder, VectorStore
from app.rag.openai_embedder import OpenAIEmbedder
from app.rag.retriever import SimpleRetriever
from app.storage.interfaces import DocumentStore
from app.storage.local_storage import LocalDocumentStore


@lru_cache
def get_document_store() -> DocumentStore:
    return LocalDocumentStore(get_settings().upload_dir)


@lru_cache
def get_chunker() -> Chunker:
    return SectionAwareChunker()


@lru_cache
def get_metadata_extractor() -> DocumentMetadataExtractor:
    return CompositeMetadataExtractor(llm_client=get_llm_client())


@lru_cache
def get_relationship_extractor() -> SectionAnnotationExtractor:
    return SectionAnnotationExtractor(llm_client=get_llm_client())


@lru_cache
def get_embedder() -> Embedder:
    # Swap point: selected via EMBEDDING_PROVIDER in .env. "fake" (default) needs
    # no credentials/network access -- tests force this regardless of .env (see
    # tests/conftest.py) so the suite stays hermetic.
    settings = get_settings()
    if settings.embedding_provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("EMBEDDING_PROVIDER=openai requires OPENAI_API_KEY to be set")
        return OpenAIEmbedder(api_key=settings.openai_api_key, model=settings.openai_embedding_model)
    return FakeEmbedder()


@lru_cache
def get_sparse_embedder() -> SparseEmbedder:
    # Swap point: selected via SPARSE_EMBEDDING_PROVIDER in .env. "fake"
    # (default) needs no downloads/network access -- tests force this
    # regardless of .env (see tests/conftest.py) so the suite stays hermetic.
    settings = get_settings()
    if settings.sparse_embedding_provider == "fastembed":
        return FastEmbedSparseEmbedder(model_name=settings.bm25_model)
    return FakeSparseEmbedder()


@lru_cache
def get_vector_store() -> VectorStore:
    # Swap point: replace with QdrantVectorStore(url=..., collection=...) once
    # rag/qdrant_store.py is implemented.
    return InMemoryVectorStore()


@lru_cache
def get_retriever() -> Retriever:
    return SimpleRetriever(
        embedder=get_embedder(),
        vector_store=get_vector_store(),
        sparse_embedder=get_sparse_embedder(),
    )


@lru_cache
def get_llm_client() -> LLMClient:
    # Swap point: selected via LLM_PROVIDER in .env. "fake" (default) needs no
    # credentials/network access -- tests force this regardless of .env (see
    # tests/conftest.py) so the suite stays hermetic. To add another provider,
    # implement LLMClient in a sibling module (see OpenAIClient's docstring) and
    # add one branch here.
    settings = get_settings()
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("LLM_PROVIDER=openai requires OPENAI_API_KEY to be set")
        return OpenAIClient(api_key=settings.openai_api_key, model=settings.openai_model)
    if settings.llm_provider == "anthropic":
        raise NotImplementedError(
            "LLM_PROVIDER=anthropic is not implemented yet -- add app/llm/anthropic_client.py "
            "implementing LLMClient (generate + generate_with_tools) and wire it in above, "
            "the same pattern OpenAIClient follows."
        )
    return FakeLLMClient()


@lru_cache
def get_ingestion_pipeline() -> IngestionPipeline:
    return IngestionPipeline(
        chunker=get_chunker(),
        embedder=get_embedder(),
        vector_store=get_vector_store(),
        metadata_extractor=get_metadata_extractor(),
        relationship_extractor=get_relationship_extractor(),
        sparse_embedder=get_sparse_embedder(),
    )


@lru_cache
def get_search_agent() -> SearchAgent:
    return SearchAgent(retriever=get_retriever(), llm_client=get_llm_client())


@lru_cache
def get_chat_orchestrator() -> ChatOrchestrator:
    return ChatOrchestrator(search_agent=get_search_agent())
