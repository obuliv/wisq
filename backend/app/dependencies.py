"""Composition root: wires concrete implementations behind each interface.

Everything below the "swap point" comments defaults to an in-memory fake so the API
is runnable end-to-end without a real embedding model, Qdrant collection, or LLM
provider. Replace those lines (and fill in rag/qdrant_store.py + a real LLMClient)
when the RAG/LLM deep-dive implementations are ready -- nothing else in this file
or the routers needs to change.
"""

from functools import lru_cache

from app.chat.orchestrator import ChatOrchestrator
from app.config import get_settings
from app.ingestion.chunking import Chunker, SectionAwareChunker
from app.ingestion.metadata import CompositeMetadataExtractor, DocumentMetadataExtractor
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.relationships import SectionAnnotationExtractor
from app.llm.fakes import FakeLLMClient
from app.llm.interfaces import LLMClient
from app.rag.fakes import FakeEmbedder, InMemoryVectorStore
from app.rag.interfaces import Embedder, Retriever, VectorStore
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
    # Swap point: replace with a real Embedder (OpenAI / sentence-transformers / ...).
    return FakeEmbedder()


@lru_cache
def get_vector_store() -> VectorStore:
    # Swap point: replace with QdrantVectorStore(url=..., collection=...) once
    # rag/qdrant_store.py is implemented.
    return InMemoryVectorStore()


@lru_cache
def get_retriever() -> Retriever:
    return SimpleRetriever(embedder=get_embedder(), vector_store=get_vector_store())


@lru_cache
def get_llm_client() -> LLMClient:
    # Swap point: replace with a real provider-backed LLMClient.
    return FakeLLMClient()


@lru_cache
def get_ingestion_pipeline() -> IngestionPipeline:
    return IngestionPipeline(
        chunker=get_chunker(),
        embedder=get_embedder(),
        vector_store=get_vector_store(),
        metadata_extractor=get_metadata_extractor(),
        relationship_extractor=get_relationship_extractor(),
    )


@lru_cache
def get_chat_orchestrator() -> ChatOrchestrator:
    return ChatOrchestrator(retriever=get_retriever(), llm_client=get_llm_client())
