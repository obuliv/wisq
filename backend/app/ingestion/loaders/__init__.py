from app.ingestion.loaders.base import LOADER_REGISTRY, DocumentLoader, RawSection
from app.ingestion.loaders.docx_loader import DocxLoader

LOADER_REGISTRY[".docx"] = DocxLoader()

__all__ = ["DocumentLoader", "RawSection", "LOADER_REGISTRY", "DocxLoader"]
