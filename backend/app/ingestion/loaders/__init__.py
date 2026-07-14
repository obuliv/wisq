from app.ingestion.loaders.base import LOADER_REGISTRY, DocumentLoader, Element
from app.ingestion.loaders.docx_loader import DocxLoader

LOADER_REGISTRY[".docx"] = DocxLoader()

__all__ = ["DocumentLoader", "Element", "LOADER_REGISTRY", "DocxLoader"]
