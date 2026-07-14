from pathlib import Path
from typing import Protocol


class DocumentStore(Protocol):
    """Persists the raw uploaded file. Local disk today; swap for an S3/MinIO-backed
    implementation later without touching ingestion or API code."""

    def save(self, doc_id: str, filename: str, content: bytes) -> str:
        """Persist the file and return a storage path/key to record on the Document row."""
        ...

    def load(self, storage_path: str) -> Path:
        """Return a local filesystem path to the file's bytes (downloading if needed)."""
        ...
