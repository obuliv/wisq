from pathlib import Path

from app.storage.interfaces import DocumentStore


class LocalDocumentStore(DocumentStore):
    """Stores uploaded files on a local (volume-mounted) directory, one subfolder per doc_id."""

    def __init__(self, root_dir: str) -> None:
        self._root = Path(root_dir)
        self._root.mkdir(parents=True, exist_ok=True)

    def save(self, doc_id: str, filename: str, content: bytes) -> str:
        doc_dir = self._root / doc_id
        doc_dir.mkdir(parents=True, exist_ok=True)
        path = doc_dir / filename
        path.write_bytes(content)
        return str(path)

    def load(self, storage_path: str) -> Path:
        return Path(storage_path)
