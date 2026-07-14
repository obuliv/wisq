from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass
class RawSection:
    """One logical unit of extracted text (e.g. a paragraph or heading block),
    normalized across source formats before chunking."""

    text: str
    locator: str | None = None  # e.g. "paragraph 12", "heading: Introduction"
    metadata: dict = field(default_factory=dict)


class DocumentLoader(Protocol):
    """One implementation per source format. Add a new format by writing a loader
    and registering it in LOADER_REGISTRY — nothing downstream (chunking, embedding,
    storage) needs to change."""

    def load(self, file_path: Path) -> list[RawSection]: ...


# Keyed by lowercase file extension (including the dot), e.g. ".docx".
LOADER_REGISTRY: dict[str, DocumentLoader] = {}


def get_loader(extension: str) -> DocumentLoader:
    loader = LOADER_REGISTRY.get(extension.lower())
    if loader is None:
        raise ValueError(f"No document loader registered for extension {extension!r}")
    return loader
