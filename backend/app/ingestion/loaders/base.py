from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass
class Element:
    """One structural unit of extracted content, normalized across source formats
    (docx, and later pdf) so everything downstream -- metadata/relationship
    extraction, chunking -- is format-agnostic.

    `heading_path` is the breadcrumb of section/subsection headings this element
    is nested under (e.g. ["Conflicts and Precedence"]); for a Title-category
    element itself, the path includes its own text as the last entry.
    """

    text: str
    category: str  # "Title" | "NarrativeText" | "ListItem" | "Table" | ...
    heading_path: list[str] = field(default_factory=list)
    locator: str | None = None  # e.g. "page 2, element 14"
    metadata: dict = field(default_factory=dict)


class DocumentLoader(Protocol):
    """One implementation per source format. Add a new format by writing a loader
    and registering it in LOADER_REGISTRY — nothing downstream (metadata
    extraction, chunking, embedding, storage) needs to change."""

    def load(self, file_path: Path) -> list[Element]: ...


# Keyed by lowercase file extension (including the dot), e.g. ".docx".
LOADER_REGISTRY: dict[str, DocumentLoader] = {}


def get_loader(extension: str) -> DocumentLoader:
    loader = LOADER_REGISTRY.get(extension.lower())
    if loader is None:
        raise ValueError(f"No document loader registered for extension {extension!r}")
    return loader
