from dataclasses import dataclass, field
from typing import Protocol

from app.ingestion.loaders.base import Element


@dataclass
class Chunk:
    doc_id: str
    text: str
    locator: str | None = None
    metadata: dict = field(default_factory=dict)


class Chunker(Protocol):
    def split(self, doc_id: str, elements: list[Element]) -> list[Chunk]: ...


class SectionAwareChunker:
    """Groups elements under their nearest heading into one chunk per section,
    splitting further only when a section's content exceeds max_chars. Each
    chunk's `metadata["heading_path"]` carries the section/subsection breadcrumb
    from the loader, so retrieval/citation stays section-aware. Operates purely
    on our normalized `Element` type, so it's identical for docx and (later) pdf.
    """

    def __init__(self, max_chars: int = 1500) -> None:
        self._max_chars = max_chars

    def split(self, doc_id: str, elements: list[Element]) -> list[Chunk]:
        chunks: list[Chunk] = []
        buffer: list[str] = []
        buffer_len = 0
        has_content = False  # True once buffer holds more than just its heading
        heading_path: list[str] = []
        locator: str | None = None

        def flush() -> None:
            nonlocal buffer, buffer_len, has_content
            # A title with no content under it (e.g. the document's own root
            # title, immediately followed by the first real section) isn't a
            # useful standalone chunk -- drop it rather than emit an embedding
            # for a bare heading.
            if has_content:
                text = "\n".join(buffer).strip()
                if text:
                    chunks.append(
                        Chunk(
                            doc_id=doc_id,
                            text=text,
                            locator=locator,
                            metadata={"heading_path": list(heading_path)},
                        )
                    )
            buffer = []
            buffer_len = 0
            has_content = False

        for element in elements:
            if element.category == "Title":
                flush()
                heading_path = element.heading_path
                locator = element.locator
                buffer = [element.text]
                buffer_len = len(element.text)
                continue

            if buffer_len + len(element.text) > self._max_chars and buffer:
                flush()
                # Re-seed the continuation chunk with the (sub)heading so it's
                # still self-describing on its own.
                heading_text = heading_path[-1] if heading_path else None
                buffer = [heading_text] if heading_text else []
                buffer_len = len(heading_text) if heading_text else 0
                locator = element.locator

            buffer.append(element.text)
            buffer_len += len(element.text)
            has_content = True

        flush()
        return chunks
