from dataclasses import dataclass, field
from typing import Protocol

from app.ingestion.loaders.base import RawSection


@dataclass
class Chunk:
    doc_id: str
    text: str
    locator: str | None = None
    metadata: dict = field(default_factory=dict)


class Chunker(Protocol):
    def split(self, doc_id: str, sections: list[RawSection]) -> list[Chunk]: ...


class SimpleChunker:
    """Placeholder default: one chunk per RawSection, merged up to max_chars.

    Chunking strategy (token-aware splitting, overlap, semantic grouping) is a RAG
    deep-dive decision — this exists only so the ingestion pipeline is runnable
    end-to-end until that design lands.
    """

    def __init__(self, max_chars: int = 1000) -> None:
        self._max_chars = max_chars

    def split(self, doc_id: str, sections: list[RawSection]) -> list[Chunk]:
        chunks: list[Chunk] = []
        buffer: list[str] = []
        buffer_len = 0
        first_locator: str | None = None

        def flush() -> None:
            nonlocal buffer, buffer_len, first_locator
            if buffer:
                chunks.append(
                    Chunk(doc_id=doc_id, text="\n".join(buffer), locator=first_locator)
                )
            buffer = []
            buffer_len = 0
            first_locator = None

        for section in sections:
            if first_locator is None:
                first_locator = section.locator
            if buffer_len + len(section.text) > self._max_chars and buffer:
                flush()
                first_locator = section.locator
            buffer.append(section.text)
            buffer_len += len(section.text)

        flush()
        return chunks
