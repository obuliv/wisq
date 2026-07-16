import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, field_validator

from app.ingestion.llm_extraction import extract_json
from app.ingestion.loaders.base import Element
from app.llm.interfaces import LLMClient

_YEAR_RE = re.compile(r"\b(20\d{2})\b")
_SEMVER_RE = re.compile(r"\bv?(\d+\.\d+(?:\.\d+)?)\b", re.IGNORECASE)
_V_INT_RE = re.compile(r"\bv(\d+)\b", re.IGNORECASE)
# Explicit "version X" phrasing wins over the patterns above, and must be
# stripped as one unit -- otherwise a bare number like "version 1" (no year/v
# prefix pattern to match) is left dangling in the cleaned title.
_VERSION_PHRASE_RE = re.compile(r"\bversion\s+([\w.]+)", re.IGNORECASE)
_SEPARATORS_RE = re.compile(r"[-_]+")
_TRAILING_PUNCT_RE = re.compile(r"[,\-–—\s]+$")
_EXTRA_SPACE_RE = re.compile(r"\s{2,}")
_NON_WORD_RE = re.compile(r"[^\w\s]")


def normalize_title(title: str) -> str:
    """Canonical key for grouping/matching documents by title: lowercase, strip
    punctuation and collapse whitespace. Shared by version-grouping
    (Document.document_group_key, see versioning.py) and cross-document
    relationship-target resolution (see relationships.py) -- one normalization
    for both problems."""
    cleaned = _NON_WORD_RE.sub(" ", title.lower())
    return _EXTRA_SPACE_RE.sub(" ", cleaned).strip()


_UNRESTRICTED_SENTINELS = {
    "worldwide",
    "global",
    "globally",
    "all",
    "all locations",
    "all countries",
    "all regions",
    "everywhere",
    "any location",
    "any country",
}


class GeographicScope(BaseModel):
    included: list[str] = []
    excluded: list[str] = []

    @field_validator("included")
    @classmethod
    def _normalize_unrestricted(cls, value: list[str]) -> list[str]:
        # An LLM naturally writes a sentinel like "worldwide"/"global" to mean
        # "no geographic restriction" -- but the any_or_empty filter predicate
        # (rag/fakes.py) requires an EMPTY list for that meaning, not a literal
        # string with no overlap against a specific query region (e.g.
        # geography="Singapore" would otherwise wrongly exclude a chunk tagged
        # regions_included=["worldwide"]). Collapse the whole list to [] if any
        # entry matches a recognized sentinel.
        if any(v.strip().lower() in _UNRESTRICTED_SENTINELS for v in value):
            return []
        return value


@dataclass
class DocumentMetadata:
    doc_type: str | None = None
    title: str | None = None
    version: str | None = None
    effective_date: date | None = None
    applicable_regions: GeographicScope | None = None


class DocumentMetadataExtractor(Protocol):
    def extract(self, filename: str, elements: list[Element]) -> DocumentMetadata: ...


class FilenameMetadataExtractor:
    """Deterministic, regex-based extraction from the filename alone -- no LLM
    call needed. High precision when the filename follows a convention like
    "Employee Handbook - version 2026.docx"; leaves fields None otherwise so
    LLMDocumentMetadataExtractor can attempt them from the document body."""

    def extract(self, filename: str, elements: list[Element]) -> DocumentMetadata:
        # Normalize separators first so "_v2" / "-v2" still word-boundary-match
        # like "v2" would (underscores/hyphens are word-adjacent otherwise).
        normalized = _SEPARATORS_RE.sub(" ", Path(filename).stem)
        return DocumentMetadata(
            title=self._extract_title(normalized), version=self._extract_version(normalized)
        )

    def _extract_version(self, normalized: str) -> str | None:
        version_phrase = _VERSION_PHRASE_RE.search(normalized)
        if version_phrase:
            return version_phrase.group(1)
        semver = _SEMVER_RE.search(normalized)
        if semver:
            return semver.group(1)
        year = _YEAR_RE.search(normalized)
        if year:
            return year.group(1)
        v_int = _V_INT_RE.search(normalized)
        return v_int.group(1) if v_int else None

    def _extract_title(self, normalized: str) -> str | None:
        cleaned = _VERSION_PHRASE_RE.sub("", normalized)
        cleaned = _SEMVER_RE.sub("", cleaned)
        cleaned = _YEAR_RE.sub("", cleaned)
        cleaned = _V_INT_RE.sub("", cleaned)
        cleaned = _TRAILING_PUNCT_RE.sub("", cleaned)
        cleaned = _EXTRA_SPACE_RE.sub(" ", cleaned).strip()
        return cleaned or None


_METADATA_SYSTEM_PROMPT = (
    "Extract document metadata from the start of a policy/handbook document. "
    'Return ONLY a JSON object: {"doc_type": string|null, "title": string|null, '
    '"effective_date": "YYYY-MM-DD"|null, '
    '"applicable_regions": {"included": [string], "excluded": [string]}|null}. '
    "Use null for anything not clearly stated in the text. Only set "
    "applicable_regions if the document explicitly states which regions/countries "
    "it applies to. If the document applies with no geographic restriction "
    "(e.g. worldwide, globally, all locations), set \"included\" to an empty "
    "list [] -- do NOT use a word like \"worldwide\" or \"global\" as an entry."
)


class _LLMDocumentMetadata(BaseModel):
    doc_type: str | None = None
    title: str | None = None
    effective_date: date | None = None
    applicable_regions: GeographicScope | None = None


class LLMDocumentMetadataExtractor:
    """Fills in fields the filename can't provide (effective_date,
    applicable_regions) via one LLM call over the document's opening elements.
    Degrades to an empty DocumentMetadata (never raises) if the LLM response
    isn't valid JSON -- e.g. the current FakeLLMClient."""

    def __init__(self, llm_client: LLMClient, max_elements: int = 20) -> None:
        self._llm_client = llm_client
        self._max_elements = max_elements

    def extract(self, filename: str, elements: list[Element]) -> DocumentMetadata:
        intro = "\n".join(el.text for el in elements[: self._max_elements])
        result = extract_json(self._llm_client, _METADATA_SYSTEM_PROMPT, intro, _LLMDocumentMetadata)
        if result is None:
            return DocumentMetadata()
        return DocumentMetadata(
            doc_type=result.doc_type,
            title=result.title,
            effective_date=result.effective_date,
            applicable_regions=result.applicable_regions,
        )


class CompositeMetadataExtractor:
    """Filename extraction first (free, high-precision for title/version), then
    LLM extraction fills the rest -- doc_type, effective_date, and
    applicable_regions realistically only come from the document body."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._filename_extractor = FilenameMetadataExtractor()
        self._llm_extractor = LLMDocumentMetadataExtractor(llm_client)

    def extract(self, filename: str, elements: list[Element]) -> DocumentMetadata:
        from_filename = self._filename_extractor.extract(filename, elements)
        from_llm = self._llm_extractor.extract(filename, elements)
        return DocumentMetadata(
            doc_type=from_filename.doc_type or from_llm.doc_type,
            title=from_filename.title or from_llm.title,
            version=from_filename.version or from_llm.version,
            effective_date=from_filename.effective_date or from_llm.effective_date,
            applicable_regions=from_filename.applicable_regions or from_llm.applicable_regions,
        )
