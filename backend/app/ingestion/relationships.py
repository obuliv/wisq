import logging
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel
from rapidfuzz import fuzz
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Document, DocumentRelationship
from app.ingestion.llm_extraction import extract_json
from app.ingestion.loaders.base import Element
from app.ingestion.metadata import GeographicScope, normalize_title
from app.llm.interfaces import LLMClient

logger = logging.getLogger(__name__)

_MATCH_THRESHOLD = 80.0  # rapidfuzz token_set_ratio, 0-100

_CUE_PHRASES = [
    "takes precedence",
    "take precedence",
    "supersede",
    "read together with",
    "refer to the global",
    "in the event of a conflict",
    "applies to",
    "covered by",
    "does not apply to",
]
_HEADING_SIGNALS = {"conflicts and precedence", "eligibility", "scope", "applicability"}


@dataclass
class CandidateSection:
    heading_path: tuple[str, ...]
    text: str
    locator: str | None


@dataclass
class RelationshipHint:
    """Compact summary of an outgoing DocumentRelationship, baked onto every
    chunk of the source document (see IngestionPipeline._enrich_chunk_metadata)
    so the model sees it on *any* search hit for that document -- not just if
    it happens to retrieve the specific section the relationship was extracted
    from, which vector search can easily miss (e.g. a "gym benefits" query
    won't reliably retrieve a "CONFLICTS AND PRECEDENCE" section)."""

    relation_type: str
    topic: str | None
    target_doc_ref: str


def find_candidate_sections(elements: list[Element]) -> list[CandidateSection]:
    """Groups elements by heading_path and keeps only sections whose heading or
    body text contains a relationship/scope cue -- avoids sending the whole
    document to the LLM for what's usually a couple of flagged sections."""
    sections: dict[tuple[str, ...], list[Element]] = {}
    order: list[tuple[str, ...]] = []
    for element in elements:
        key = tuple(element.heading_path)
        if key not in sections:
            sections[key] = []
            order.append(key)
        sections[key].append(element)

    candidates: list[CandidateSection] = []
    for key in order:
        section_elements = sections[key]
        text = "\n".join(el.text for el in section_elements)
        heading_lower = key[-1].lower() if key else ""
        haystack = f"{text} {' '.join(key)}".lower()
        if heading_lower in _HEADING_SIGNALS or any(phrase in haystack for phrase in _CUE_PHRASES):
            candidates.append(
                CandidateSection(heading_path=key, text=text, locator=section_elements[0].locator)
            )
    return candidates


class _ExtractedRelationship(BaseModel):
    target_doc_ref: str
    relation_type: Literal["precedence", "reference", "supplement"]
    topic: str | None = None
    precedence: Literal["source_over_target", "target_over_source"] | None = None
    source_text: str


class _SectionAnnotations(BaseModel):
    relationships: list[_ExtractedRelationship] = []
    geographic_scope: GeographicScope | None = None


_ANNOTATION_SYSTEM_PROMPT = (
    "You analyze one section of a company policy/handbook document. Identify:\n"
    "1. Any statement that this document relates to another named document (e.g. "
    "references it, is superseded by it, or takes precedence over it for a "
    "specific topic).\n"
    "2. Any statement narrowing which regions/countries THIS SECTION (not "
    "necessarily the whole document) applies to.\n"
    'Return ONLY JSON: {"relationships": [{"target_doc_ref": string, '
    '"relation_type": "precedence"|"reference"|"supplement", "topic": string|null, '
    '"precedence": "source_over_target"|"target_over_source"|null, '
    '"source_text": string}], "geographic_scope": {"included": [string], '
    '"excluded": [string]}|null}. Use an empty list / null when nothing applies.'
)


class SectionAnnotationExtractor:
    """One structured LLM call per candidate section, extracting both
    cross-document relationships and section-scoped geographic overrides in a
    single pass (the same "scope/conditions" sections tend to state both)."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    def extract(self, section: CandidateSection) -> _SectionAnnotations | None:
        return extract_json(
            self._llm_client, _ANNOTATION_SYSTEM_PROMPT, section.text, _SectionAnnotations
        )


async def extract_and_store_relationships(
    db: AsyncSession,
    document: Document,
    elements: list[Element],
    extractor: SectionAnnotationExtractor,
) -> tuple[dict[tuple[str, ...], GeographicScope], list[RelationshipHint]]:
    """Runs section-annotation extraction for `document`, persists resolved
    DocumentRelationship rows, and returns (a) a heading_path -> GeographicScope
    map of section-level geographic overrides for the chunker to apply, and (b)
    a RelationshipHint per outgoing relationship found, for the pipeline to bake
    onto every chunk's metadata (see IngestionPipeline._enrich_chunk_metadata).
    Never raises: a section the LLM fails to parse is simply skipped (see
    extract_json).

    Only covers relationships where `document` is the source -- a document only
    knows about its own outgoing references at its own ingestion time. The
    reverse case (an existing document later becoming the *target* of a new
    document's relationship) isn't hinted retroactively here; it would need the
    same update_metadata staleness-patching mechanism used for `is_latest`.
    """
    geo_overrides: dict[tuple[str, ...], GeographicScope] = {}
    hints: list[RelationshipHint] = []

    for section in find_candidate_sections(elements):
        annotations = extractor.extract(section)
        if annotations is None:
            continue

        if annotations.geographic_scope is not None:
            geo_overrides[section.heading_path] = annotations.geographic_scope

        for rel in annotations.relationships:
            target_id = await _resolve_target(db, rel.target_doc_ref)
            db.add(
                DocumentRelationship(
                    source_doc_id=document.id,
                    target_doc_id=target_id,
                    target_doc_ref=rel.target_doc_ref,
                    relation_type=rel.relation_type,
                    topic=rel.topic,
                    precedence=rel.precedence,
                    source_text=rel.source_text,
                    source_locator=section.locator,
                )
            )
            hints.append(
                RelationshipHint(
                    relation_type=rel.relation_type, topic=rel.topic, target_doc_ref=rel.target_doc_ref
                )
            )

    await reconcile_relationships(db, document)
    return geo_overrides, hints


async def _resolve_target(db: AsyncSession, target_doc_ref: str) -> str | None:
    """Fuzzy-matches a mentioned document name against existing Document titles.
    Returns None (leaving the relationship unresolved) when no confident match
    exists yet -- reconcile_relationships backfills it once/if that document
    is uploaded."""
    normalized_ref = normalize_title(target_doc_ref)
    result = await db.execute(select(Document).where(Document.title.is_not(None)))

    best_match: Document | None = None
    best_score = 0.0
    for candidate in result.scalars().all():
        score = fuzz.token_set_ratio(normalized_ref, normalize_title(candidate.title))
        if score > best_score:
            best_score = score
            best_match = candidate

    return best_match.id if best_match is not None and best_score >= _MATCH_THRESHOLD else None


async def reconcile_relationships(db: AsyncSession, document: Document) -> None:
    """Whenever a document finishes ingestion, re-check other documents'
    unresolved relationship target refs against its new title, backfilling
    target_doc_id where they now match -- handles references made before the
    target document existed."""
    if not document.title:
        return

    normalized_title = normalize_title(document.title)
    result = await db.execute(
        select(DocumentRelationship).where(DocumentRelationship.target_doc_id.is_(None))
    )
    for relationship in result.scalars().all():
        score = fuzz.token_set_ratio(normalize_title(relationship.target_doc_ref), normalized_title)
        if score >= _MATCH_THRESHOLD:
            relationship.target_doc_id = document.id
