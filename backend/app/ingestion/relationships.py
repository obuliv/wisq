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
    default_precedence_rule: str | None = None


_ANNOTATION_SYSTEM_PROMPT = (
    "You analyze one section of a company policy/handbook document. Identify:\n"
    "1. Any statement that this document relates to another named document (e.g. "
    "references it, is superseded by it, or takes precedence over it for a "
    "specific topic).\n"
    "2. Any statement narrowing which regions/countries THIS SECTION (not "
    "necessarily the whole document) applies to.\n"
    "3. Any GENERAL rule stated for resolving conflicts with other Acme "
    "documents/policies, but ONLY if NO specific document is named (e.g. \"the "
    "more generous benefit applies\"). The test is mechanical: if the sentence "
    "names ANY specific other document -- even when phrased as an instruction, "
    "e.g. \"for all other benefits, refer to the global Acme Employee "
    "Handbook\" -- that is a relationships entry (relation_type=\"reference\" "
    "or \"precedence\", target_doc_ref=the named document), NOT "
    "default_precedence_rule. default_precedence_rule is ONLY for a rule with "
    "no document named anywhere in the sentence. A sentence that merely POINTS "
    "to another section of THIS SAME document (e.g. \"see Section 8 for "
    "details\") is NOT itself a rule either -- leave default_precedence_rule "
    "null for that too.\n"
    "\n"
    "Example -- the text \"For all other benefits, refer to the precedence "
    "rules in the global Acme Employee Handbook. Where a conflict arises with "
    "respect to PTO, the local policy takes precedence.\" must produce:\n"
    '{"relationships": [{"target_doc_ref": "global Acme Employee Handbook", '
    '"relation_type": "reference", "topic": "other benefits", "precedence": '
    'null, "source_text": "For all other benefits, refer to the precedence '
    'rules in the global Acme Employee Handbook."}, {"target_doc_ref": "global '
    'Acme Employee Handbook", "relation_type": "precedence", "topic": "PTO", '
    '"precedence": "source_over_target", "source_text": "Where a conflict '
    'arises with respect to PTO, the local policy takes precedence."}], '
    '"geographic_scope": null, "default_precedence_rule": null}\n'
    "default_precedence_rule stays null here because BOTH statements name a "
    "specific document -- neither is a generic, no-document-named rule.\n"
    "\n"
    'Return ONLY JSON: {"relationships": [{"target_doc_ref": string, '
    '"relation_type": "precedence"|"reference"|"supplement", "topic": string|null, '
    '"precedence": "source_over_target"|"target_over_source"|null, '
    '"source_text": string}], "geographic_scope": {"included": [string], '
    '"excluded": [string]}|null, "default_precedence_rule": string|null}. Use an '
    "empty list / null when nothing applies."
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

        if annotations.default_precedence_rule and not document.default_precedence_rule:
            # First non-null extraction wins, not last -- a later candidate
            # section's (possibly weaker/mistaken) extraction shouldn't clobber
            # an already-correct one from an earlier section.
            document.default_precedence_rule = annotations.default_precedence_rule

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
    is uploaded.

    Multiple versions of the same document family (e.g. "Acme Employee
    Handbook" 2025 and 2026) share the same title, so they score identically
    against a reference like "global Acme Employee Handbook" -- comparing by
    score alone left the tie-break to whichever row the DB query happened to
    return first (arbitrary, and in practice often the superseded version).
    Compare by (score, is_latest) instead: score still strictly dominates (a
    genuinely better title match always wins), but a tie now resolves to the
    current version.
    """
    normalized_ref = normalize_title(target_doc_ref)
    result = await db.execute(select(Document).where(Document.title.is_not(None)))

    best_match: Document | None = None
    best_key: tuple[float, bool] = (0.0, False)
    for candidate in result.scalars().all():
        score = fuzz.token_set_ratio(normalized_ref, normalize_title(candidate.title))
        key = (score, candidate.is_latest)
        if key > best_key:
            best_key = key
            best_match = candidate

    return best_match.id if best_match is not None and best_key[0] >= _MATCH_THRESHOLD else None


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
