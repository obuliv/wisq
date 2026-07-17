"""Declarative registry for document-level metadata fields.

Drives the two genuinely duplicated fan-out sites for a document-level field:
IngestionPipeline._enrich_chunk_metadata (pipeline.py, copies a field onto
every chunk) and format_document_scope (chat/formatting.py, renders it into
the LLM-visible scope line). Both used to hand-list every field independently;
adding a field now means adding one entry here instead of editing both.

How to add a new document-level field
--------------------------------------
Plain field (single value, no section-level override, e.g. doc_type):
  1. db/models.py    -- add a mapped_column to Document.
  2. metadata.py     -- add it to DocumentMetadata (+ mention in
                         _METADATA_SYSTEM_PROMPT if LLM-extractable).
  3. field_registry.py -- add ONE PlainField(...) entry to PLAIN_FIELDS.
                         Drives chunk enrichment (pipeline.py) AND
                         human-readable rendering (formatting.py) -- neither
                         file needs to change.
  4. schemas.py + frontend/src/api/client.ts -- add the field to DocumentOut
                         (language boundary, stays manual).
  (4 edit sites, down from ~11-13 before this registry existed.)

Scope field (included/excluded lists, e.g. applicable_regions):
  1. db/models.py    -- add a JSON mapped_column.
  2. metadata.py     -- add an _IncludedExcludedScope subclass (one line: its
                         sentinel set) and add it to DocumentMetadata.
  3. field_registry.py -- add one SCOPE_PREFIXES entry.
  4. pipeline.py     -- one scope_chunk_fields(...) call in
                         _enrich_chunk_metadata (plus a section-override
                         resolver ONLY if this field needs one, like
                         geography's _resolve_geo_scope -- most won't).
                         formatting.py needs NO change: it already loops
                         SCOPE_PREFIXES generically.
  5. schemas.py + client.ts -- add to DocumentOut (reuse the shared
                         IncludedExcludedScope TS type).

Relationship-derived field (sourced from another table entirely, e.g.
related_documents): stays a one-off, hand-written directly in pipeline.py's
_enrich_chunk_metadata and formatting.py's format_document_scope -- there's
only ever going to be one of these, so it isn't worth generalizing.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.db.models import Document
from app.ingestion.metadata import GeographicScope, PersonnelScope


def _render_bare(key: str, metadata: dict) -> str | None:
    value = metadata.get(key)
    return f"{key}={value}" if value else None


def _render_present(key: str, metadata: dict) -> str | None:
    # Renders even when falsy (e.g. is_latest=False) -- only *absence* of the
    # key is omitted, unlike _render_bare's truthy gate.
    return f"{key}={metadata[key]}" if key in metadata else None


def _render_quoted(key: str, metadata: dict) -> str | None:
    value = metadata.get(key)
    return f'{key}="{value}"' if value else None


@dataclass
class PlainField:
    """A document-level field copied verbatim onto every chunk with no
    section-level override behavior (the majority case). NOT for fields
    needing per-section override resolution (applicable_regions -- see
    IngestionPipeline._resolve_geo_scope), the included/excluded scope shape
    (applicable_regions/applicable_personnel -- see SCOPE_PREFIXES/
    scope_chunk_fields/render_scope below, a structurally different 3-key
    fan-out), or relationship-derived fields (related_documents -- sourced
    from DocumentRelationship, not a Document column).
    """

    doc_attr: str
    chunk_key: str | None = None
    to_chunk: Callable[[Any], Any] = lambda v: v
    render: Callable[[str, dict], str | None] = _render_bare

    def __post_init__(self) -> None:
        self.chunk_key = self.chunk_key or self.doc_attr


# version/title are also plain Document columns but are deliberately excluded
# here: they're enriched onto chunks (pipeline.py) but were never rendered by
# format_document_scope, so there's no second call site to unify -- forcing
# them in would need a "don't render" flag for zero real duplication payoff.
PLAIN_FIELDS: tuple[PlainField, ...] = (
    PlainField("doc_type"),
    PlainField("effective_date", to_chunk=lambda v: v.isoformat() if v else None),
    PlainField("is_latest", render=_render_present),
    PlainField("default_precedence_rule", render=_render_quoted),
)


def plain_field_chunk_metadata(document: Document) -> dict[str, Any]:
    return {field.chunk_key: field.to_chunk(getattr(document, field.doc_attr)) for field in PLAIN_FIELDS}


# Scope fields (regions/personnel): identical 3-key fan-out shape (a nested
# dict for exact-match lookups, plus flat included/excluded lists for the
# any/not_any/any_or_empty Filters predicates in rag/fakes.py). Only geography
# has section-level override resolution -- that asymmetry stays bespoke in
# pipeline.py (_resolve_geo_scope); only the uniform SHAPE is shared here.
SCOPE_PREFIXES: dict[str, str] = {
    "regions": "applicable_regions",
    "personnel": "applicable_personnel",
}


def scope_chunk_fields(prefix: str, scope: GeographicScope | PersonnelScope | None) -> dict[str, Any]:
    field_name = SCOPE_PREFIXES[prefix]
    return {
        field_name: scope.model_dump() if scope else None,
        f"{prefix}_included": scope.included if scope else [],
        f"{prefix}_excluded": scope.excluded if scope else [],
    }


def render_scope(prefix: str, metadata: dict) -> str | None:
    included = metadata.get(f"{prefix}_included") or []
    excluded = metadata.get(f"{prefix}_excluded") or []
    if not included and not excluded:
        return None
    return f"{prefix}(included={included or 'any'}, excluded={excluded or 'none'})"
