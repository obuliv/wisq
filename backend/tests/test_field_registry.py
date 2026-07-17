from dataclasses import replace
from datetime import date

from app.chat import formatting
from app.chat.formatting import format_document_scope
from app.db.models import Document
from app.ingestion import field_registry
from app.ingestion.field_registry import (
    PLAIN_FIELDS,
    PlainField,
    plain_field_chunk_metadata,
    render_scope,
    scope_chunk_fields,
)
from app.ingestion.metadata import GeographicScope


def _make_document(**overrides) -> Document:
    document = Document(
        filename="handbook.docx",
        content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        storage_path="/tmp/handbook.docx",
    )
    for key, value in overrides.items():
        setattr(document, key, value)
    return document


def test_plain_field_chunk_metadata_applies_transforms():
    document = _make_document(
        doc_type="policy",
        effective_date=date(2026, 1, 1),
        is_latest=True,
        default_precedence_rule="the more generous benefit applies",
    )
    metadata = plain_field_chunk_metadata(document)
    assert metadata == {
        "doc_type": "policy",
        "effective_date": "2026-01-01",
        "is_latest": True,
        "default_precedence_rule": "the more generous benefit applies",
    }


def test_plain_field_chunk_metadata_handles_missing_effective_date():
    document = _make_document(doc_type=None, effective_date=None)
    metadata = plain_field_chunk_metadata(document)
    assert metadata["effective_date"] is None


def test_scope_chunk_fields_present_vs_absent():
    scope = GeographicScope(included=["Taiwan"], excluded=[])
    present = scope_chunk_fields("regions", scope)
    assert present == {
        "applicable_regions": {"included": ["Taiwan"], "excluded": []},
        "regions_included": ["Taiwan"],
        "regions_excluded": [],
    }

    absent = scope_chunk_fields("regions", None)
    assert absent == {
        "applicable_regions": None,
        "regions_included": [],
        "regions_excluded": [],
    }


def test_render_scope_omits_when_both_lists_empty():
    assert render_scope("regions", {"regions_included": [], "regions_excluded": []}) is None
    assert render_scope("regions", {}) is None


def test_render_scope_renders_included_and_excluded():
    metadata = {"regions_included": ["Taiwan"], "regions_excluded": ["China"]}
    assert render_scope("regions", metadata) == "regions(included=['Taiwan'], excluded=['China'])"


def test_new_plain_field_flows_through_enrichment_and_rendering_with_no_other_edits(monkeypatch):
    """Proves the registry mechanism: a brand-new PlainField reaches both
    plain_field_chunk_metadata (pipeline.py's chunk-enrichment fan-out) and the
    real, unmodified format_document_scope (formatting.py's rendering) via one
    registry entry -- no changes needed to either of those two files."""
    dummy_field = PlainField("confidentiality_level")
    extra_fields = (*PLAIN_FIELDS, dummy_field)

    document = _make_document(doc_type="policy")
    document.confidentiality_level = "restricted"  # arbitrary attribute, no DB round-trip

    # plain_field_chunk_metadata looks up PLAIN_FIELDS in field_registry's own
    # module globals, but formatting.py did `from ... import PLAIN_FIELDS`,
    # binding its own name to the original tuple -- both need patching to
    # simulate "a field was added to the registry" end-to-end.
    monkeypatch.setattr(field_registry, "PLAIN_FIELDS", extra_fields)
    monkeypatch.setattr(formatting, "PLAIN_FIELDS", extra_fields)

    metadata = plain_field_chunk_metadata(document)
    assert metadata["confidentiality_level"] == "restricted"

    rendered = format_document_scope("doc-1", metadata)
    assert "confidentiality_level=restricted" in rendered


def test_plain_field_defaults_chunk_key_to_doc_attr():
    field = PlainField("doc_type")
    assert field.chunk_key == "doc_type"

    named = PlainField("doc_type", chunk_key="kind")
    assert named.chunk_key == "kind"


def test_plain_fields_registry_is_unchanged_by_replace():
    # dataclasses.replace sanity check -- confirms PlainField behaves like a
    # normal dataclass instance (used nowhere in prod code, just documents intent).
    field = PLAIN_FIELDS[0]
    copy = replace(field, chunk_key="renamed")
    assert copy.chunk_key == "renamed"
    assert field.chunk_key != "renamed"
