import pytest

from app.ingestion.metadata import FilenameMetadataExtractor, GeographicScope, normalize_title


@pytest.mark.parametrize(
    ("filename", "expected_title", "expected_version"),
    [
        ("Employee Handbook - version 2026.docx", "Employee Handbook", "2026"),
        ("APAC_Benefits_Handbook_v2.docx", "APAC Benefits Handbook", "2"),
        ("Acme Global Employee Handbook.docx", "Acme Global Employee Handbook", None),
        ("Regional-PTO-Policy-v2.1.docx", "Regional PTO Policy", "2.1"),
    ],
)
def test_filename_metadata_extraction(filename, expected_title, expected_version):
    metadata = FilenameMetadataExtractor().extract(filename, elements=[])
    assert metadata.title == expected_title
    assert metadata.version == expected_version


def test_normalize_title_ignores_case_and_punctuation():
    assert normalize_title("  Acme Global Employee Handbook!! ") == "acme global employee handbook"
    assert normalize_title("Acme, Global Employee Handbook") == normalize_title(
        "acme global employee handbook"
    )


@pytest.mark.parametrize("sentinel", ["Worldwide", "  Global  ", "GLOBALLY", "all locations"])
def test_geographic_scope_normalizes_unrestricted_sentinels_to_empty(sentinel):
    # Regression test: an LLM naturally writes a sentinel like "worldwide" to
    # mean "no restriction," but the any_or_empty filter predicate requires an
    # empty list for that meaning -- a literal sentinel string silently fails to
    # match any specific geography query instead.
    scope = GeographicScope(included=[sentinel], excluded=[])
    assert scope.included == []


def test_geographic_scope_leaves_real_region_names_unchanged():
    scope = GeographicScope(included=["Singapore", "Japan"], excluded=["Malaysia"])
    assert scope.included == ["Singapore", "Japan"]
    assert scope.excluded == ["Malaysia"]
