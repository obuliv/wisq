import pytest

from app.ingestion.metadata import FilenameMetadataExtractor, normalize_title


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
