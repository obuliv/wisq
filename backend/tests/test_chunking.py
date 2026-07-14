from app.ingestion.chunking import SectionAwareChunker
from app.ingestion.loaders.base import Element


def test_section_aware_chunker_groups_by_heading_and_drops_bare_titles():
    elements = [
        Element(text="Acme Employee Handbook", category="Title", heading_path=["Acme Employee Handbook"]),
        Element(
            text="Conflicts and Precedence",
            category="Title",
            heading_path=["Conflicts and Precedence"],
        ),
        Element(
            text="This regional handbook is read together with the global handbook.",
            category="NarrativeText",
            heading_path=["Conflicts and Precedence"],
        ),
        Element(
            text="Eligibility",
            category="Title",
            heading_path=["Conflicts and Precedence", "Eligibility"],
        ),
        Element(
            text="This section applies only to employees in Singapore and Malaysia.",
            category="NarrativeText",
            heading_path=["Conflicts and Precedence", "Eligibility"],
        ),
    ]

    chunks = SectionAwareChunker().split("doc-1", elements)

    # The lone "Acme Employee Handbook" title (no content under it) is dropped.
    assert len(chunks) == 2
    assert chunks[0].metadata["heading_path"] == ["Conflicts and Precedence"]
    assert "read together with the global handbook" in chunks[0].text
    assert chunks[1].metadata["heading_path"] == ["Conflicts and Precedence", "Eligibility"]
    assert "Singapore and Malaysia" in chunks[1].text


def test_section_aware_chunker_splits_oversized_sections():
    # Splitting happens at element (paragraph) boundaries, not mid-paragraph --
    # this mirrors a section with many real paragraphs whose combined length
    # exceeds max_chars.
    elements = [Element(text="Big Section", category="Title", heading_path=["Big Section"])]
    elements += [
        Element(text=f"Paragraph {i}: " + "word " * 10, category="NarrativeText", heading_path=["Big Section"])
        for i in range(10)
    ]

    chunks = SectionAwareChunker(max_chars=200).split("doc-1", elements)

    assert len(chunks) > 1
    assert all(c.metadata["heading_path"] == ["Big Section"] for c in chunks)
