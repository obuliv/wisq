from collections.abc import Iterable

import docx
from sqlalchemy import select

from app.db.models import Document, DocumentRelationship
from app.db.session import async_session_factory
from app.ingestion.chunking import SectionAwareChunker
from app.ingestion.metadata import CompositeMetadataExtractor
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.relationships import SectionAnnotationExtractor
from app.llm.interfaces import Message
from app.rag.fakes import FakeEmbedder, InMemoryVectorStore

PRECEDENCE_TEXT = (
    "This regional handbook is read together with the global Acme Employee "
    "Handbook. Where a conflict arises specifically with respect to PAID TIME "
    "OFF (PTO), the LOCAL PTO POLICY set out in this APAC Benefits Handbook "
    "TAKES PRECEDENCE over any conflicting PTO provision in the global handbook "
    "for employees covered by this document."
)

PRECEDENCE_JSON = """
{
  "relationships": [
    {
      "target_doc_ref": "global Acme Employee Handbook",
      "relation_type": "precedence",
      "topic": "PTO",
      "precedence": "source_over_target",
      "source_text": "TAKES PRECEDENCE over any conflicting PTO provision"
    }
  ],
  "geographic_scope": null
}
"""

GEO_JSON = (
    '{"relationships": [], "geographic_scope": '
    '{"included": ["Singapore", "Malaysia"], "excluded": []}}'
)


class ScriptedLLMClient:
    """Returns a canned response keyed by a substring of the prompted content --
    lets one test exercise both the relationship and geo-scope extraction paths
    without needing a real LLM."""

    def __init__(self, responses: dict[str, str]) -> None:
        self._responses = responses

    def generate(self, messages: list[Message], stream: bool = True) -> Iterable[str]:
        content = messages[-1].content
        for needle, response in self._responses.items():
            if needle in content:
                yield response
                return
        yield "{}"


def _make_apac_docx(path) -> None:
    document = docx.Document()
    document.add_paragraph("APAC Benefits Handbook", style="Title")
    document.add_paragraph("Conflicts and Precedence", style="Heading 1")
    document.add_paragraph(PRECEDENCE_TEXT)
    document.add_paragraph("Eligibility", style="Heading 2")
    document.add_paragraph("This section applies only to employees in Singapore and Malaysia.")
    document.save(str(path))


async def test_pipeline_applies_relationship_and_geo_override_end_to_end(tmp_path):
    docx_path = tmp_path / "APAC Benefits Handbook - version 2.docx"
    _make_apac_docx(docx_path)

    llm = ScriptedLLMClient({"TAKES PRECEDENCE": PRECEDENCE_JSON, "Singapore and Malaysia": GEO_JSON})
    vector_store = InMemoryVectorStore()
    pipeline = IngestionPipeline(
        chunker=SectionAwareChunker(),
        embedder=FakeEmbedder(),
        vector_store=vector_store,
        metadata_extractor=CompositeMetadataExtractor(llm_client=llm),
        relationship_extractor=SectionAnnotationExtractor(llm_client=llm),
    )

    async with async_session_factory() as db:
        document = Document(
            filename=docx_path.name,
            content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            storage_path=str(docx_path),
        )
        db.add(document)
        await db.commit()
        await db.refresh(document)
        doc_id = document.id

        await pipeline.run(db, doc_id, docx_path)

        await db.refresh(document)
        assert document.status == "ready"
        assert document.title == "APAC Benefits Handbook"  # from filename
        assert document.version == "2"  # from filename
        assert document.is_latest is True

        result = await db.execute(
            select(DocumentRelationship).where(DocumentRelationship.source_doc_id == doc_id)
        )
        relationship = result.scalar_one()
        assert relationship.relation_type == "precedence"
        assert relationship.topic == "PTO"

    # Pull everything back out of the vector store to check per-chunk metadata --
    # the Eligibility section should carry the narrower geo override, not null.
    query_vector = FakeEmbedder().embed(["query"])[0]
    all_chunks = {
        tuple(scored.chunk.metadata["heading_path"]): scored.chunk
        for scored in vector_store.search(query_vector, top_k=10)
    }

    eligibility_chunk = all_chunks[("Conflicts and Precedence", "Eligibility")]
    assert eligibility_chunk.metadata["applicable_regions"] == {
        "included": ["Singapore", "Malaysia"],
        "excluded": [],
    }
    assert eligibility_chunk.metadata["is_latest"] is True

    precedence_chunk = all_chunks[("Conflicts and Precedence",)]
    assert precedence_chunk.metadata["applicable_regions"] is None
