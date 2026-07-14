from pathlib import Path

from unstructured.partition.docx import partition_docx

from app.ingestion.loaders.base import Element


class DocxLoader:
    """Extracts structural elements (titles, sections, paragraphs, tables) from a
    .docx file via `unstructured`, and builds a heading-path breadcrumb for each
    element by walking a depth-keyed stack of preceding Title-category elements.
    A PdfLoader can reuse this exact heading-stack logic against partition_pdf's
    output, since unstructured normalizes both formats to the same element shape.
    """

    def load(self, file_path: Path) -> list[Element]:
        raw_elements = partition_docx(filename=str(file_path))

        elements: list[Element] = []
        stack: list[tuple[int, str]] = []  # (depth, heading_text)

        for i, el in enumerate(raw_elements):
            text = str(el).strip()
            if not text:
                continue

            category = el.category
            depth = getattr(el.metadata, "category_depth", None) or 0
            page_number = getattr(el.metadata, "page_number", None)
            locator = f"page {page_number}, element {i}" if page_number is not None else f"element {i}"

            if category == "Title":
                while stack and stack[-1][0] >= depth:
                    stack.pop()
                stack.append((depth, text))

            elements.append(
                Element(
                    text=text,
                    category=category,
                    heading_path=[heading for _, heading in stack],
                    locator=locator,
                )
            )

        return elements
