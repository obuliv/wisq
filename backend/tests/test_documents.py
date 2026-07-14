import io

import docx


def _make_docx_bytes(text: str) -> bytes:
    document = docx.Document()
    document.add_paragraph(text)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def test_upload_docx_gets_indexed(client):
    content = _make_docx_bytes("Hello from a test document.")
    response = client.post(
        "/api/documents",
        files={
            "file": (
                "sample.docx",
                content,
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
    )
    assert response.status_code == 201
    doc_id = response.json()["id"]
    assert response.json()["status"] == "queued"

    detail = client.get(f"/api/documents/{doc_id}")
    assert detail.status_code == 200
    assert detail.json()["status"] == "ready"


def test_list_documents(client):
    response = client.get("/api/documents")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_unsupported_extension_rejected(client):
    response = client.post(
        "/api/documents",
        files={"file": ("sample.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400


def test_missing_document_404s(client):
    response = client.get("/api/documents/does-not-exist")
    assert response.status_code == 404
