"""Tests for document management API endpoints."""

import io
from unittest.mock import AsyncMock, MagicMock, patch


def test_upload_txt_document(client):
    """Upload a TXT file — should index chunks and return upload response."""
    file_content = b"Hello world this is a test document with some content."
    with patch("app.routers.documents.ingest_file", new_callable=AsyncMock) as mock_ingest:
        mock_ingest.return_value = 3
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("test.txt", io.BytesIO(file_content), "text/plain")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "test.txt"
    assert data["chunks_indexed"] == 3
    assert "message" in data


def test_upload_pdf_document(client):
    """Upload a PDF file — should call ingest and return chunks_indexed."""
    file_content = b"%PDF-1.4 fake pdf content"
    with patch("app.routers.documents.ingest_file", new_callable=AsyncMock) as mock_ingest:
        mock_ingest.return_value = 5
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("report.pdf", io.BytesIO(file_content), "application/pdf")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "report.pdf"
    assert data["chunks_indexed"] == 5


def test_upload_unsupported_format(client):
    """Uploading an unsupported format (.mp3) must return 422."""
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("audio.mp3", io.BytesIO(b"fake"), "audio/mpeg")},
    )
    assert response.status_code == 422


def test_list_documents(client):
    """List endpoint should return documents aggregated from ES."""
    mock_agg_response = {
        "aggregations": {
            "files": {
                "buckets": [
                    {"key": "doc1.pdf", "doc_count": 10},
                    {"key": "doc2.txt", "doc_count": 4},
                ]
            }
        }
    }
    mock_es = MagicMock()
    mock_es.search.return_value = mock_agg_response

    with patch("app.routers.documents.Request") as _:
        # Patch at the router level — override app.state.es directly
        import app.main as main_module
        original_es = main_module.app.state.es
        main_module.app.state.es = mock_es
        try:
            response = client.get("/api/v1/documents")
        finally:
            main_module.app.state.es = original_es

    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    assert data["total"] == 2
    filenames = [d["filename"] for d in data["documents"]]
    assert "doc1.pdf" in filenames
    assert "doc2.txt" in filenames


def test_delete_document(client):
    """Delete endpoint should call delete_by_query and return deleted count."""
    mock_es = MagicMock()
    mock_es.delete_by_query.return_value = {"deleted": 7}

    import app.main as main_module
    original_es = main_module.app.state.es
    main_module.app.state.es = mock_es
    try:
        response = client.delete("/api/v1/documents/doc1.pdf")
    finally:
        main_module.app.state.es = original_es

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "doc1.pdf"
    assert data["deleted"] == 7
    mock_es.delete_by_query.assert_called_once()
