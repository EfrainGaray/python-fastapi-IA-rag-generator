"""Tests for observability middleware — X-Request-ID and X-Latency-Ms headers."""

from unittest.mock import AsyncMock, patch

from app.models.schemas import Source


def test_health_has_observability_headers(client):
    """Health endpoint must include X-Request-ID and X-Latency-Ms headers."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "x-request-id" in response.headers
    assert "x-latency-ms" in response.headers


def test_custom_request_id_is_echoed(client):
    """If X-Request-ID is provided in the request, it must be echoed in the response."""
    custom_id = "test-request-id-12345"
    response = client.get("/health", headers={"X-Request-ID": custom_id})
    assert response.status_code == 200
    assert response.headers.get("x-request-id") == custom_id


def test_rag_ask_has_observability_headers(client):
    """RAG ask endpoint must include observability headers."""
    mock_sources = [Source(filename="doc1.pdf", page=1)]

    with patch("app.routers.rag.search_and_rerank") as mock_search, \
         patch("app.routers.rag.get_llm_service") as mock_factory:
        mock_search.return_value = (["Context text."], mock_sources, 0.9)
        mock_service = AsyncMock()
        mock_service.generate = AsyncMock(return_value="The answer.")
        mock_factory.return_value = mock_service

        response = client.post(
            "/api/v1/rag/ask",
            json={"question": "Test?", "server": "ollama", "model": "llama3"},
        )

    assert response.status_code == 200
    assert "x-request-id" in response.headers
    assert "x-latency-ms" in response.headers
    # Latency must be a parseable float
    latency = float(response.headers["x-latency-ms"])
    assert latency >= 0


def test_direct_ask_has_observability_headers(client):
    """Direct ask endpoint must include observability headers."""
    with patch("app.routers.direct.get_llm_service") as mock_factory:
        mock_service = AsyncMock()
        mock_service.generate = AsyncMock(return_value="Direct answer.")
        mock_factory.return_value = mock_service

        response = client.post(
            "/api/v1/direct/ask",
            json={"question": "Test?", "server": "ollama", "model": "llama3"},
        )

    assert response.status_code == 200
    assert "x-request-id" in response.headers
    assert "x-latency-ms" in response.headers


def test_latency_ms_is_numeric(client):
    """X-Latency-Ms header must be a valid numeric string."""
    response = client.get("/health")
    latency_str = response.headers.get("x-latency-ms", "")
    try:
        val = float(latency_str)
        assert val >= 0
    except ValueError:
        raise AssertionError(f"X-Latency-Ms is not numeric: {latency_str!r}")


def test_rag_response_includes_latency_ms_in_body(client):
    """RAGResponse body must include latency_ms field."""
    mock_sources = [Source(filename="doc1.pdf", page=1)]

    with patch("app.routers.rag.search_and_rerank") as mock_search, \
         patch("app.routers.rag.get_llm_service") as mock_factory:
        mock_search.return_value = (["Context."], mock_sources, 0.8)
        mock_service = AsyncMock()
        mock_service.generate = AsyncMock(return_value="Answer text.")
        mock_factory.return_value = mock_service

        response = client.post(
            "/api/v1/rag/ask",
            json={"question": "Test?", "server": "ollama", "model": "llama3"},
        )

    assert response.status_code == 200
    data = response.json()
    assert "latency_ms" in data
    assert data["latency_ms"] >= 0
    assert "chunks_retrieved" in data
    assert data["chunks_retrieved"] >= 0


def test_direct_response_includes_latency_ms_in_body(client):
    """DirectResponse body must include latency_ms field."""
    with patch("app.routers.direct.get_llm_service") as mock_factory:
        mock_service = AsyncMock()
        mock_service.generate = AsyncMock(return_value="Answer.")
        mock_factory.return_value = mock_service

        response = client.post(
            "/api/v1/direct/ask",
            json={"question": "Test?", "server": "ollama", "model": "llama3"},
        )

    assert response.status_code == 200
    data = response.json()
    assert "latency_ms" in data
    assert data["latency_ms"] >= 0
