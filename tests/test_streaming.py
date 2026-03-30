"""Tests for SSE streaming endpoints."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

from app.models.schemas import Source


async def _fake_stream(*args, **kwargs):
    """Async generator that yields two tokens."""
    yield "Hello"
    yield " world"


def test_rag_ask_stream_returns_event_stream(client):
    """When stream=True, rag/ask must return text/event-stream content-type."""
    mock_sources = [Source(filename="doc1.pdf", page=1)]

    mock_service = MagicMock()
    mock_service.stream_generate = _fake_stream

    with patch("app.routers.rag.search_and_rerank") as mock_search, \
         patch("app.routers.rag.get_llm_service") as mock_factory:
        mock_search.return_value = (["Relevant context."], mock_sources, 0.9)
        mock_factory.return_value = mock_service

        response = client.post(
            "/api/v1/rag/ask",
            json={
                "question": "How do I make a call?",
                "server": "ollama",
                "model": "llama3",
                "stream": True,
            },
        )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]


def test_rag_ask_stream_false_returns_json(client):
    """When stream=False (default), rag/ask must return normal JSON."""
    mock_sources = [Source(filename="doc1.pdf", page=1)]

    with patch("app.routers.rag.search_and_rerank") as mock_search, \
         patch("app.routers.rag.get_llm_service") as mock_factory:
        mock_search.return_value = (["Relevant context."], mock_sources, 0.9)
        mock_service = AsyncMock()
        mock_service.generate = AsyncMock(return_value="Normal answer.")
        mock_factory.return_value = mock_service

        response = client.post(
            "/api/v1/rag/ask",
            json={
                "question": "How do I make a call?",
                "server": "ollama",
                "model": "llama3",
                "stream": False,
            },
        )

    assert response.status_code == 200
    assert "application/json" in response.headers["content-type"]
    data = response.json()
    assert "answer" in data
    assert data["answer"] == "Normal answer."


def test_direct_ask_stream_returns_event_stream(client):
    """When stream=True, direct/ask must return text/event-stream content-type."""
    mock_service = MagicMock()
    mock_service.stream_generate = _fake_stream

    with patch("app.routers.direct.get_llm_service") as mock_factory:
        mock_factory.return_value = mock_service

        response = client.post(
            "/api/v1/direct/ask",
            json={
                "question": "What is Python?",
                "server": "ollama",
                "model": "llama3",
                "stream": True,
            },
        )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]


def test_direct_ask_stream_false_returns_json(client):
    """When stream=False (default), direct/ask must return normal JSON."""
    with patch("app.routers.direct.get_llm_service") as mock_factory:
        mock_service = AsyncMock()
        mock_service.generate = AsyncMock(return_value="Direct answer.")
        mock_factory.return_value = mock_service

        response = client.post(
            "/api/v1/direct/ask",
            json={
                "question": "What is Python?",
                "server": "ollama",
                "model": "llama3",
                "stream": False,
            },
        )

    assert response.status_code == 200
    assert "application/json" in response.headers["content-type"]
    data = response.json()
    assert data["answer"] == "Direct answer."


def test_rag_stream_sse_format(client):
    """SSE events must be newline-delimited and parseable JSON with token/done fields."""
    mock_sources = [Source(filename="doc1.pdf", page=1)]

    mock_service = MagicMock()
    mock_service.stream_generate = _fake_stream

    with patch("app.routers.rag.search_and_rerank") as mock_search, \
         patch("app.routers.rag.get_llm_service") as mock_factory:
        mock_search.return_value = (["Relevant context."], mock_sources, 0.9)
        mock_factory.return_value = mock_service

        response = client.post(
            "/api/v1/rag/ask",
            json={
                "question": "How do I make a call?",
                "server": "ollama",
                "model": "llama3",
                "stream": True,
            },
        )

    assert response.status_code == 200
    raw = response.text
    # Each SSE event starts with "data: "
    events = [line for line in raw.splitlines() if line.startswith("data: ")]
    assert len(events) >= 1

    # All events must be valid JSON with token + done fields
    for event in events:
        payload = json.loads(event[len("data: "):])
        assert "token" in payload
        assert "done" in payload

    # Last event must have done=True
    last = json.loads(events[-1][len("data: "):])
    assert last["done"] is True


async def _empty_stream(*args, **kwargs):
    """Async generator that yields nothing (simulates no LLM output)."""
    return
    yield  # make it an async generator


def test_rag_stream_no_es_results(client_empty_es):
    """When ES returns no hits, the RAG ask endpoint returns a no-results JSON response.

    The streaming path is only reached after ES finds documents; if ES is empty the
    router returns early with a non-streaming JSON body.
    """
    with patch("app.routers.rag.search_and_rerank") as mock_search:
        mock_search.return_value = ([], [], None)

        response = client_empty_es.post(
            "/api/v1/rag/ask",
            json={
                "question": "A question with no indexed documents",
                "server": "ollama",
                "model": "llama3",
                "stream": True,
            },
        )

    # Router returns early with JSON (not SSE) when there are no results
    assert response.status_code == 200
    data = response.json()
    assert "Lo siento" in data["answer"]
    assert data["sources"] == []


def test_rag_stream_done_true_in_final_chunk(client):
    """The final SSE chunk must carry done=True regardless of token count."""
    mock_sources = [Source(filename="doc1.pdf", page=1)]
    mock_service = MagicMock()
    mock_service.stream_generate = _fake_stream

    with patch("app.routers.rag.search_and_rerank") as mock_search, \
         patch("app.routers.rag.get_llm_service") as mock_factory:
        mock_search.return_value = (["Context text."], mock_sources, 0.85)
        mock_factory.return_value = mock_service

        response = client.post(
            "/api/v1/rag/ask",
            json={
                "question": "Testing done flag",
                "server": "ollama",
                "model": "llama3",
                "stream": True,
            },
        )

    events = [
        line for line in response.text.splitlines() if line.startswith("data: ")
    ]
    assert len(events) >= 1
    final_payload = json.loads(events[-1][len("data: "):])
    assert final_payload["done"] is True
    # All non-final chunks must have done=False
    for event in events[:-1]:
        payload = json.loads(event[len("data: "):])
        assert payload["done"] is False
