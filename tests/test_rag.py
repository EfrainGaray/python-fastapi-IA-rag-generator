from unittest.mock import AsyncMock, MagicMock, patch

from app.exceptions import LLMProviderError
from app.models.schemas import Source


def test_rag_ask_returns_answer(client):
    mock_sources = [Source(filename="doc1.pdf", page=1)]
    with patch("app.routers.rag.search_and_rerank") as mock_search, \
         patch("app.routers.rag.get_llm_service") as mock_factory:
        mock_search.return_value = (["Relevant context text."], mock_sources, 0.9)
        mock_service = AsyncMock()
        mock_service.generate = AsyncMock(return_value="Here is the answer.")
        mock_factory.return_value = mock_service

        response = client.post(
            "/api/v1/rag/ask",
            json={
                "question": "How do I make a call?",
                "server": "ollama",
                "model": "llama3",
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert data["answer"] == "Here is the answer."
    assert isinstance(data["sources"], list)
    assert len(data["sources"]) > 0


def test_rag_ask_no_results(client):
    with patch("app.routers.rag.search_and_rerank") as mock_search:
        mock_search.return_value = ([], [], None)

        response = client.post(
            "/api/v1/rag/ask",
            json={
                "question": "Completely unknown topic",
                "server": "ollama",
                "model": "llama3",
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert "Lo siento" in data["answer"]
    assert data["sources"] == []


def test_rag_ask_llm_error(client):
    mock_sources = [Source(filename="doc1.pdf", page=1)]
    with patch("app.routers.rag.search_and_rerank") as mock_search, \
         patch("app.routers.rag.get_llm_service") as mock_factory:
        mock_search.return_value = (["Relevant context text."], mock_sources, 0.9)
        mock_service = AsyncMock()
        mock_service.generate = AsyncMock(
            side_effect=LLMProviderError("Service unavailable")
        )
        mock_factory.return_value = mock_service

        response = client.post(
            "/api/v1/rag/ask",
            json={
                "question": "How do I send a message?",
                "server": "ollama",
                "model": "llama3",
            },
        )

    assert response.status_code == 502


def test_rag_ask_question_at_max_length(client):
    """A question exactly at the 2000-character max_length must be accepted."""
    max_question = "A" * 2000
    mock_sources = [Source(filename="doc1.pdf", page=1)]

    with patch("app.routers.rag.search_and_rerank") as mock_search, \
         patch("app.routers.rag.get_llm_service") as mock_factory:
        mock_search.return_value = (["Context."], mock_sources, 0.8)
        mock_service = AsyncMock()
        mock_service.generate = AsyncMock(return_value="Answer to long question.")
        mock_factory.return_value = mock_service

        response = client.post(
            "/api/v1/rag/ask",
            json={"question": max_question, "server": "ollama", "model": "llama3"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Answer to long question."


def test_rag_ask_question_exceeds_max_length(client):
    """A question exceeding 2000 characters must be rejected with 422."""
    too_long = "B" * 2001
    response = client.post(
        "/api/v1/rag/ask",
        json={"question": too_long, "server": "ollama", "model": "llama3"},
    )
    assert response.status_code == 422


def test_rag_ask_server_openai(client):
    """RAG ask must work with server=openai."""
    mock_sources = [Source(filename="doc1.pdf", page=1)]
    with patch("app.routers.rag.search_and_rerank") as mock_search, \
         patch("app.routers.rag.get_llm_service") as mock_factory:
        mock_search.return_value = (["Context."], mock_sources, 0.7)
        mock_service = AsyncMock()
        mock_service.generate = AsyncMock(return_value="OpenAI answer.")
        mock_factory.return_value = mock_service

        response = client.post(
            "/api/v1/rag/ask",
            json={"question": "What is AI?", "server": "openai", "model": "gpt-4o-mini"},
        )

    assert response.status_code == 200
    assert response.json()["answer"] == "OpenAI answer."


def test_rag_ask_server_huggingface(client):
    """RAG ask must work with server=huggingface."""
    mock_sources = [Source(filename="doc1.pdf", page=1)]
    with patch("app.routers.rag.search_and_rerank") as mock_search, \
         patch("app.routers.rag.get_llm_service") as mock_factory:
        mock_search.return_value = (["Context."], mock_sources, 0.6)
        mock_service = AsyncMock()
        mock_service.generate = AsyncMock(return_value="HuggingFace answer.")
        mock_factory.return_value = mock_service

        response = client.post(
            "/api/v1/rag/ask",
            json={
                "question": "Explain neural networks.",
                "server": "huggingface",
                "model": "mistralai/Mistral-7B-v0.1",
            },
        )

    assert response.status_code == 200
    assert response.json()["answer"] == "HuggingFace answer."


def test_rag_ask_invalid_server(client):
    """RAG ask must reject an unknown server value with 422."""
    response = client.post(
        "/api/v1/rag/ask",
        json={"question": "Hello?", "server": "unknown_provider", "model": "x"},
    )
    assert response.status_code == 422
