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
