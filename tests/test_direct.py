from unittest.mock import AsyncMock, patch


def test_direct_ask_ollama(client):
    with patch("app.routers.direct.get_llm_service") as mock_factory:
        mock_service = AsyncMock()
        mock_service.generate = AsyncMock(return_value="Test answer")
        mock_factory.return_value = mock_service

        response = client.post(
            "/api/v1/direct/ask",
            json={"question": "What is Python?", "server": "ollama", "model": "llama3"},
        )

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert data["answer"] == "Test answer"


def test_direct_ask_invalid_server(client):
    response = client.post(
        "/api/v1/direct/ask",
        json={"question": "Hello", "server": "invalid", "model": "some-model"},
    )
    assert response.status_code == 422


def test_direct_ask_empty_question(client):
    response = client.post(
        "/api/v1/direct/ask",
        json={"question": "", "server": "ollama", "model": "llama3"},
    )
    assert response.status_code == 422
