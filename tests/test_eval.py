"""Tests for the RAG evaluation endpoint."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.models.schemas import Source


def test_eval_returns_scores(client):
    """Eval endpoint should return answer, sources, and scores."""
    mock_sources = [Source(filename="doc1.pdf", page=1)]

    # Embedding mock returns different vectors to produce a non-trivial cosine similarity
    embed_call_count = [0]

    def fake_encode(text, convert_to_tensor=False):
        embed_call_count[0] += 1
        # Return slightly different embeddings for different calls
        if embed_call_count[0] % 2 == 1:
            return [1.0, 0.0, 0.0] + [0.0] * 381
        else:
            return [0.9, 0.1, 0.0] + [0.0] * 381

    client.app.state.embedding_model.encode.side_effect = fake_encode

    with patch("app.routers.eval.search_and_rerank") as mock_search, \
         patch("app.routers.eval.get_llm_service") as mock_factory:
        mock_search.return_value = (
            ["The answer is in this document."],
            mock_sources,
            0.85,
        )
        mock_service = AsyncMock()
        mock_service.generate = AsyncMock(return_value="Here is the evaluated answer.")
        mock_factory.return_value = mock_service

        response = client.post(
            "/api/v1/rag/eval",
            json={
                "question": "How do I make a call?",
                "expected_answer": "Press the call button.",
                "server": "ollama",
                "model": "llama3",
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert "scores" in data
    scores = data["scores"]
    assert "answer_similarity" in scores
    assert "source_coverage" in scores
    assert "latency_ms" in scores
    # Similarity must be a float in [0, 1]
    assert 0.0 <= scores["answer_similarity"] <= 1.0
    assert 0.0 <= scores["source_coverage"] <= 1.0
    assert scores["latency_ms"] >= 0


def test_eval_no_results(client):
    """Eval endpoint should handle empty ES results gracefully."""
    def fake_encode(text, convert_to_tensor=False):
        return [1.0] * 3 + [0.0] * 381

    client.app.state.embedding_model.encode.side_effect = fake_encode

    with patch("app.routers.eval.search_and_rerank") as mock_search, \
         patch("app.routers.eval.get_llm_service") as mock_factory:
        mock_search.return_value = ([], [], None)
        mock_service = AsyncMock()
        mock_service.generate = AsyncMock(return_value="No context answer.")
        mock_factory.return_value = mock_service

        response = client.post(
            "/api/v1/rag/eval",
            json={
                "question": "Unknown question",
                "expected_answer": "Expected text here",
                "server": "ollama",
                "model": "llama3",
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["sources"] == []
    assert data["scores"]["source_coverage"] == 0.0


def test_eval_invalid_server(client):
    """Eval endpoint must return 422 for invalid server."""
    response = client.post(
        "/api/v1/rag/eval",
        json={
            "question": "Test",
            "expected_answer": "Expected",
            "server": "invalid_server",
            "model": "some-model",
        },
    )
    assert response.status_code == 422


def test_eval_perfect_match_similarity(client):
    """When expected_answer == generated answer, answer_similarity must be ~1.0."""
    identical_text = "The sky is blue."

    def fake_encode_identical(text, convert_to_tensor=False):
        # Return same vector regardless of input — perfect cosine similarity
        return [1.0, 0.5, 0.25] + [0.0] * 381

    client.app.state.embedding_model.encode.side_effect = fake_encode_identical

    with patch("app.routers.eval.search_and_rerank") as mock_search, \
         patch("app.routers.eval.get_llm_service") as mock_factory:
        mock_search.return_value = (
            ["The sky is blue."],
            [Source(filename="sky.pdf", page=1)],
            0.95,
        )
        mock_service = AsyncMock()
        # LLM returns the same text as expected
        mock_service.generate = AsyncMock(return_value=identical_text)
        mock_factory.return_value = mock_service

        response = client.post(
            "/api/v1/rag/eval",
            json={
                "question": "What color is the sky?",
                "expected_answer": identical_text,
                "server": "ollama",
                "model": "llama3",
            },
        )

    assert response.status_code == 200
    scores = response.json()["scores"]
    # Identical embeddings → cosine similarity == 1.0
    assert scores["answer_similarity"] == pytest.approx(1.0, abs=1e-4)


def test_eval_zero_source_coverage(client):
    """Source coverage must be 0.0 when source texts share no keywords with expected_answer."""
    def fake_encode(text, convert_to_tensor=False):
        return [0.5, 0.5] + [0.0] * 382

    client.app.state.embedding_model.encode.side_effect = fake_encode

    with patch("app.routers.eval.search_and_rerank") as mock_search, \
         patch("app.routers.eval.get_llm_service") as mock_factory:
        # Source text contains completely unrelated content
        mock_search.return_value = (
            ["xylophone zephyr quasar"],
            [Source(filename="unrelated.txt", page=1)],
            0.5,
        )
        mock_service = AsyncMock()
        mock_service.generate = AsyncMock(return_value="Some generated answer.")
        mock_factory.return_value = mock_service

        response = client.post(
            "/api/v1/rag/eval",
            json={
                "question": "Tell me about biology.",
                # Keywords: "photosynthesis", "chlorophyll" — none appear in source
                "expected_answer": "Photosynthesis uses chlorophyll.",
                "server": "ollama",
                "model": "llama3",
            },
        )

    assert response.status_code == 200
    scores = response.json()["scores"]
    assert scores["source_coverage"] == 0.0
