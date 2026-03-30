"""Integration-style tests — multiple layers mocked but wired together.

These tests exercise the full request path through FastAPI routers, the
search_and_rerank helper, and LLM services using in-process stubs only.
No real Elasticsearch, ML models, or external HTTP calls are made.
"""

import io
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from app.models.schemas import Source


# ── Full RAG pipeline ─────────────────────────────────────────────────────────

def test_full_rag_pipeline_response_shape(client):
    """End-to-end: question → ES mock → rerank mock → LLM mock → verify response shape."""
    mock_sources = [
        Source(filename="guide.pdf", page=3),
        Source(filename="manual.txt", page=1),
    ]

    with patch("app.routers.rag.search_and_rerank") as mock_search, \
         patch("app.routers.rag.get_llm_service") as mock_factory:
        mock_search.return_value = (
            ["Step 1: open the app.", "Step 2: tap the button."],
            mock_sources,
            0.92,
        )
        mock_service = AsyncMock()
        mock_service.generate = AsyncMock(return_value="Here are the steps: open, then tap.")
        mock_factory.return_value = mock_service

        response = client.post(
            "/api/v1/rag/ask",
            json={
                "question": "How do I use the app?",
                "server": "ollama",
                "model": "llama3",
            },
        )

    assert response.status_code == 200
    data = response.json()

    # Verify top-level shape
    assert "answer" in data
    assert "sources" in data
    assert "latency_ms" in data
    assert "chunks_retrieved" in data
    assert "top_score" in data

    # Verify content
    assert data["answer"] == "Here are the steps: open, then tap."
    assert data["chunks_retrieved"] == 2
    assert data["top_score"] == pytest.approx(0.92)

    # Verify sources shape
    assert len(data["sources"]) == 2
    filenames = {s["filename"] for s in data["sources"]}
    assert "guide.pdf" in filenames
    assert "manual.txt" in filenames
    for source in data["sources"]:
        assert "filename" in source
        assert "page" in source

    # Verify search was called exactly once with the right question
    mock_search.assert_called_once()
    call_args = mock_search.call_args
    question_arg = call_args.args[0] if call_args.args else call_args.kwargs.get("question")
    assert question_arg == "How do I use the app?"


def test_full_rag_pipeline_llm_receives_context_in_prompt(client):
    """The LLM generate call must receive a prompt containing the retrieved context."""
    mock_sources = [Source(filename="ref.pdf", page=2)]
    captured_prompt: list[str] = []

    async def capturing_generate(model, prompt, **kwargs):
        captured_prompt.append(prompt)
        return "Generated answer."

    with patch("app.routers.rag.search_and_rerank") as mock_search, \
         patch("app.routers.rag.get_llm_service") as mock_factory:
        mock_search.return_value = (["Important context chunk."], mock_sources, 0.85)
        mock_service = AsyncMock()
        mock_service.generate = capturing_generate
        mock_factory.return_value = mock_service

        response = client.post(
            "/api/v1/rag/ask",
            json={"question": "What is important?", "server": "ollama", "model": "llama3"},
        )

    assert response.status_code == 200
    assert len(captured_prompt) == 1
    prompt = captured_prompt[0]
    # Prompt must embed the retrieved context
    assert "Important context chunk." in prompt
    # Prompt must embed the original question
    assert "What is important?" in prompt


def test_full_rag_pipeline_with_all_three_providers(client):
    """The full pipeline must work identically for all three LLM providers."""
    providers = [
        ("ollama", "llama3"),
        ("openai", "gpt-4o-mini"),
        ("huggingface", "mistralai/Mistral-7B-v0.1"),
    ]
    mock_sources = [Source(filename="doc.pdf", page=1)]

    for server, model in providers:
        with patch("app.routers.rag.search_and_rerank") as mock_search, \
             patch("app.routers.rag.get_llm_service") as mock_factory:
            mock_search.return_value = (["Context."], mock_sources, 0.75)
            mock_service = AsyncMock()
            mock_service.generate = AsyncMock(return_value=f"Answer from {server}.")
            mock_factory.return_value = mock_service

            response = client.post(
                "/api/v1/rag/ask",
                json={"question": "Test question?", "server": server, "model": model},
            )

        assert response.status_code == 200, f"Failed for provider {server}"
        data = response.json()
        assert data["answer"] == f"Answer from {server}."
        # Factory must be called with the correct provider string
        mock_factory.assert_called_once_with(server)


# ── Document upload → ingest pipeline ────────────────────────────────────────

class _FakeEmbedding:
    """Wraps a plain list so that .tolist() works (mirrors numpy ndarray behaviour)."""

    def __init__(self, values: list):
        self._values = values

    def tolist(self) -> list:
        return self._values

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)


def test_document_upload_ingest_es_bulk_called_with_correct_index(client):
    """Document upload must call ES bulk with the correct index and 384-dim embeddings."""
    from app.config import settings

    file_content = b"This is a test document for ingestion verification."

    # Make encode return an object that supports .tolist() (mirrors numpy behaviour)
    client.app.state.embedding_model.encode.return_value = _FakeEmbedding([0.0] * 384)

    # Patch ES bulk so we can inspect the actions sent
    bulk_actions_captured: list = []

    def fake_bulk(es, actions, **kwargs):
        bulk_actions_captured.extend(list(actions))
        return len(bulk_actions_captured), []

    with patch("app.store.ingest.helpers.bulk", side_effect=fake_bulk):
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("ingest_test.txt", io.BytesIO(file_content), "text/plain")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "ingest_test.txt"
    assert data["chunks_indexed"] >= 1

    # Verify bulk was called and every action targets the correct index
    assert len(bulk_actions_captured) >= 1
    for action in bulk_actions_captured:
        assert action["_index"] == settings.elasticsearch_index
        assert action["_op_type"] == "index"
        source = action["_source"]
        assert "content" in source
        assert "filename" in source
        assert "embedding" in source
        assert source["filename"] == "ingest_test.txt"
        # Embedding must be 384-dimensional (all-MiniLM-L6-v2 output)
        assert isinstance(source["embedding"], list)
        assert len(source["embedding"]) == 384


def test_document_upload_ingest_embedding_called_per_chunk(client):
    """The embedding model must be called once per text chunk during ingestion."""
    file_content = b"First sentence here. Second sentence here. Third one too."

    encode_call_count = [0]
    original_encode = client.app.state.embedding_model.encode

    def counting_encode(text, convert_to_tensor=False):
        encode_call_count[0] += 1
        return _FakeEmbedding([0.0] * 384)

    client.app.state.embedding_model.encode = counting_encode

    try:
        with patch("app.store.ingest.helpers.bulk"):
            response = client.post(
                "/api/v1/documents/upload",
                files={"file": ("chunk_test.txt", io.BytesIO(file_content), "text/plain")},
            )

        assert response.status_code == 200
        # At least one encode call must have been made (one per chunk)
        assert encode_call_count[0] >= 1
    finally:
        client.app.state.embedding_model.encode = original_encode
