"""Shared pytest fixtures for the RAG API test suite."""
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

# ── Mock heavy ML libraries before importing the app ──────────────────────────
# This allows tests to run without sentence_transformers installed.
_st_mock = MagicMock()
sys.modules.setdefault("sentence_transformers", _st_mock)
sys.modules.setdefault("sentence_transformers.cross_encoder", _st_mock)

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.models.schemas import Source  # noqa: E402


@pytest.fixture()
def mock_embedding_model():
    """SentenceTransformer stub — returns a fixed 384-dim list."""
    model = MagicMock()
    model.encode.return_value = [0.0] * 384
    return model


@pytest.fixture()
def mock_rerank_model():
    """CrossEncoder stub — returns descending scores so ordering is stable."""
    model = MagicMock()
    model.predict.return_value = [0.9, 0.8, 0.7]
    return model


@pytest.fixture()
def mock_es():
    """Elasticsearch client stub that returns three fake hits."""
    es = MagicMock()
    es.search.return_value = {
        "hits": {
            "hits": [
                {"_source": {"content": "Chunk one.", "filename": "doc1.pdf", "page_number": 1}},
                {"_source": {"content": "Chunk two.", "filename": "doc2.pdf", "page_number": 2}},
                {"_source": {"content": "Chunk three.", "filename": "doc3.pdf", "page_number": 3}},
            ]
        }
    }
    return es


@pytest.fixture()
def mock_es_empty():
    """Elasticsearch client stub that returns no hits."""
    es = MagicMock()
    es.search.return_value = {"hits": {"hits": []}}
    return es


@pytest.fixture()
def client(mock_embedding_model, mock_rerank_model, mock_es):
    """TestClient with mocked app.state — no real models or ES needed."""
    app.state.embedding_model = mock_embedding_model
    app.state.rerank_model = mock_rerank_model
    app.state.es = mock_es
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture()
def client_empty_es(mock_embedding_model, mock_rerank_model, mock_es_empty):
    """TestClient whose ES always returns zero hits."""
    app.state.embedding_model = mock_embedding_model
    app.state.rerank_model = mock_rerank_model
    app.state.es = mock_es_empty
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
