"""Tests for app.config.Settings — defaults and env-var overrides."""

import pytest


def test_default_ollama_url():
    """Default Ollama API URL must point to localhost."""
    from app.config import Settings

    s = Settings()
    assert s.ollama_api_url == "http://localhost:11434/api/generate"


def test_default_elasticsearch_url():
    """Default Elasticsearch URL must point to the Docker service name."""
    from app.config import Settings

    s = Settings()
    assert s.elasticsearch_url == "http://elasticsearch:9200"


def test_default_elasticsearch_index():
    """Default index name must be 'documents'."""
    from app.config import Settings

    s = Settings()
    assert s.elasticsearch_index == "documents"


def test_default_embedding_model():
    """Default embedding model must be all-MiniLM-L6-v2."""
    from app.config import Settings

    s = Settings()
    assert s.embedding_model == "all-MiniLM-L6-v2"


def test_default_rerank_model():
    """Default rerank model must be cross-encoder/ms-marco-MiniLM-L-2-v2."""
    from app.config import Settings

    s = Settings()
    assert s.rerank_model == "cross-encoder/ms-marco-MiniLM-L-2-v2"


def test_default_top_k():
    """Default top_k must be 10."""
    from app.config import Settings

    s = Settings()
    assert s.top_k == 10


def test_default_top_n():
    """Default top_n must be 3."""
    from app.config import Settings

    s = Settings()
    assert s.top_n == 3


def test_default_api_keys_are_empty_strings():
    """Default openai_api_key and huggingface_api_token must be empty strings."""
    from app.config import Settings

    s = Settings()
    assert s.openai_api_key == ""
    assert s.huggingface_api_token == ""


def test_top_k_is_positive_integer():
    """top_k must be a positive integer in the default config."""
    from app.config import Settings

    s = Settings()
    assert isinstance(s.top_k, int)
    assert s.top_k > 0


def test_top_n_is_positive_integer():
    """top_n must be a positive integer in the default config."""
    from app.config import Settings

    s = Settings()
    assert isinstance(s.top_n, int)
    assert s.top_n > 0


def test_settings_read_top_k_from_env(monkeypatch):
    """Settings must read top_k from the TOP_K environment variable."""
    monkeypatch.setenv("TOP_K", "25")
    from importlib import reload
    import app.config as config_module

    reload(config_module)
    assert config_module.Settings().top_k == 25
    reload(config_module)  # restore for other tests


def test_settings_read_top_n_from_env(monkeypatch):
    """Settings must read top_n from the TOP_N environment variable."""
    monkeypatch.setenv("TOP_N", "7")
    from importlib import reload
    import app.config as config_module

    reload(config_module)
    assert config_module.Settings().top_n == 7
    reload(config_module)


def test_settings_read_elasticsearch_index_from_env(monkeypatch):
    """Settings must read elasticsearch_index from the ELASTICSEARCH_INDEX env var."""
    monkeypatch.setenv("ELASTICSEARCH_INDEX", "my_custom_index")
    from importlib import reload
    import app.config as config_module

    reload(config_module)
    assert config_module.Settings().elasticsearch_index == "my_custom_index"
    reload(config_module)


def test_settings_read_ollama_url_from_env(monkeypatch):
    """Settings must read ollama_api_url from the OLLAMA_API_URL env var."""
    monkeypatch.setenv("OLLAMA_API_URL", "http://custom-ollama:11434/api/generate")
    from importlib import reload
    import app.config as config_module

    reload(config_module)
    assert config_module.Settings().ollama_api_url == "http://custom-ollama:11434/api/generate"
    reload(config_module)


def test_settings_read_openai_api_key_from_env(monkeypatch):
    """Settings must read openai_api_key from the OPENAI_API_KEY env var."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-xyz")
    from importlib import reload
    import app.config as config_module

    reload(config_module)
    assert config_module.Settings().openai_api_key == "sk-test-key-xyz"
    reload(config_module)


def test_settings_read_huggingface_token_from_env(monkeypatch):
    """Settings must read huggingface_api_token from the HUGGINGFACE_API_TOKEN env var."""
    monkeypatch.setenv("HUGGINGFACE_API_TOKEN", "hf_test_token")
    from importlib import reload
    import app.config as config_module

    reload(config_module)
    assert config_module.Settings().huggingface_api_token == "hf_test_token"
    reload(config_module)
