import pytest

from app.services.factory import get_llm_service
from app.services.huggingface import HuggingFaceService
from app.services.ollama import OllamaService
from app.services.openai import OpenAIService


def test_factory_ollama():
    service = get_llm_service("ollama")
    assert isinstance(service, OllamaService)


def test_factory_openai():
    service = get_llm_service("openai")
    assert isinstance(service, OpenAIService)


def test_factory_huggingface():
    service = get_llm_service("huggingface")
    assert isinstance(service, HuggingFaceService)


def test_factory_invalid():
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        get_llm_service("invalid_provider")
