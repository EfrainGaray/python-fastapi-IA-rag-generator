import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.ollama import OllamaService
from app.services.openai import OpenAIService
from app.services.huggingface import HuggingFaceService
from app.config import settings


@pytest.mark.asyncio
async def test_ollama_generate():
    service = OllamaService()
    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "Ollama reply"}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("app.services.ollama.httpx.AsyncClient", return_value=mock_client):
        result = await service.generate("llama3", "Tell me something.")

    assert result == "Ollama reply"
    mock_client.post.assert_called_once()
    call_kwargs = mock_client.post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs.args[1]
    assert payload["model"] == "llama3"
    assert payload["prompt"] == "Tell me something."
    assert payload["stream"] is False


@pytest.mark.asyncio
async def test_openai_generate():
    service = OpenAIService()

    mock_message = MagicMock()
    mock_message.content = "  OpenAI reply  "
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    mock_completions = AsyncMock()
    mock_completions.create = AsyncMock(return_value=mock_completion)
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions

    service._client = MagicMock()
    service._client.chat = mock_chat

    result = await service.generate("gpt-4o-mini", "Hello")
    assert result == "OpenAI reply"


@pytest.mark.asyncio
async def test_huggingface_generate():
    """HuggingFace service must NOT send [MASK] prefix and must use token from settings."""
    service = HuggingFaceService()

    mock_response = MagicMock()
    mock_response.json.return_value = [{"generated_text": "HF reply"}]
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("app.services.huggingface.httpx.AsyncClient", return_value=mock_client):
        result = await service.generate("mistralai/Mistral-7B-v0.1", "What is AI?")

    assert result == "HF reply"

    call_args = mock_client.post.call_args
    sent_payload = call_args.kwargs.get("json") or call_args.args[1]
    sent_headers = call_args.kwargs.get("headers") or call_args.args[2]

    # Must NOT contain [MASK] prefix
    assert not sent_payload["inputs"].startswith("[MASK]")
    assert sent_payload["inputs"] == "What is AI?"

    # Must use token from settings, not a hardcoded value
    assert "Authorization" in sent_headers
    assert sent_headers["Authorization"] == f"Bearer {settings.huggingface_api_token}"
    # Token must come from settings, not hardcoded (old token removed in v2)
