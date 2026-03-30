from app.services.base import LLMService
from app.services.huggingface import HuggingFaceService
from app.services.ollama import OllamaService
from app.services.openai import OpenAIService

_REGISTRY: dict[str, type[LLMService]] = {
    "ollama": OllamaService,
    "openai": OpenAIService,
    "huggingface": HuggingFaceService,
}


def get_llm_service(provider: str) -> LLMService:
    """Return an LLMService instance for the given provider name.

    Args:
        provider: One of "ollama", "openai", or "huggingface".

    Raises:
        ValueError: If the provider is not recognised.
    """
    service_class = _REGISTRY.get(provider)
    if service_class is None:
        valid = ", ".join(sorted(_REGISTRY))
        raise ValueError(
            f"Unknown LLM provider '{provider}'. Valid options: {valid}."
        )
    return service_class()
