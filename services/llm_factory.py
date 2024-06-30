# interfaces/llm_service.py

from services.ollama_service import OllamaService
from services.openai_service import OpenAIService
from services.huggingface_service import HuggingFaceService  # Importar el nuevo servicio
from interfaces.llm_service import LLMService

class LLMFactory:
    _services = {
        "ollama": OllamaService,
        "openai": OpenAIService,
        "huggingface": HuggingFaceService,  # Incluir el nuevo servicio aquí
    }

    @staticmethod
    def get_llm_service(provider: str) -> LLMService:
        service_class = LLMFactory._services.get(provider)
        if not service_class:
            raise ValueError(f"Invalid LLM provider specified: {provider}")
        return service_class()
