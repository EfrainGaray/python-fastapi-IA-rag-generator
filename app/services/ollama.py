import httpx
from loguru import logger

from app.config import settings
from app.exceptions import LLMProviderError
from app.services.base import LLMService


class OllamaService(LLMService):
    async def generate(self, model: str, prompt: str, max_tokens: int = 512) -> str:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.2,
            },
        }
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(settings.ollama_api_url, json=payload)
                r.raise_for_status()
                return r.json()["response"]
        except httpx.HTTPStatusError as exc:
            logger.error(f"Ollama HTTP error {exc.response.status_code}: {exc.response.text}")
            raise LLMProviderError(f"Ollama returned HTTP {exc.response.status_code}") from exc
        except httpx.RequestError as exc:
            logger.error(f"Ollama request error: {exc}")
            raise LLMProviderError(f"Cannot reach Ollama at {settings.ollama_api_url}") from exc
