import json
from typing import AsyncGenerator

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

    async def stream_generate(
        self, model: str, prompt: str, max_tokens: int = 512
    ) -> AsyncGenerator[str, None]:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.2,
            },
        }
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream(
                    "POST", settings.ollama_api_url, json=payload
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        token = data.get("response", "")
                        if token:
                            yield token
                        if data.get("done", False):
                            break
        except httpx.HTTPStatusError as exc:
            logger.error(f"Ollama stream HTTP error {exc.response.status_code}")
            raise LLMProviderError(f"Ollama returned HTTP {exc.response.status_code}") from exc
        except httpx.RequestError as exc:
            logger.error(f"Ollama stream request error: {exc}")
            raise LLMProviderError(f"Cannot reach Ollama at {settings.ollama_api_url}") from exc
