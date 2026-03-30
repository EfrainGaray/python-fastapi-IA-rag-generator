from typing import AsyncGenerator

from openai import AsyncOpenAI
from loguru import logger

from app.config import settings
from app.exceptions import LLMProviderError
from app.services.base import LLMService


class OpenAIService(LLMService):
    def __init__(self) -> None:
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def generate(self, model: str, prompt: str, max_tokens: int = 512) -> str:
        try:
            response = await self._client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.error(f"OpenAI error: {exc}")
            raise LLMProviderError(f"OpenAI generation failed: {exc}") from exc

    async def stream_generate(
        self, model: str, prompt: str, max_tokens: int = 512
    ) -> AsyncGenerator[str, None]:
        try:
            stream = await self._client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.2,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta
                token = delta.content or ""
                if token:
                    yield token
        except Exception as exc:
            logger.error(f"OpenAI stream error: {exc}")
            raise LLMProviderError(f"OpenAI streaming failed: {exc}") from exc
