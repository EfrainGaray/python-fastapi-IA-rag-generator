import httpx
from loguru import logger

from app.config import settings
from app.exceptions import LLMProviderError
from app.services.base import LLMService


class HuggingFaceService(LLMService):
    _BASE_URL = "https://api-inference.huggingface.co/models"

    async def generate(self, model: str, prompt: str, max_tokens: int = 512) -> str:
        endpoint = f"{self._BASE_URL}/{model}"
        headers = {
            "Authorization": f"Bearer {settings.huggingface_api_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": 0.2,
                "return_full_text": False,
            },
        }
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(endpoint, json=payload, headers=headers)
                r.raise_for_status()
                data = r.json()
                if isinstance(data, list) and data:
                    return data[0].get("generated_text", "").strip()
                return str(data).strip()
        except httpx.HTTPStatusError as exc:
            logger.error(
                f"HuggingFace HTTP error {exc.response.status_code}: {exc.response.text}"
            )
            raise LLMProviderError(
                f"HuggingFace returned HTTP {exc.response.status_code}"
            ) from exc
        except httpx.RequestError as exc:
            logger.error(f"HuggingFace request error: {exc}")
            raise LLMProviderError("Cannot reach HuggingFace inference API") from exc
