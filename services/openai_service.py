import os
from openai import OpenAI
from config import settings
from interfaces.llm_service import LLMService

class OpenAIService(LLMService):
    def __init__(self):
        self.client = OpenAI(
            api_key=settings.openai_api_key
        )

    def generate(self, model: str, prompt: str, max_tokens: int = 512) -> str:
        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model,
            max_tokens=max_tokens,
            temperature=0.2
        )
        return response.choices[0].message.content.strip()

