# services/huggingface_service.py

import requests
from config import settings
from interfaces.llm_service import LLMService

class HuggingFaceService(LLMService):
    def generate(self, model: str, prompt: str, max_tokens: int = 512) -> str:
        endpoint = f"https://api-inference.huggingface.co/models/{model}"
        headers = {
            "Authorization": f"Bearer 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",  # Aquí deberías manejar la autorización de Hugging Face adecuadamente
            "Content-Type": "application/json"
        }
        data = {
            "inputs": "[MASK]"+prompt,
            "parameters": {
                "max_length": max_tokens,
                "temperature": 0.2
            }
        }
        response = requests.post(endpoint, json=data, headers=headers)
        response.raise_for_status()
        return response.json()[0]['generated_text'].strip()  # Asegúrate de ajustar según la respuesta real de Hugging Face
