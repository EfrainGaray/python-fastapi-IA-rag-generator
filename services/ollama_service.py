import requests
from config import settings
from interfaces.llm_service import LLMService

class OllamaService(LLMService):
    def generate(self, model: str, prompt: str, max_tokens: int = 512) -> str:
        response = requests.post(settings.ollama_api_url, json={
            'model': model, 
            'prompt': prompt, 
            'stream': False,
            'max_tokens': max_tokens,
            'temperature': 0.2
        })
        response.raise_for_status()
        return response.json()['response']
