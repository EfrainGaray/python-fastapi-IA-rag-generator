from pydantic_settings import BaseSettings
from pydantic import AnyUrl
from typing import List

class Settings(BaseSettings):
    ollama_api_url: AnyUrl
    openai_api_key: str
    huggingface_api_token: str
    class Config:
        env_file = ".env" 

settings = Settings()
