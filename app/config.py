from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    ollama_api_url: str = "http://localhost:11434/api/generate"
    openai_api_key: str = ""
    huggingface_api_token: str = ""
    elasticsearch_url: str = "http://elasticsearch:9200"
    elasticsearch_index: str = "documents"
    embedding_model: str = "all-MiniLM-L6-v2"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-2-v2"
    top_k: int = 10
    top_n: int = 3

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


settings = Settings()
