from typing import Literal
from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    server: Literal["ollama", "openai", "huggingface"]
    model: str = Field(..., min_length=1)


class Source(BaseModel):
    filename: str
    page: int | str


class RAGResponse(BaseModel):
    answer: str
    sources: list[Source]


class DirectResponse(BaseModel):
    answer: str


class HealthResponse(BaseModel):
    status: str
