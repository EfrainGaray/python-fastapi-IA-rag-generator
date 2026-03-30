from typing import Literal
from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    server: Literal["ollama", "openai", "huggingface"]
    model: str = Field(..., min_length=1)
    stream: bool = False


class Source(BaseModel):
    filename: str
    page: int | str


class StreamChunk(BaseModel):
    token: str
    done: bool
    sources: list[Source] = []


class RAGResponse(BaseModel):
    answer: str
    sources: list[Source]
    latency_ms: float = 0.0
    chunks_retrieved: int = 0
    top_score: float | None = None


class DirectResponse(BaseModel):
    answer: str
    latency_ms: float = 0.0


class HealthResponse(BaseModel):
    status: str


# ── Document management schemas ───────────────────────────────────────────────

class DocumentInfo(BaseModel):
    filename: str
    chunks: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total: int


class UploadResponse(BaseModel):
    filename: str
    chunks_indexed: int
    message: str


# ── Evaluation schemas ────────────────────────────────────────────────────────

class EvalRequest(BaseModel):
    question: str
    expected_answer: str
    server: Literal["ollama", "openai", "huggingface"]
    model: str


class EvalScores(BaseModel):
    answer_similarity: float
    source_coverage: float
    latency_ms: float


class EvalResponse(BaseModel):
    answer: str
    sources: list[Source]
    scores: EvalScores
