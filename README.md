# RAG API

A production-ready Retrieval-Augmented Generation (RAG) API built with FastAPI, Elasticsearch, and multiple LLM backends (Ollama, OpenAI, HuggingFace).

## Architecture

```
graph TD
    Client -->|POST /api/v1/rag/ask| RAGRouter
    Client -->|POST /api/v1/direct/ask| DirectRouter
    RAGRouter --> EmbeddingModel[SentenceTransformer]
    RAGRouter --> Elasticsearch
    RAGRouter --> CrossEncoder[CrossEncoder Reranker]
    RAGRouter --> LLMFactory
    DirectRouter --> LLMFactory
    LLMFactory -->|ollama| OllamaService
    LLMFactory -->|openai| OpenAIService
    LLMFactory -->|huggingface| HuggingFaceService
    OllamaService -->|httpx async| OllamaServer
    OpenAIService -->|async openai| OpenAIAPI
    HuggingFaceService -->|httpx async| HuggingFaceAPI
```

## Quick Start

### Prerequisites

- Docker and Docker Compose
- (Optional) A running Ollama instance for local LLM inference

### 1. Clone and configure

```bash
git clone <repo-url>
cd rag-modernize
cp .env.example .env
# Edit .env with your API keys / URLs
```

### 2. Start services

```bash
docker compose up --build
```

The API will be available at `http://localhost:8000`. Elasticsearch starts at `http://localhost:9200`.

### 3. Verify

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

## Configuration

All settings are read from environment variables (or a `.env` file in the project root):

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_API_URL` | `http://localhost:11434/api/generate` | Ollama generate endpoint |
| `OPENAI_API_KEY` | `""` | OpenAI API key |
| `HUGGINGFACE_API_TOKEN` | `""` | HuggingFace Inference API token |
| `ELASTICSEARCH_URL` | `http://elasticsearch:9200` | Elasticsearch connection URL |
| `ELASTICSEARCH_INDEX` | `documents` | Index name for document chunks |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model for embeddings |
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-2-v2` | CrossEncoder model for reranking |
| `TOP_K` | `10` | Number of candidates to retrieve from ES |
| `TOP_N` | `3` | Number of top results after reranking |

## API Reference

### GET /health

Health check.

```bash
curl http://localhost:8000/health
```

Response:
```json
{"status": "ok"}
```

### POST /api/v1/rag/ask

Retrieve relevant context from Elasticsearch, rerank, and generate an answer.

```bash
curl -X POST http://localhost:8000/api/v1/rag/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do I send a WhatsApp message?",
    "server": "ollama",
    "model": "llama3"
  }'
```

Response:
```json
{
  "answer": "To send a WhatsApp message, open the app and ...",
  "sources": [
    {"filename": "guide.pdf", "page": 4},
    {"filename": "faq.pdf", "page": 1}
  ]
}
```

### POST /api/v1/direct/ask

Direct LLM generation without retrieval.

```bash
curl -X POST http://localhost:8000/api/v1/direct/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is Python?",
    "server": "openai",
    "model": "gpt-4o-mini"
  }'
```

Response:
```json
{"answer": "Python is a high-level programming language ..."}
```

Valid values for `server`: `ollama`, `openai`, `huggingface`.

## Ingesting Documents

Place documents (PDF, DOCX, TXT, XLSX) in a directory, then run:

```bash
# Inside Docker
docker compose exec api python scripts/ingest.py /app/docs

# Locally (requires dependencies installed)
python scripts/ingest.py ./docs
```

The script tracks processed files in `.processed_files.txt` inside the docs directory to avoid re-indexing on subsequent runs.

## Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run with short traceback
pytest tests/ -v --tb=short
```

## Docker Deployment

```bash
# Build and start in detached mode
docker compose up --build -d

# View logs
docker compose logs -f api

# Ingest documents
docker compose exec api python scripts/ingest.py /app/docs

# Stop services
docker compose down
```

Models are loaded once at startup via FastAPI's `lifespan` and stored on `app.state`. To swap models, update the relevant environment variables and restart the container.
