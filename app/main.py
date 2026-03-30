from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger

from app.config import settings
from app.middleware import observability_middleware
from app.routers import direct, health, rag
from app.routers import documents, eval as eval_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy singletons once at startup; release them on shutdown."""
    from elasticsearch import Elasticsearch
    from sentence_transformers import CrossEncoder, SentenceTransformer

    logger.info(f"Loading embedding model: {settings.embedding_model}")
    app.state.embedding_model = SentenceTransformer(settings.embedding_model)

    logger.info(f"Loading rerank model: {settings.rerank_model}")
    app.state.rerank_model = CrossEncoder(settings.rerank_model)

    logger.info(f"Connecting to Elasticsearch: {settings.elasticsearch_url}")
    app.state.es = Elasticsearch(settings.elasticsearch_url)

    logger.info("Application startup complete.")
    yield

    logger.info("Shutting down — closing Elasticsearch connection.")
    app.state.es.close()


app = FastAPI(
    title="RAG API",
    version="2.0.0",
    description="Retrieval-Augmented Generation API with Elasticsearch and multiple LLM backends.",
    lifespan=lifespan,
)

app.add_middleware(BaseHTTPMiddleware, dispatch=observability_middleware)

app.include_router(health.router)
app.include_router(direct.router, prefix="/api/v1/direct")
app.include_router(rag.router, prefix="/api/v1/rag")
app.include_router(documents.router, prefix="/api/v1/documents")
app.include_router(eval_router.router, prefix="/api/v1/rag")
