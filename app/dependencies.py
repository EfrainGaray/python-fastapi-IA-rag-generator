"""FastAPI dependency helpers that pull singletons from app.state.

All heavy objects (SentenceTransformer, CrossEncoder, Elasticsearch) are
loaded once during the lifespan and stored on ``app.state``.  The functions
below expose them as injectable FastAPI dependencies.
"""

from fastapi import Request
from elasticsearch import Elasticsearch
from sentence_transformers import CrossEncoder, SentenceTransformer


def get_embedding_model(request: Request) -> SentenceTransformer:
    return request.app.state.embedding_model


def get_rerank_model(request: Request) -> CrossEncoder:
    return request.app.state.rerank_model


def get_es_client(request: Request) -> Elasticsearch:
    return request.app.state.es
