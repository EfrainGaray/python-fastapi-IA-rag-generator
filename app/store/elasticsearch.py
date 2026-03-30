from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from app.config import settings
from app.exceptions import VectorStoreError
from app.models.schemas import Source

if TYPE_CHECKING:
    from elasticsearch import Elasticsearch
    from sentence_transformers import CrossEncoder, SentenceTransformer


def search_and_rerank(
    question: str,
    embedding_model: Any,
    rerank_model: Any,
    es: Any,
) -> tuple[list[str], list[Source]]:
    """Embed question, retrieve top-K docs from ES, rerank, return top-N."""
    try:
        question_embedding: list[float] = (
            embedding_model.encode(question, convert_to_tensor=False).tolist()
        )
    except Exception as exc:
        raise VectorStoreError(f"Embedding generation failed: {exc}") from exc

    try:
        search_response: dict[str, Any] = es.search(
            index=settings.elasticsearch_index,
            body={
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": (
                                "cosineSimilarity(params.query_vector, 'embedding') + 1.0"
                            ),
                            "params": {"query_vector": question_embedding},
                        },
                    }
                },
                "size": settings.top_k,
            },
        )
    except Exception as exc:
        raise VectorStoreError(f"Elasticsearch query failed: {exc}") from exc

    hits = search_response["hits"]["hits"]
    if not hits:
        logger.info("No documents found in Elasticsearch for the query.")
        return [], []

    texts = [hit["_source"]["content"] for hit in hits]
    raw_sources = [
        (hit["_source"]["filename"], hit["_source"].get("page_number", "N/A"))
        for hit in hits
    ]

    rerank_inputs = [[question, text] for text in texts]
    scores = rerank_model.predict(rerank_inputs)

    ranked = sorted(
        zip(scores, texts, raw_sources), key=lambda x: x[0], reverse=True
    )

    top_texts = [r[1] for r in ranked[: settings.top_n]]
    top_sources = [
        Source(filename=r[2][0], page=r[2][1]) for r in ranked[: settings.top_n]
    ]

    return top_texts, top_sources
