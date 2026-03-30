"""RAG evaluation endpoint — no RAGAS dependency."""

import time

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from loguru import logger

from app.exceptions import LLMProviderError, VectorStoreError
from app.models.schemas import EvalRequest, EvalResponse, EvalScores
from app.services.factory import get_llm_service
from app.store.elasticsearch import search_and_rerank

router = APIRouter(tags=["Evaluation"])

_PROMPT_TEMPLATE = (
    "Actua como un asistente amigable para personas mayores. "
    "Aqui hay informacion relevante:\n{context}\n\n"
    "Esta informacion fue extraida de los siguientes archivos y paginas:\n{sources_info}\n\n"
    "Por favor, responde la siguiente pregunta de manera clara y simple.\n\n"
    "Pregunta: {question}\n"
    "Respuesta (maximo 2000 caracteres):"
)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=float)
    vb = np.array(b, dtype=float)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


def _source_coverage(sources_texts: list[str], expected_answer: str) -> float:
    """Fraction of sources that contain at least one keyword from expected_answer."""
    if not sources_texts:
        return 0.0
    keywords = {
        w.lower().strip(".,;:!?\"'")
        for w in expected_answer.split()
        if len(w) > 3
    }
    if not keywords:
        return 0.0
    covered = sum(
        1
        for text in sources_texts
        if any(kw in text.lower() for kw in keywords)
    )
    return round(covered / len(sources_texts), 4)


@router.post("/eval", response_model=EvalResponse)
async def eval_rag(request: EvalRequest, req: Request) -> EvalResponse:
    embedding_model = req.app.state.embedding_model
    rerank_model = req.app.state.rerank_model
    es = req.app.state.es

    t0 = time.perf_counter()

    # Retrieve context
    try:
        top_texts, top_sources, _ = search_and_rerank(
            request.question, embedding_model, rerank_model, es
        )
    except VectorStoreError as exc:
        logger.error(f"Vector store error in eval: {exc}")
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if top_texts:
        sources_info = "\n".join(
            f"Archivo: {s.filename}, Pagina: {s.page}" for s in top_sources
        )
        context = "\n".join(top_texts)
        prompt = _PROMPT_TEMPLATE.format(
            context=context,
            sources_info=sources_info,
            question=request.question,
        )
    else:
        prompt = f"Question: {request.question}\nAnswer:"

    # Generate answer
    try:
        llm = get_llm_service(request.server)
        answer = await llm.generate(request.model, prompt)
    except LLMProviderError as exc:
        logger.error(f"LLM provider error in eval: {exc}")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception(f"Unexpected error in eval: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    # Compute scores
    def _to_list(v):
        return v.tolist() if hasattr(v, "tolist") else list(v)

    expected_emb = _to_list(
        embedding_model.encode(request.expected_answer, convert_to_tensor=False)
    )
    answer_emb = _to_list(
        embedding_model.encode(answer, convert_to_tensor=False)
    )
    answer_similarity = round(_cosine_similarity(expected_emb, answer_emb), 4)

    source_cov = _source_coverage(top_texts, request.expected_answer)

    logger.info(
        f"Eval | question={request.question!r} | similarity={answer_similarity} | "
        f"coverage={source_cov} | latency_ms={latency_ms}"
    )

    return EvalResponse(
        answer=answer,
        sources=top_sources,
        scores=EvalScores(
            answer_similarity=answer_similarity,
            source_coverage=source_cov,
            latency_ms=latency_ms,
        ),
    )
