from fastapi import APIRouter, HTTPException, Request
from loguru import logger

from app.exceptions import LLMProviderError, VectorStoreError
from app.models.schemas import QuestionRequest, RAGResponse
from app.services.factory import get_llm_service
from app.store.elasticsearch import search_and_rerank

router = APIRouter(tags=["RAG"])

_PROMPT_TEMPLATE = (
    "Actua como un asistente amigable para personas mayores. "
    "Aqui hay informacion relevante:\n{context}\n\n"
    "Esta informacion fue extraida de los siguientes archivos y paginas:\n{sources_info}\n\n"
    "Por favor, responde la siguiente pregunta de manera clara y simple, "
    "proporcionando instrucciones paso a paso si es necesario.\n\n"
    "Pregunta: {question}\n"
    "Respuesta (maximo 2000 caracteres):"
)

_NO_RESULTS_PROMPT = (
    "Hola, soy un asistente conversacional disenado para ayudar a las personas "
    "con tareas basicas de uso de su telefono movil.\n"
    "No encontre informacion relevante en los documentos disponibles para tu pregunta: {question}"
)


@router.post("/ask", response_model=RAGResponse)
async def ask(request: QuestionRequest, req: Request) -> RAGResponse:
    embedding_model = req.app.state.embedding_model
    rerank_model = req.app.state.rerank_model
    es = req.app.state.es

    try:
        top_texts, top_sources = search_and_rerank(
            request.question,
            embedding_model,
            rerank_model,
            es,
        )
    except VectorStoreError as exc:
        logger.error(f"Vector store error: {exc}")
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if not top_texts:
        logger.info(f"No relevant texts found for: {request.question!r}")
        return RAGResponse(
            answer="Lo siento, no se encontraron respuestas relevantes a tu pregunta.",
            sources=[],
        )

    sources_info = "\n".join(
        f"Archivo: {s.filename}, Pagina: {s.page}" for s in top_sources
    )
    context = "\n".join(top_texts)
    prompt = _PROMPT_TEMPLATE.format(
        context=context,
        sources_info=sources_info,
        question=request.question,
    )

    try:
        llm = get_llm_service(request.server)
        answer = await llm.generate(request.model, prompt)
    except LLMProviderError as exc:
        logger.error(f"LLM provider error: {exc}")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception(f"Unexpected error during LLM generation: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    logger.info(
        f"RAG ask | question={request.question!r} | sources={len(top_sources)} | answer_len={len(answer)}"
    )
    return RAGResponse(answer=answer, sources=top_sources)
