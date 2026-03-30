from fastapi import APIRouter, HTTPException
from loguru import logger

from app.exceptions import LLMProviderError
from app.models.schemas import DirectResponse, QuestionRequest
from app.services.factory import get_llm_service

router = APIRouter(tags=["Direct Search"])


@router.post("/ask", response_model=DirectResponse)
async def ask(request: QuestionRequest) -> DirectResponse:
    try:
        llm = get_llm_service(request.server)
        prompt = f"Question: {request.question}\nAnswer:"
        answer = await llm.generate(request.model, prompt)
        logger.info(f"Direct ask | question={request.question!r} | answer_len={len(answer)}")
        return DirectResponse(answer=answer)
    except LLMProviderError as exc:
        logger.error(f"LLM provider error: {exc}")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception(f"Unexpected error in direct ask: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
