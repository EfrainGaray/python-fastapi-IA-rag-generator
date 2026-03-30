import json
import time
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from app.exceptions import LLMProviderError
from app.models.schemas import DirectResponse, QuestionRequest, StreamChunk
from app.services.factory import get_llm_service

router = APIRouter(tags=["Direct Search"])


@router.post("/ask", response_model=DirectResponse)
async def ask(request: QuestionRequest):
    t0 = time.perf_counter()
    prompt = f"Question: {request.question}\nAnswer:"

    if request.stream:
        async def event_generator() -> AsyncGenerator[str, None]:
            try:
                llm = get_llm_service(request.server)
                async for token in llm.stream_generate(request.model, prompt):
                    chunk = StreamChunk(token=token, done=False)
                    yield f"data: {chunk.model_dump_json()}\n\n"
                final_chunk = StreamChunk(token="", done=True)
                yield f"data: {final_chunk.model_dump_json()}\n\n"
            except LLMProviderError as exc:
                logger.error(f"LLM provider error during streaming: {exc}")
                error_data = json.dumps({"error": str(exc)})
                yield f"data: {error_data}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    try:
        llm = get_llm_service(request.server)
        answer = await llm.generate(request.model, prompt)
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        logger.info(f"Direct ask | question={request.question!r} | answer_len={len(answer)}")
        return DirectResponse(answer=answer, latency_ms=latency_ms)
    except LLMProviderError as exc:
        logger.error(f"LLM provider error: {exc}")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception(f"Unexpected error in direct ask: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
