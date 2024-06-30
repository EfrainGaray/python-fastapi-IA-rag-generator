from fastapi import APIRouter, HTTPException
from models.request_models import QuestionRequest
from services.llm_factory import LLMFactory
from logger import get_logger

logger = get_logger()
router = APIRouter()

@router.post("/ask")
def ask(request: QuestionRequest):
    try:
        question = request.question
        server = request.server
        model = request.model

        # Generación de respuesta usando LLM
        llm_service = LLMFactory.get_llm_service(server)
        prompt = f"Question: {question}\nAnswer:"
        answer = llm_service.generate(model, prompt)

        logger.info(f"Question: {question} - Answer: {answer}")
        return {"answer": answer}

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
