from fastapi import FastAPI
from controllers.rag_controller import router as rag_router
from controllers.direct_search_controller import router as direct_search_router

app = FastAPI()

app.include_router(direct_search_router, prefix="/api/v1/direct", tags=["Direct Search"])
app.include_router(rag_router, prefix="/api/v1/rag", tags=["RAG"])
