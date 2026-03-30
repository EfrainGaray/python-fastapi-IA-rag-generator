from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.models.schemas import HealthCheckDetail, HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(req: Request) -> JSONResponse:
    checks: dict[str, str] = {}

    # Check Elasticsearch
    try:
        es = req.app.state.es
        checks["elasticsearch"] = "ok" if es.ping() else "error"
    except Exception:
        checks["elasticsearch"] = "error"

    # Check embedding model with a trivial encode
    try:
        embedding_model = req.app.state.embedding_model
        embedding_model.encode("")
        checks["embedding_model"] = "ok"
    except Exception:
        checks["embedding_model"] = "error"

    overall = "ok" if all(v == "ok" for v in checks.values()) else "degraded"
    status_code = 200 if overall == "ok" else 503

    body = HealthResponse(
        status=overall,
        checks=HealthCheckDetail(**checks),
        version="2.0.0",
    )
    return JSONResponse(content=body.model_dump(), status_code=status_code)
