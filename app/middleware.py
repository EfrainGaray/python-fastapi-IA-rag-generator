import time
import uuid
from collections import defaultdict

from fastapi import Request
from fastapi.responses import JSONResponse
from loguru import logger

# ── Rate limiter state ────────────────────────────────────────────────────────
_RATE_LIMIT_REQUESTS = 60   # max requests per window
_RATE_LIMIT_WINDOW = 60.0   # seconds

# Maps IP -> list of request timestamps (sliding window)
_rate_limit_store: defaultdict[str, list[float]] = defaultdict(list)


async def rate_limit_middleware(request: Request, call_next):
    """Sliding-window rate limiter: max 60 requests/minute per IP."""
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    window_start = now - _RATE_LIMIT_WINDOW

    timestamps = _rate_limit_store[client_ip]
    # Drop timestamps outside the sliding window
    timestamps[:] = [t for t in timestamps if t > window_start]

    if len(timestamps) >= _RATE_LIMIT_REQUESTS:
        oldest = timestamps[0]
        retry_after = int(_RATE_LIMIT_WINDOW - (now - oldest)) + 1
        return JSONResponse(
            status_code=429,
            content={"detail": f"Rate limit exceeded. Try again in {retry_after} seconds."},
        )

    timestamps.append(now)
    return await call_next(request)


async def observability_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = round((time.perf_counter() - start) * 1000, 1)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Latency-Ms"] = str(latency_ms)
    logger.info(
        f"req_id={request_id} method={request.method} path={request.url.path} "
        f"status={response.status_code} latency_ms={latency_ms}"
    )
    return response
