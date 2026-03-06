import time
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from src.configs import server

logger = logging.getLogger(__name__)


def register_middleware(app: FastAPI) -> None:
    _add_cors(app)
    _add_request_logging(app)


def _add_cors(app: FastAPI) -> None:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=server.cors.origins,
        allow_credentials=server.cors.allow_credentials,
        allow_methods=server.cors.allow_methods,
        allow_headers=server.cors.allow_headers,
    )


def _add_request_logging(app: FastAPI) -> None:
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = round(time.perf_counter() - start, 4)
        logger.info(
            f"{request.method} {request.url.path} — {response.status_code} ({elapsed}s)"
        )
        return response