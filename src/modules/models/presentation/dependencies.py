"""
FastAPI dependencies for the models module.

The engine is constructed once at startup (see ``src/core/lifespan.py``) and
lives on ``app.state.engine``. Controllers depend on
``get_inference_engine`` to retrieve it. The dependency raises
``ServiceUnavailableException`` (503) if engine construction failed at
startup or the engine reports not-ready.
"""
from __future__ import annotations

from fastapi import Request

from src.shared.exceptions import ServiceUnavailableException
from src.shared.responses import ErrorDetail
from src.shared.inference import InferenceEngine


def get_inference_engine(request: Request) -> InferenceEngine:
    engine: InferenceEngine | None = getattr(request.app.state, "engine", None)
    if engine is None or not engine.ready:
        raise ServiceUnavailableException(
            message="Inference engine is not ready",
            error_detail=ErrorDetail(
                title="Service Unavailable",
                code="INFERENCE_ENGINE_UNAVAILABLE",
                status=503,
                details=["The inference engine failed to initialise at startup."],
            ),
        )
    return engine


__all__ = ["get_inference_engine"]
