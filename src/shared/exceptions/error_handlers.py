"""
Global error handlers for FastAPI application.
Catches all exceptions and returns consistent API responses.
"""

import logging
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from src.shared.responses.api_response import ApiResponse, ErrorDetail
from src.shared.exceptions.exceptions import AppException
from src.shared.inference import (
    ConfigurationError as InferenceConfigurationError,
    ExplanationError as InferenceExplanationError,
    InferenceError,
    ModelError as InferenceModelError,
    ValidationError as InferenceValidationError,
)

logger = logging.getLogger(__name__)


def register_error_handlers(app: FastAPI) -> None:

    @app.exception_handler(AppException)
    async def handle_app_exception(_req: Request, exc: AppException):
        response = ApiResponse.failure(error=exc.error_detail, message=exc.message)
        return JSONResponse(
            status_code=exc.error_detail.status,
            content=response.model_dump(exclude_none=True, by_alias=True),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_request_validation_error(_req: Request, exc: RequestValidationError):
        builder = ErrorDetail.builder("Validation Failed", "VALIDATION_ERROR", 400)
        for error in exc.errors():
            field = ".".join(str(loc) for loc in error["loc"])
            builder.add_field_error(field, error["msg"])
        response = ApiResponse.failure(
            error=builder.build(),
            message="Please check your input and try again",
        )
        return JSONResponse(status_code=400, content=response.model_dump(exclude_none=True, by_alias=True))

    @app.exception_handler(ValidationError)
    async def handle_pydantic_validation_error(_req: Request, exc: ValidationError):
        builder = ErrorDetail.builder("Validation Failed", "VALIDATION_ERROR", 400)
        for error in exc.errors():
            field = ".".join(str(loc) for loc in error["loc"])
            builder.add_field_error(field, error["msg"])
        response = ApiResponse.failure(
            error=builder.build(),
            message="Please check your input and try again",
        )
        return JSONResponse(status_code=400, content=response.model_dump(exclude_none=True, by_alias=True))

    @app.exception_handler(InferenceValidationError)
    async def handle_inference_validation_error(_req: Request, exc: InferenceValidationError):
        builder = ErrorDetail.builder("Validation Failed", "INFERENCE_VALIDATION_ERROR", 422)
        for error in exc.errors():
            loc = error.get("loc") or []
            field = ".".join(str(p) for p in loc) or "_"
            builder.add_field_error(field, str(error.get("msg", "Invalid value")))
        if not exc.errors():
            builder.add_detail(str(exc) or "Invalid inference input")
        response = ApiResponse.failure(
            error=builder.build(),
            message="Please check your input and try again",
        )
        return JSONResponse(status_code=422, content=response.model_dump(exclude_none=True, by_alias=True))

    @app.exception_handler(InferenceConfigurationError)
    async def handle_inference_configuration_error(_req: Request, exc: InferenceConfigurationError):
        logger.error("Inference configuration error: %s", exc, exc_info=True)
        error = ErrorDetail(
            title="Service Unavailable",
            code="INFERENCE_CONFIGURATION_ERROR",
            status=503,
            details=[str(exc) or "Inference engine is misconfigured."],
        )
        response = ApiResponse.failure(
            error=error,
            message="The service is temporarily unavailable",
        )
        return JSONResponse(status_code=503, content=response.model_dump(exclude_none=True, by_alias=True))

    @app.exception_handler(InferenceModelError)
    async def handle_inference_model_error(_req: Request, exc: InferenceModelError):
        logger.error("Inference model error: %s", exc, exc_info=True)
        error = ErrorDetail(
            title="Inference Failed",
            code="INFERENCE_MODEL_ERROR",
            status=500,
            details=[str(exc) or "The model failed during inference."],
        )
        response = ApiResponse.failure(
            error=error,
            message="Something went wrong. Please try again later",
        )
        return JSONResponse(status_code=500, content=response.model_dump(exclude_none=True, by_alias=True))

    @app.exception_handler(InferenceExplanationError)
    async def handle_inference_explanation_error(_req: Request, exc: InferenceExplanationError):
        exc_str = str(exc)
        is_key_error = (
            "API key" in exc_str
            or "API_KEY_INVALID" in exc_str
            or "LLM generate call failed" in exc_str
        )
        if is_key_error:
            logger.error("LLM dependency error: %s", exc)
            error = ErrorDetail(
                title="AI Explanation Unavailable",
                code="LLM_UNAVAILABLE",
                status=503,
                details=["The AI explanation service is temporarily unavailable. "
                         "Please contact your administrator."],
            )
            response = ApiResponse.failure(
                error=error,
                message="AI explanation service is unavailable",
            )
            return JSONResponse(status_code=503, content=response.model_dump(exclude_none=True, by_alias=True))

        logger.error("Inference explanation error: %s", exc, exc_info=True)
        error = ErrorDetail(
            title="Explanation Failed",
            code="INFERENCE_EXPLANATION_ERROR",
            status=500,
            details=[exc_str or "The explanation step failed."],
        )
        response = ApiResponse.failure(
            error=error,
            message="Prediction explanation could not be generated",
        )
        return JSONResponse(status_code=500, content=response.model_dump(exclude_none=True, by_alias=True))

    @app.exception_handler(InferenceError)
    async def handle_inference_error(_req: Request, exc: InferenceError):
        logger.error("Unclassified inference error: %s", exc, exc_info=True)
        error = ErrorDetail(
            title="Inference Error",
            code="INFERENCE_ERROR",
            status=500,
            details=[str(exc) or "An inference error occurred."],
        )
        response = ApiResponse.failure(
            error=error,
            message="Something went wrong. Please try again later",
        )
        return JSONResponse(status_code=500, content=response.model_dump(exclude_none=True, by_alias=True))

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(req: Request, exc: StarletteHTTPException):
        user_messages = {
            400: "Please check your request and try again",
            401: "Please log in to continue",
            403: "You don't have permission to perform this action",
            404: "The page you're looking for doesn't exist",
            405: "This action is not allowed",
            500: "Something went wrong. Please try again later",
            503: "The service is temporarily unavailable",
        }
        detail_messages = {
            404: f"{req.method} {req.url.path} was not found",
            405: f"{req.method} is not allowed for {req.url.path}",
        }
        error = ErrorDetail(
            title=str(exc.detail),
            code=str(exc.detail).upper().replace(" ", "_"),
            status=exc.status_code,
            details=[detail_messages.get(exc.status_code, str(exc.detail))],
        )
        response = ApiResponse.failure(
            error=error,
            message=user_messages.get(exc.status_code, "An error occurred. Please try again"),
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=response.model_dump(exclude_none=True, by_alias=True),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_req: Request, exc: Exception):
        logger.error(f"Unexpected error: {exc}", exc_info=True)
        error = ErrorDetail(
            title="Internal Server Error",
            code="INTERNAL_ERROR",
            status=500,
            details=["An unexpected error occurred. Please try again later."],
        )
        response = ApiResponse.failure(
            error=error,
            message="Something went wrong. Please try again later",
        )
        return JSONResponse(status_code=500, content=response.model_dump(exclude_none=True, by_alias=True))