from fastapi import APIRouter, Depends

from src.shared.responses import ApiResponse
from src.shared.exceptions import InternalServerException
from src.shared.responses import ErrorDetail
from src.modules.auth.presentation.dependencies import get_current_user, require_role
from src.modules.auth.domain.models.enums import Role
from src.modules.auth.domain.models.user import User
from src.modules.models.internal import inference_engine
from src.modules.models.presentation.dtos.requests import (
    PredictRequest,
    ExplainRequest,
    BatchPredictRequest,
)
from src.modules.models.presentation.dtos.responses import (
    PredictionResponse,
    ExplanationResponse,
    PredictionWithExplanationResponse,
)

router = APIRouter(dependencies=[Depends(require_role(Role.ADMIN, Role.DOCTOR))])


@router.post("/predict")
async def predict(
    body: PredictRequest,
    current_user: User = Depends(get_current_user),
):
    """Run prediction for a single patient. Returns recommendation + confidence + safety."""
    try:
        payload = inference_engine.predict(body.to_context())
    except FileNotFoundError as e:
        raise InternalServerException(
            message="Model files not found. Please check your model configuration.",
            error_detail=ErrorDetail(
                title="Model Not Found", code="MODEL_NOT_FOUND", status=500,
                details=[str(e)],
            ),
        )

    return ApiResponse.ok(
        value=PredictionResponse.from_payload(payload),
        message="Prediction successful",
    )


@router.post("/predict-with-explanation")
async def predict_with_explanation(
    body: PredictRequest,
    current_user: User = Depends(get_current_user),
):
    """Run prediction + LLM explanation for a single patient."""
    try:
        payload = inference_engine.predict_with_explanation(body.to_context())
    except FileNotFoundError as e:
        raise InternalServerException(
            message="Model files not found.",
            error_detail=ErrorDetail(
                title="Model Not Found", code="MODEL_NOT_FOUND", status=500,
                details=[str(e)],
            ),
        )

    return ApiResponse.ok(
        value=PredictionWithExplanationResponse.from_payload(payload),
        message="Prediction with explanation successful",
    )


@router.post("/explain")
async def explain(
    body: ExplainRequest,
    current_user: User = Depends(get_current_user),
):
    """Generate LLM explanation from an existing prediction payload."""
    payload = {"patient": body.patient, "decision": body.decision, "safety": body.safety, "fairness": body.fairness}

    try:
        explanation = inference_engine.explain(payload)
    except Exception as e:
        raise InternalServerException(
            message="Failed to generate explanation.",
            error_detail=ErrorDetail(
                title="Explanation Failed", code="EXPLANATION_FAILED", status=500,
                details=[str(e)],
            ),
        )

    return ApiResponse.ok(
        value=ExplanationResponse.from_payload(explanation),
        message="Explanation generated",
    )


@router.post("/predict-batch")
async def predict_batch(
    body: BatchPredictRequest,
    current_user: User = Depends(get_current_user),
):
    """Run predictions for multiple patients. Max 50."""
    try:
        contexts = [p.to_context() for p in body.patients]
        payloads = inference_engine.predict_batch(contexts)
    except FileNotFoundError as e:
        raise InternalServerException(
            message="Model files not found.",
            error_detail=ErrorDetail(
                title="Model Not Found", code="MODEL_NOT_FOUND", status=500,
                details=[str(e)],
            ),
        )

    return ApiResponse.ok(
        value=[PredictionResponse.from_payload(p) for p in payloads],
        message=f"Batch prediction complete: {len(payloads)} patients",
    )