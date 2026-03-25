import asyncio

from fastapi import APIRouter, Depends

from src.shared.responses import ApiResponse
from src.shared.inference import InferenceEngine, PatientInput
from src.modules.auth.presentation.dependencies import get_current_user, require_role
from src.modules.auth.domain.models.enums import Role
from src.modules.auth.domain.models.user import User
from src.modules.models.presentation.dependencies import get_inference_engine
from src.modules.models.presentation.dtos.requests import BatchPredictRequest

router = APIRouter(dependencies=[Depends(require_role(Role.ADMIN, Role.DOCTOR))])


@router.post("/predict")
async def predict(
    body: PatientInput,
    current_user: User = Depends(get_current_user),
    engine: InferenceEngine = Depends(get_inference_engine),
):
    """Run prediction for a single patient. Returns flat PredictionResult."""
    result = await engine.apredict(body, explain=False)
    return ApiResponse.ok(
        value=result.model_dump(mode="json"),
        message="Prediction successful",
    )


@router.post("/predict-with-explanation")
async def predict_with_explanation(
    body: PatientInput,
    current_user: User = Depends(get_current_user),
    engine: InferenceEngine = Depends(get_inference_engine),
):
    """Run prediction + LLM explanation for a single patient."""
    result = await engine.apredict(body, explain="require")
    return ApiResponse.ok(
        value=result.model_dump(mode="json"),
        message="Prediction with explanation successful",
    )


@router.post("/predict-batch")
async def predict_batch(
    body: BatchPredictRequest,
    current_user: User = Depends(get_current_user),
    engine: InferenceEngine = Depends(get_inference_engine),
):
    """Run predictions for multiple patients. Max 50."""
    results = await asyncio.to_thread(engine.predict_batch, body.patients)
    return ApiResponse.ok(
        value=[r.model_dump(mode="json") for r in results],
        message=f"Batch prediction complete: {len(results)} patients",
    )
