from uuid import UUID

from fastapi import APIRouter, Depends, Query

from src.shared.responses import ApiResponse, PaginatedResponse
from src.modules.auth.presentation.dependencies import get_current_user, require_role
from src.modules.auth.domain.models.enums import Role
from src.modules.auth.domain.models.user import User
from src.modules.predictions.domain.services.prediction_service import PredictionService
from src.modules.predictions.presentation.dependencies import get_prediction_service
from src.modules.predictions.presentation.dtos.requests import (
    CreatePredictionRequest,
    DoctorDecisionRequest,
)
from src.modules.predictions.presentation.dtos.responses import PredictionResponse

router = APIRouter(dependencies=[Depends(require_role(Role.ADMIN, Role.DOCTOR))])


@router.post("", status_code=201)
async def create_prediction(
    body: CreatePredictionRequest,
    current_user: User = Depends(get_current_user),
    service: PredictionService = Depends(get_prediction_service),
):
    """Run ML prediction + explanation on a medical record and store the result."""
    prediction = await service.create_prediction(
        medical_record_id=body.medical_record_id,
        patient_id=body.patient_id,
        doctor_id=current_user.id,
    )
    return ApiResponse.ok(
        value=PredictionResponse.from_prediction(prediction),
        message="Prediction created",
    )


@router.get("/{prediction_id}")
async def get_prediction(
    prediction_id: UUID,
    service: PredictionService = Depends(get_prediction_service),
):
    """Get a single prediction with full details."""
    prediction = await service.get_prediction(prediction_id)
    return ApiResponse.ok(value=PredictionResponse.from_prediction(prediction))


@router.patch("/{prediction_id}/decision")
async def record_decision(
    prediction_id: UUID,
    body: DoctorDecisionRequest,
    service: PredictionService = Depends(get_prediction_service),
):
    """Record the doctor's final decision (accept or override)."""
    prediction = await service.record_doctor_decision(
        prediction_id=prediction_id,
        decision=body.decision,
        final_treatment=body.final_treatment,
        doctor_notes=body.doctor_notes,
    )
    return ApiResponse.ok(
        value=PredictionResponse.from_prediction(prediction),
        message="Decision recorded",
    )


@router.get("/patient/{patient_id}")
async def get_patient_predictions(
    patient_id: UUID,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    service: PredictionService = Depends(get_prediction_service),
):
    """Get prediction history for a patient."""
    predictions, total = await service.get_patient_predictions(
        patient_id=patient_id, page=page, page_size=page_size,
    )
    return PaginatedResponse.ok(
        value=[PredictionResponse.from_prediction(p) for p in predictions],
        page=page,
        total=total,
        page_size=page_size,
    )