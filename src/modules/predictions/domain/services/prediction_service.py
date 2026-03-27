from typing import Sequence, Tuple
from uuid import UUID

from src.shared.inference import InferenceEngine
from src.modules.predictions.domain.models.prediction import Prediction
from src.modules.predictions.domain.models.enums import DoctorDecision
from src.modules.predictions.domain.repositories.prediction_repository import PredictionRepository
from src.modules.predictions.internal.medical_record_mapper import to_context
from src.modules.predictions.internal.prediction_mapper import to_prediction
from src.modules.patients.domain.repositories.medical_record_repository import MedicalRecordRepository
from src.shared.exceptions import NotFoundException, BadRequestException
from src.shared.responses import ErrorDetail


class PredictionService:
    def __init__(
        self,
        prediction_repo: PredictionRepository,
        medical_record_repo: MedicalRecordRepository,
        engine: InferenceEngine,
    ):
        self.prediction_repo = prediction_repo
        self.record_repo = medical_record_repo
        self.engine = engine

    async def create_prediction(
        self,
        medical_record_id: UUID,
        patient_id: UUID,
        doctor_id: UUID,
    ) -> Prediction:
        """
        Fetch medical record, run inference + explanation via the shared
        InferenceEngine, persist the structured result.
        """
        record = await self.record_repo.get_by_id(medical_record_id)
        if not record:
            raise NotFoundException(
                message="Medical record not found",
                error_detail=ErrorDetail(
                    title="Not Found", code="RECORD_NOT_FOUND", status=404,
                    details=[f"No medical record found with id {medical_record_id}"],
                ),
            )

        if record.patient_id != patient_id:
            raise BadRequestException(
                message="Medical record does not belong to this patient",
                error_detail=ErrorDetail(
                    title="Bad Request", code="RECORD_PATIENT_MISMATCH", status=400,
                    details=["The medical record does not belong to the specified patient"],
                ),
            )

        patient = to_context(record)
        result = await self.engine.apredict(patient, explain="require")

        prediction = to_prediction(
            result,
            medical_record_id=medical_record_id,
            patient_id=patient_id,
            doctor_id=doctor_id,
        )
        return await self.prediction_repo.create(prediction)

    async def record_doctor_decision(
        self,
        prediction_id: UUID,
        decision: str,
        final_treatment: str | None = None,
        doctor_notes: str | None = None,
    ) -> Prediction:
        """Record the doctor's final decision on a prediction."""
        prediction = await self._get_prediction(prediction_id)

        data = {"doctor_decision": DoctorDecision[decision]}

        if decision == "ACCEPTED":
            data["final_treatment"] = prediction.recommended_treatment
        elif decision == "OVERRIDDEN":
            if not final_treatment:
                raise BadRequestException(
                    message="Final treatment is required when overriding",
                    error_detail=ErrorDetail(
                        title="Bad Request", code="TREATMENT_REQUIRED", status=400,
                        details=["You must specify the final treatment when overriding the recommendation"],
                    ),
                )
            data["final_treatment"] = final_treatment

        if doctor_notes is not None:
            data["doctor_notes"] = doctor_notes

        return await self.prediction_repo.update(prediction, data)

    async def get_prediction(self, prediction_id: UUID) -> Prediction:
        return await self._get_prediction(prediction_id)

    async def get_patient_predictions(
        self, patient_id: UUID, page: int = 1, page_size: int = 20,
    ) -> Tuple[Sequence[Prediction], int]:
        return await self.prediction_repo.get_paginated_by_patient(
            patient_id, page, page_size,
        )

    async def _get_prediction(self, prediction_id: UUID) -> Prediction:
        prediction = await self.prediction_repo.get_by_id(prediction_id)
        if not prediction:
            raise NotFoundException(
                message="Prediction not found",
                error_detail=ErrorDetail(
                    title="Not Found", code="PREDICTION_NOT_FOUND", status=404,
                    details=[f"No prediction found with id {prediction_id}"],
                ),
            )
        return prediction
