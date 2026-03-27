from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from src.shared.database import get_db
from src.shared.inference import InferenceEngine
from src.modules.models.presentation.dependencies import get_inference_engine
from src.modules.predictions.domain.repositories.prediction_repository import PredictionRepository
from src.modules.predictions.domain.services.prediction_service import PredictionService
from src.modules.patients.domain.repositories.medical_record_repository import MedicalRecordRepository


def get_prediction_service(
    session: AsyncSession = Depends(get_db),
    engine: InferenceEngine = Depends(get_inference_engine),
) -> PredictionService:
    return PredictionService(
        prediction_repo=PredictionRepository(session),
        medical_record_repo=MedicalRecordRepository(session),
        engine=engine,
    )
