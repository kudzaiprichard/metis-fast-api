from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.shared.database import get_db
from src.modules.predictions.domain.repositories.prediction_repository import PredictionRepository
from src.modules.predictions.domain.services.prediction_service import PredictionService
from src.modules.patients.domain.repositories.medical_record_repository import MedicalRecordRepository


def get_prediction_service(session: AsyncSession = Depends(get_db)) -> PredictionService:
    return PredictionService(
        prediction_repo=PredictionRepository(session),
        medical_record_repo=MedicalRecordRepository(session),
    )