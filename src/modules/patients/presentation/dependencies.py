from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.shared.database import get_db
from src.modules.patients.domain.repositories.patient_repository import PatientRepository
from src.modules.patients.domain.repositories.medical_record_repository import MedicalRecordRepository
from src.modules.patients.domain.services.patient_service import PatientService


def get_patient_service(session: AsyncSession = Depends(get_db)) -> PatientService:
    return PatientService(
        patient_repo=PatientRepository(session),
        medical_record_repo=MedicalRecordRepository(session),
    )