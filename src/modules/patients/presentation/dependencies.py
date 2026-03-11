from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.shared.database import get_db
from src.shared.neo4j import get_neo4j_db
from src.shared.neo4j.neo4j_graph_database import Neo4jGraphDatabase
from src.modules.patients.domain.repositories.patient_repository import PatientRepository
from src.modules.patients.domain.repositories.medical_record_repository import MedicalRecordRepository
from src.modules.patients.domain.services.patient_service import PatientService
from src.modules.patients.domain.services.similar_patient_service import SimilarPatientService


def get_patient_service(session: AsyncSession = Depends(get_db)) -> PatientService:
    return PatientService(
        patient_repo=PatientRepository(session),
        medical_record_repo=MedicalRecordRepository(session),
    )


def get_similar_patient_service(
    session: AsyncSession = Depends(get_db),
    neo4j_db: Neo4jGraphDatabase = Depends(get_neo4j_db),
) -> SimilarPatientService:
    return SimilarPatientService(
        patient_repo=PatientRepository(session),
        medical_record_repo=MedicalRecordRepository(session),
        neo4j_db=neo4j_db,
    )