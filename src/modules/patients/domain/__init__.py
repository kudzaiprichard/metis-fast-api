# ── domain/__init__.py ──
from src.modules.patients.domain.models.patient import Patient
from src.modules.patients.domain.models.medical_record import MedicalRecord
from src.modules.patients.domain.repositories.patient_repository import PatientRepository
from src.modules.patients.domain.repositories.medical_record_repository import MedicalRecordRepository
from src.modules.patients.domain.services.patient_service import PatientService

__all__ = ["Patient", "MedicalRecord", "PatientRepository", "MedicalRecordRepository", "PatientService"]


# ── domain/models/__init__.py ──
from src.modules.patients.domain.models.patient import Patient
from src.modules.patients.domain.models.medical_record import MedicalRecord

__all__ = ["Patient", "MedicalRecord"]


# ── domain/repositories/__init__.py ──
from src.modules.patients.domain.repositories.patient_repository import PatientRepository
from src.modules.patients.domain.repositories.medical_record_repository import MedicalRecordRepository

__all__ = ["PatientRepository", "MedicalRecordRepository"]


# ── domain/services/__init__.py ──
from src.modules.patients.domain.services.patient_service import PatientService

__all__ = ["PatientService"]