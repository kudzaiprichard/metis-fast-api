from datetime import date
from typing import Sequence, Tuple
from uuid import UUID

from src.modules.patients.domain.models.patient import Patient
from src.modules.patients.domain.models.medical_record import MedicalRecord
from src.modules.patients.domain.repositories.patient_repository import PatientRepository
from src.modules.patients.domain.repositories.medical_record_repository import MedicalRecordRepository
from src.shared.exceptions import NotFoundException, ConflictException
from src.shared.responses import ErrorDetail


class PatientService:
    def __init__(
        self,
        patient_repo: PatientRepository,
        medical_record_repo: MedicalRecordRepository,
    ):
        self.patient_repo = patient_repo
        self.record_repo = medical_record_repo

    # ── Patient CRUD ──

    async def create_patient(
        self,
        first_name: str,
        last_name: str,
        date_of_birth: date,
        gender: str,
        email: str | None = None,
        phone: str | None = None,
        address: str | None = None,
    ) -> Patient:
        if email and await self.patient_repo.email_exists(email):
            error = ErrorDetail.builder("Creation Failed", "EMAIL_EXISTS", 409)
            error.add_field_error("email", "A patient with this email already exists")
            raise ConflictException(
                message="A patient with this email already exists",
                error_detail=error.build(),
            )

        patient = Patient(
            first_name=first_name,
            last_name=last_name,
            date_of_birth=date_of_birth,
            gender=gender,
            email=email,
            phone=phone,
            address=address,
        )

        return await self.patient_repo.create(patient)

    async def get_patient(self, patient_id: UUID) -> Patient:
        patient = await self.patient_repo.get_by_id(patient_id)
        if not patient:
            raise NotFoundException(
                message="Patient not found",
                error_detail=ErrorDetail(
                    title="Not Found",
                    code="PATIENT_NOT_FOUND",
                    status=404,
                    details=[f"No patient found with id {patient_id}"],
                ),
            )
        return patient

    async def get_patients(
        self,
        page: int = 1,
        page_size: int = 20,
    ) -> Tuple[Sequence[Patient], int]:
        return await self.patient_repo.paginate(
            page=page,
            page_size=page_size,
            order_by="created_at",
            descending=True,
        )

    async def update_patient(
        self,
        patient_id: UUID,
        first_name: str | None = None,
        last_name: str | None = None,
        date_of_birth: date | None = None,
        gender: str | None = None,
        email: str | None = None,
        phone: str | None = None,
        address: str | None = None,
    ) -> Patient:
        patient = await self.get_patient(patient_id)
        data = {}

        if first_name is not None:
            data["first_name"] = first_name
        if last_name is not None:
            data["last_name"] = last_name
        if date_of_birth is not None:
            data["date_of_birth"] = date_of_birth
        if gender is not None:
            data["gender"] = gender
        if phone is not None:
            data["phone"] = phone
        if address is not None:
            data["address"] = address

        if email is not None and email != patient.email:
            if await self.patient_repo.email_exists(email):
                error = ErrorDetail.builder("Update Failed", "EMAIL_EXISTS", 409)
                error.add_field_error("email", "A patient with this email already exists")
                raise ConflictException(
                    message="A patient with this email already exists",
                    error_detail=error.build(),
                )
            data["email"] = email

        if not data:
            return patient

        return await self.patient_repo.update(patient, data)

    async def delete_patient(self, patient_id: UUID) -> None:
        patient = await self.get_patient(patient_id)
        await self.patient_repo.delete(patient)

    # ── Medical Records ──

    async def add_medical_record(
        self,
        patient_id: UUID,
        age: int,
        bmi: float,
        hba1c_baseline: float,
        egfr: float,
        diabetes_duration: float,
        fasting_glucose: float,
        c_peptide: float,
        cvd: int,
        ckd: int,
        nafld: int,
        hypertension: int,
        bp_systolic: float,
        ldl: float,
        hdl: float,
        triglycerides: float,
        alt: float,
        notes: str | None = None,
    ) -> MedicalRecord:
        # Verify patient exists
        await self.get_patient(patient_id)

        record = MedicalRecord(
            patient_id=patient_id,
            age=age,
            bmi=bmi,
            hba1c_baseline=hba1c_baseline,
            egfr=egfr,
            diabetes_duration=diabetes_duration,
            fasting_glucose=fasting_glucose,
            c_peptide=c_peptide,
            cvd=cvd,
            ckd=ckd,
            nafld=nafld,
            hypertension=hypertension,
            bp_systolic=bp_systolic,
            ldl=ldl,
            hdl=hdl,
            triglycerides=triglycerides,
            alt=alt,
            notes=notes,
        )

        return await self.record_repo.create(record)

    async def get_medical_records(
        self, patient_id: UUID, skip: int = 0, limit: int = 50
    ) -> Sequence[MedicalRecord]:
        await self.get_patient(patient_id)
        return await self.record_repo.get_by_patient(patient_id, skip, limit)

    async def get_medical_record(
        self, patient_id: UUID, record_id: UUID
    ) -> MedicalRecord:
        await self.get_patient(patient_id)

        record = await self.record_repo.get_by_id(record_id)
        if not record or record.patient_id != patient_id:
            raise NotFoundException(
                message="Medical record not found",
                error_detail=ErrorDetail(
                    title="Not Found",
                    code="RECORD_NOT_FOUND",
                    status=404,
                    details=[f"No medical record found with id {record_id}"],
                ),
            )
        return record

    async def get_latest_medical_record(self, patient_id: UUID) -> MedicalRecord:
        await self.get_patient(patient_id)

        record = await self.record_repo.get_latest_for_patient(patient_id)
        if not record:
            raise NotFoundException(
                message="No medical records found for this patient",
                error_detail=ErrorDetail(
                    title="Not Found",
                    code="NO_RECORDS",
                    status=404,
                    details=["This patient has no medical records"],
                ),
            )
        return record