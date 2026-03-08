from uuid import UUID
from datetime import date, datetime
from typing import Optional, List

from pydantic import BaseModel, Field

from src.modules.patients.domain.models.patient import Patient
from src.modules.patients.domain.models.medical_record import MedicalRecord


class MedicalRecordResponse(BaseModel):
    id: UUID
    patient_id: UUID = Field(alias="patientId")
    age: int
    bmi: float
    hba1c_baseline: float = Field(alias="hba1cBaseline")
    egfr: float
    diabetes_duration: float = Field(alias="diabetesDuration")
    fasting_glucose: float = Field(alias="fastingGlucose")
    c_peptide: float = Field(alias="cPeptide")
    cvd: int
    ckd: int
    nafld: int
    hypertension: int
    bp_systolic: float = Field(alias="bpSystolic")
    ldl: float
    hdl: float
    triglycerides: float
    alt: float
    notes: Optional[str] = None
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")

    class Config:
        populate_by_name = True
        from_attributes = True

    @staticmethod
    def from_record(record: MedicalRecord) -> "MedicalRecordResponse":
        return MedicalRecordResponse(
            id=record.id,
            patientId=record.patient_id,
            age=record.age,
            bmi=record.bmi,
            hba1cBaseline=record.hba1c_baseline,
            egfr=record.egfr,
            diabetesDuration=record.diabetes_duration,
            fastingGlucose=record.fasting_glucose,
            cPeptide=record.c_peptide,
            cvd=record.cvd,
            ckd=record.ckd,
            nafld=record.nafld,
            hypertension=record.hypertension,
            bpSystolic=record.bp_systolic,
            ldl=record.ldl,
            hdl=record.hdl,
            triglycerides=record.triglycerides,
            alt=record.alt,
            notes=record.notes,
            createdAt=record.created_at,
            updatedAt=record.updated_at,
        )


class PatientResponse(BaseModel):
    id: UUID
    first_name: str = Field(alias="firstName")
    last_name: str = Field(alias="lastName")
    date_of_birth: date = Field(alias="dateOfBirth")
    gender: str
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")

    class Config:
        populate_by_name = True
        from_attributes = True

    @staticmethod
    def from_patient(patient: Patient) -> "PatientResponse":
        return PatientResponse(
            id=patient.id,
            firstName=patient.first_name,
            lastName=patient.last_name,
            dateOfBirth=patient.date_of_birth,
            gender=patient.gender,
            email=patient.email,
            phone=patient.phone,
            address=patient.address,
            createdAt=patient.created_at,
            updatedAt=patient.updated_at,
        )


class PatientDetailResponse(BaseModel):
    id: UUID
    first_name: str = Field(alias="firstName")
    last_name: str = Field(alias="lastName")
    date_of_birth: date = Field(alias="dateOfBirth")
    gender: str
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    medical_records: List[MedicalRecordResponse] = Field(alias="medicalRecords")
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")

    class Config:
        populate_by_name = True
        from_attributes = True

    @staticmethod
    def from_patient(patient: Patient) -> "PatientDetailResponse":
        return PatientDetailResponse(
            id=patient.id,
            firstName=patient.first_name,
            lastName=patient.last_name,
            dateOfBirth=patient.date_of_birth,
            gender=patient.gender,
            email=patient.email,
            phone=patient.phone,
            address=patient.address,
            medicalRecords=[
                MedicalRecordResponse.from_record(r) for r in patient.medical_records
            ],
            createdAt=patient.created_at,
            updatedAt=patient.updated_at,
        )