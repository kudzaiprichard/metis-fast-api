from datetime import date
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field, model_validator


# ── Patient DTOs ──

class CreatePatientRequest(BaseModel):
    first_name: str = Field(min_length=1, max_length=100)
    last_name: str = Field(min_length=1, max_length=100)
    date_of_birth: date
    gender: str = Field(pattern="^(male|female|other)$")
    email: Optional[EmailStr] = None
    phone: Optional[str] = Field(None, max_length=20)
    address: Optional[str] = Field(None, max_length=500)


class UpdatePatientRequest(BaseModel):
    first_name: Optional[str] = Field(None, min_length=1, max_length=100)
    last_name: Optional[str] = Field(None, min_length=1, max_length=100)
    date_of_birth: Optional[date] = None
    gender: Optional[str] = Field(None, pattern="^(male|female|other)$")
    email: Optional[EmailStr] = None
    phone: Optional[str] = Field(None, max_length=20)
    address: Optional[str] = Field(None, max_length=500)


# ── Medical Record DTOs ──

class CreateMedicalRecordRequest(BaseModel):
    age: int = Field(ge=18, le=120)
    bmi: float = Field(ge=10, le=80)
    hba1c_baseline: float = Field(ge=3, le=20)
    egfr: float = Field(ge=5, le=200)
    diabetes_duration: float = Field(ge=0, le=60)
    fasting_glucose: float = Field(ge=50, le=500)
    c_peptide: float = Field(ge=0, le=10)
    cvd: int = Field(ge=0, le=1)
    ckd: int = Field(ge=0, le=1)
    nafld: int = Field(ge=0, le=1)
    hypertension: int = Field(ge=0, le=1)
    bp_systolic: float = Field(ge=60, le=250)
    ldl: float = Field(ge=20, le=400)
    hdl: float = Field(ge=10, le=150)
    triglycerides: float = Field(ge=30, le=800)
    alt: float = Field(ge=5, le=500)
    notes: Optional[str] = Field(None, max_length=1000)


# ── Similar Patient DTOs ──

class FindSimilarPatientsRequest(BaseModel):
    """Find similar patients in tabular format with pagination."""
    patient_id: Optional[UUID] = Field(None, description="Uses latest medical record")
    medical_record_id: Optional[UUID] = Field(None, description="Uses specific medical record")
    limit: int = Field(default=20, ge=1, le=100, description="Max total matches from Neo4j")
    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=10, ge=1, le=50, description="Results per page")
    treatment_filter: Optional[str] = None
    min_similarity: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def require_at_least_one_id(self):
        if not self.patient_id and not self.medical_record_id:
            raise ValueError("Either 'patient_id' or 'medical_record_id' must be provided")
        return self


class FindSimilarPatientsGraphRequest(BaseModel):
    """Find similar patients in graph format for visualization."""
    patient_id: Optional[UUID] = Field(None, description="Uses latest medical record")
    medical_record_id: Optional[UUID] = Field(None, description="Uses specific medical record")
    limit: int = Field(default=5, ge=1, le=20)
    treatment_filter: Optional[str] = None

    @model_validator(mode="after")
    def require_at_least_one_id(self):
        if not self.patient_id and not self.medical_record_id:
            raise ValueError("Either 'patient_id' or 'medical_record_id' must be provided")
        return self