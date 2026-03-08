from uuid import UUID
from typing import Optional

from pydantic import BaseModel, Field


class CreatePredictionRequest(BaseModel):
    medical_record_id: UUID
    patient_id: UUID


class DoctorDecisionRequest(BaseModel):
    decision: str = Field(pattern="^(ACCEPTED|OVERRIDDEN)$")
    final_treatment: Optional[str] = Field(None, pattern="^(Metformin|GLP-1|SGLT-2|DPP-4|Insulin)$")
    doctor_notes: Optional[str] = Field(None, max_length=1000)