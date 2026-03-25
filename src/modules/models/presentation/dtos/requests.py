from pydantic import BaseModel, Field

from src.shared.inference import PatientInput


class BatchPredictRequest(BaseModel):
    patients: list[PatientInput] = Field(min_length=1, max_length=50)
