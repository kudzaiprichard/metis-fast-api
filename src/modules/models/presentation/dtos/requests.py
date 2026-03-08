from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
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

    def to_context(self) -> dict:
        """Convert to the dict format the inference engine expects."""
        return self.model_dump()


class ExplainRequest(BaseModel):
    """Accepts a full prediction payload to generate LLM explanation."""
    patient: dict
    decision: dict
    safety: dict
    fairness: dict


class BatchPredictRequest(BaseModel):
    patients: list[PredictRequest] = Field(min_length=1, max_length=50)