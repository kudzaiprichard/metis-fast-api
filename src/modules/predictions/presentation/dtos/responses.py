from uuid import UUID
from datetime import datetime
from typing import Optional, Dict, List

from pydantic import BaseModel, Field

from src.modules.predictions.domain.models.prediction import Prediction


class SafetyDetailResponse(BaseModel):
    recommended_contraindications: List[str] = Field(alias="recommendedContraindications")
    recommended_warnings: List[str] = Field(alias="recommendedWarnings")
    excluded_treatments: Dict[str, List[str]] = Field(alias="excludedTreatments")
    all_warnings: Dict[str, List[str]] = Field(alias="allWarnings")

    class Config:
        populate_by_name = True


class ExplanationResponse(BaseModel):
    summary: str
    runner_up: str = Field(alias="runnerUp")
    confidence: str
    safety: str
    monitoring: str
    disclaimer: str

    class Config:
        populate_by_name = True


class PredictionResponse(BaseModel):
    id: UUID
    medical_record_id: UUID = Field(alias="medicalRecordId")
    patient_id: UUID = Field(alias="patientId")
    created_by: UUID = Field(alias="createdBy")

    # Model recommendation
    recommended_treatment: str = Field(alias="recommendedTreatment")
    recommended_idx: int = Field(alias="recommendedIdx")
    confidence_pct: int = Field(alias="confidencePct")
    confidence_label: str = Field(alias="confidenceLabel")
    mean_gap: float = Field(alias="meanGap")
    runner_up: str = Field(alias="runnerUp")
    runner_up_win_rate: float = Field(alias="runnerUpWinRate")
    win_rates: Dict[str, float] = Field(alias="winRates")
    posterior_means: Dict[str, float] = Field(alias="posteriorMeans")

    # Safety
    safety_status: str = Field(alias="safetyStatus")
    safety_details: SafetyDetailResponse = Field(alias="safetyDetails")

    # Fairness
    fairness: dict

    # Explanation
    explanation: ExplanationResponse

    # Doctor decision
    doctor_decision: str = Field(alias="doctorDecision")
    final_treatment: Optional[str] = Field(None, alias="finalTreatment")
    doctor_notes: Optional[str] = Field(None, alias="doctorNotes")

    # Timestamps
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")

    class Config:
        populate_by_name = True
        from_attributes = True

    @staticmethod
    def from_prediction(prediction: Prediction) -> "PredictionResponse":
        return PredictionResponse(
            id=prediction.id,
            medicalRecordId=prediction.medical_record_id,
            patientId=prediction.patient_id,
            createdBy=prediction.created_by,
            recommendedTreatment=prediction.recommended_treatment,
            recommendedIdx=prediction.recommended_idx,
            confidencePct=prediction.confidence_pct,
            confidenceLabel=prediction.confidence_label,
            meanGap=prediction.mean_gap,
            runnerUp=prediction.runner_up,
            runnerUpWinRate=prediction.runner_up_win_rate,
            winRates=prediction.win_rates,
            posteriorMeans=prediction.posterior_means,
            safetyStatus=prediction.safety_status,
            safetyDetails=SafetyDetailResponse(
                recommendedContraindications=prediction.safety_details["recommended_contraindications"],
                recommendedWarnings=prediction.safety_details["recommended_warnings"],
                excludedTreatments=prediction.safety_details["excluded_treatments"],
                allWarnings=prediction.safety_details["all_warnings"],
            ),
            fairness=prediction.fairness,
            explanation=ExplanationResponse(
                summary=prediction.explanation_summary,
                runnerUp=prediction.explanation_runner_up,
                confidence=prediction.explanation_confidence,
                safety=prediction.explanation_safety,
                monitoring=prediction.explanation_monitoring,
                disclaimer=prediction.explanation_disclaimer,
            ),
            doctorDecision=prediction.doctor_decision.value,
            finalTreatment=prediction.final_treatment,
            doctorNotes=prediction.doctor_notes,
            createdAt=prediction.created_at,
            updatedAt=prediction.updated_at,
        )