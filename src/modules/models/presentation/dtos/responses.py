from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PatientContextResponse(BaseModel):
    age: int
    bmi: float
    hba1c_baseline: float = Field(alias="hba1cBaseline")
    egfr: float
    diabetes_duration: float = Field(alias="diabetesDuration")
    fasting_glucose: float = Field(alias="fastingGlucose")
    c_peptide: float = Field(alias="cPeptide")
    bp_systolic: float = Field(alias="bpSystolic")
    ldl: float
    hdl: float
    triglycerides: float
    cvd: str
    ckd: str
    nafld: str
    hypertension: str

    class Config:
        populate_by_name = True


class DecisionResponse(BaseModel):
    recommended_treatment: str = Field(alias="recommendedTreatment")
    recommended_idx: int = Field(alias="recommendedIdx")
    confidence_pct: int = Field(alias="confidencePct")
    confidence_label: str = Field(alias="confidenceLabel")
    win_rates: Dict[str, float] = Field(alias="winRates")
    posterior_means: Dict[str, float] = Field(alias="posteriorMeans")
    runner_up: str = Field(alias="runnerUp")
    runner_up_win_rate: float = Field(alias="runnerUpWinRate")
    mean_gap: float = Field(alias="meanGap")
    n_draws: int = Field(alias="nDraws")

    class Config:
        populate_by_name = True


class SafetyResponse(BaseModel):
    status: str
    recommended_contraindications: List[str] = Field(alias="recommendedContraindications")
    recommended_warnings: List[str] = Field(alias="recommendedWarnings")
    excluded_treatments: Dict[str, List[str]] = Field(alias="excludedTreatments")
    all_warnings: Dict[str, List[str]] = Field(alias="allWarnings")

    class Config:
        populate_by_name = True


class FairnessResponse(BaseModel):
    decision_features: List[str] = Field(alias="decisionFeatures")
    dual_use_features: Dict[str, str] = Field(alias="dualUseFeatures")
    excluded_protected_features: List[str] = Field(alias="excludedProtectedFeatures")
    statement: str

    class Config:
        populate_by_name = True


class PredictionResponse(BaseModel):
    patient: PatientContextResponse
    decision: DecisionResponse
    safety: SafetyResponse
    fairness: FairnessResponse

    @staticmethod
    def from_payload(payload: dict) -> "PredictionResponse":
        return PredictionResponse(
            patient=PatientContextResponse(
                age=payload["patient"]["age"],
                bmi=payload["patient"]["bmi"],
                hba1cBaseline=payload["patient"]["hba1c_baseline"],
                egfr=payload["patient"]["egfr"],
                diabetesDuration=payload["patient"]["diabetes_duration"],
                fastingGlucose=payload["patient"]["fasting_glucose"],
                cPeptide=payload["patient"]["c_peptide"],
                bpSystolic=payload["patient"]["bp_systolic"],
                ldl=payload["patient"]["ldl"],
                hdl=payload["patient"]["hdl"],
                triglycerides=payload["patient"]["triglycerides"],
                cvd=payload["patient"]["cvd"],
                ckd=payload["patient"]["ckd"],
                nafld=payload["patient"]["nafld"],
                hypertension=payload["patient"]["hypertension"],
            ),
            decision=DecisionResponse(
                recommendedTreatment=payload["decision"]["recommended_treatment"],
                recommendedIdx=payload["decision"]["recommended_idx"],
                confidencePct=payload["decision"]["confidence_pct"],
                confidenceLabel=payload["decision"]["confidence_label"],
                winRates=payload["decision"]["win_rates"],
                posteriorMeans=payload["decision"]["posterior_means"],
                runnerUp=payload["decision"]["runner_up"],
                runnerUpWinRate=payload["decision"]["runner_up_win_rate"],
                meanGap=payload["decision"]["mean_gap"],
                nDraws=payload["decision"]["n_draws"],
            ),
            safety=SafetyResponse(
                status=payload["safety"]["status"],
                recommendedContraindications=payload["safety"]["recommended_contraindications"],
                recommendedWarnings=payload["safety"]["recommended_warnings"],
                excludedTreatments=payload["safety"]["excluded_treatments"],
                allWarnings=payload["safety"]["all_warnings"],
            ),
            fairness=FairnessResponse(
                decisionFeatures=payload["fairness"]["decision_features"],
                dualUseFeatures=payload["fairness"]["dual_use_features"],
                excludedProtectedFeatures=payload["fairness"]["excluded_protected_features"],
                statement=payload["fairness"]["statement"],
            ),
        )


class ExplanationResponse(BaseModel):
    recommendation_summary: str = Field(alias="recommendationSummary")
    runner_up_analysis: str = Field(alias="runnerUpAnalysis")
    confidence_statement: str = Field(alias="confidenceStatement")
    safety_assessment: str = Field(alias="safetyAssessment")
    monitoring_note: str = Field(alias="monitoringNote")
    disclaimer: str

    class Config:
        populate_by_name = True

    @staticmethod
    def from_payload(payload: dict) -> "ExplanationResponse":
        return ExplanationResponse(
            recommendationSummary=payload["recommendation_summary"],
            runnerUpAnalysis=payload["runner_up_analysis"],
            confidenceStatement=payload["confidence_statement"],
            safetyAssessment=payload["safety_assessment"],
            monitoringNote=payload["monitoring_note"],
            disclaimer=payload["disclaimer"],
        )


class PredictionWithExplanationResponse(BaseModel):
    patient: PatientContextResponse
    decision: DecisionResponse
    safety: SafetyResponse
    fairness: FairnessResponse
    explanation: ExplanationResponse

    @staticmethod
    def from_payload(payload: dict) -> "PredictionWithExplanationResponse":
        base = PredictionResponse.from_payload(payload)
        return PredictionWithExplanationResponse(
            patient=base.patient,
            decision=base.decision,
            safety=base.safety,
            fairness=base.fairness,
            explanation=ExplanationResponse.from_payload(payload["explanation"]),
        )