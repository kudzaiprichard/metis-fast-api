from uuid import UUID
from datetime import date, datetime
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from src.modules.patients.domain.models.patient import Patient
from src.modules.patients.domain.models.medical_record import MedicalRecord


# ── Medical Record Response ──

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


# ── Patient Responses ──

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


# ── Similar Patient Responses ──

class SimilarPatientProfileResponse(BaseModel):
    age: int
    gender: str
    ethnicity: str
    hba1c_baseline: float = Field(alias="hba1cBaseline")
    c_peptide: float = Field(alias="cPeptide")
    bmi: float
    egfr: float
    diabetes_duration: float = Field(alias="diabetesDuration")
    bp_systolic: int = Field(alias="bpSystolic")
    fasting_glucose: float = Field(alias="fastingGlucose")

    class Config:
        populate_by_name = True

    @staticmethod
    def from_dict(data: dict) -> "SimilarPatientProfileResponse":
        return SimilarPatientProfileResponse(
            age=data["age"],
            gender=data["gender"],
            ethnicity=data["ethnicity"],
            hba1cBaseline=data["hba1c_baseline"],
            cPeptide=data["c_peptide"],
            bmi=data["bmi"],
            egfr=data["egfr"],
            diabetesDuration=data["diabetes_duration"],
            bpSystolic=data["bp_systolic"],
            fastingGlucose=data["fasting_glucose"],
        )


class SimilarPatientOutcomeResponse(BaseModel):
    hba1c_reduction: float = Field(alias="hba1cReduction")
    hba1c_followup: float = Field(alias="hba1cFollowup")
    time_to_target: str = Field(alias="timeToTarget")
    adverse_events: str = Field(alias="adverseEvents")
    outcome_category: str = Field(alias="outcomeCategory")
    success: bool

    class Config:
        populate_by_name = True

    @staticmethod
    def from_dict(data: dict) -> "SimilarPatientOutcomeResponse":
        return SimilarPatientOutcomeResponse(
            hba1cReduction=data["hba1c_reduction"],
            hba1cFollowup=data["hba1c_followup"],
            timeToTarget=data["time_to_target"],
            adverseEvents=data["adverse_events"],
            outcomeCategory=data["outcome_category"],
            success=data["success"],
        )


class SimilarPatientCaseResponse(BaseModel):
    case_id: str = Field(alias="caseId")
    similarity_score: float = Field(alias="similarityScore")
    clinical_similarity: float = Field(alias="clinicalSimilarity")
    comorbidity_similarity: float = Field(alias="comorbiditySimilarity")
    profile: SimilarPatientProfileResponse
    comorbidities: List[str]
    treatment_given: str = Field(alias="treatmentGiven")
    drug_class: str = Field(alias="drugClass")
    outcome: SimilarPatientOutcomeResponse

    class Config:
        populate_by_name = True

    @staticmethod
    def from_dict(data: dict) -> "SimilarPatientCaseResponse":
        return SimilarPatientCaseResponse(
            caseId=data["case_id"],
            similarityScore=data["similarity_score"],
            clinicalSimilarity=data["clinical_similarity"],
            comorbiditySimilarity=data["comorbidity_similarity"],
            profile=SimilarPatientProfileResponse.from_dict(data["profile"]),
            comorbidities=data["comorbidities"],
            treatmentGiven=data["treatment_given"],
            drugClass=data["drug_class"],
            outcome=SimilarPatientOutcomeResponse.from_dict(data["outcome"]),
        )


class SimilarPatientsResponse(BaseModel):
    patient_id: str = Field(alias="patientId")
    similar_cases: List[SimilarPatientCaseResponse] = Field(alias="similarCases")
    total_found: int = Field(alias="totalFound")
    filters_applied: Dict[str, Any] = Field(alias="filtersApplied")

    class Config:
        populate_by_name = True


# ── Graph Responses ──

class GraphNodeStyleResponse(BaseModel):
    color: str
    size: str
    shape: str

    @staticmethod
    def from_dict(data: dict) -> "GraphNodeStyleResponse":
        return GraphNodeStyleResponse(**data)


class GraphEdgeStyleResponse(BaseModel):
    width: int
    color: str

    @staticmethod
    def from_dict(data: dict) -> "GraphEdgeStyleResponse":
        return GraphEdgeStyleResponse(**data)


class GraphNodeResponse(BaseModel):
    id: str
    type: str
    label: str
    data: Dict[str, Any]
    style: GraphNodeStyleResponse

    @staticmethod
    def from_dict(data: dict) -> "GraphNodeResponse":
        return GraphNodeResponse(
            id=data["id"],
            type=data["type"],
            label=data["label"],
            data=data["data"],
            style=GraphNodeStyleResponse.from_dict(data["style"]),
        )


class GraphEdgeResponse(BaseModel):
    id: str
    source: str
    target: str
    type: str
    label: str
    data: Dict[str, Any]
    style: GraphEdgeStyleResponse

    @staticmethod
    def from_dict(data: dict) -> "GraphEdgeResponse":
        return GraphEdgeResponse(
            id=data["id"],
            source=data["source"],
            target=data["target"],
            type=data["type"],
            label=data["label"],
            data=data["data"],
            style=GraphEdgeStyleResponse.from_dict(data["style"]),
        )


class GraphMetadataResponse(BaseModel):
    query_patient: Dict[str, Any] = Field(alias="queryPatient")
    filters_applied: Dict[str, Any] = Field(alias="filtersApplied")
    results_found: int = Field(alias="resultsFound")
    similarity_range: Optional[Dict[str, Any]] = Field(None, alias="similarityRange")

    class Config:
        populate_by_name = True

    @staticmethod
    def from_dict(data: dict) -> "GraphMetadataResponse":
        return GraphMetadataResponse(
            queryPatient=data["query_patient"],
            filtersApplied=data["filters_applied"],
            resultsFound=data["results_found"],
            similarityRange=data.get("similarity_range"),
        )


class SimilarPatientsGraphResponse(BaseModel):
    patient_id: str = Field(alias="patientId")
    nodes: List[GraphNodeResponse]
    edges: List[GraphEdgeResponse]
    metadata: GraphMetadataResponse

    class Config:
        populate_by_name = True

    @staticmethod
    def from_dict(data: dict) -> "SimilarPatientsGraphResponse":
        return SimilarPatientsGraphResponse(
            patientId=data["patient_id"],
            nodes=[GraphNodeResponse.from_dict(n) for n in data["nodes"]],
            edges=[GraphEdgeResponse.from_dict(e) for e in data["edges"]],
            metadata=GraphMetadataResponse.from_dict(data["metadata"]),
        )


# ── Similar Patient Detail Responses ──

class SimilarPatientDemographicsResponse(BaseModel):
    age: int
    gender: str
    ethnicity: str
    age_group: str = Field(alias="ageGroup")

    class Config:
        populate_by_name = True

    @staticmethod
    def from_dict(data: dict) -> "SimilarPatientDemographicsResponse":
        return SimilarPatientDemographicsResponse(
            age=data["age"],
            gender=data["gender"],
            ethnicity=data["ethnicity"],
            ageGroup=data["age_group"],
        )


class SimilarPatientClinicalFeaturesResponse(BaseModel):
    hba1c_baseline: float = Field(alias="hba1cBaseline")
    diabetes_duration: float = Field(alias="diabetesDuration")
    fasting_glucose: float = Field(alias="fastingGlucose")
    c_peptide: float = Field(alias="cPeptide")
    egfr: float
    bmi: float
    bp_systolic: int = Field(alias="bpSystolic")
    bp_diastolic: int = Field(alias="bpDiastolic")
    alt: float
    ldl: float
    hdl: float
    triglycerides: float
    previous_prediabetes: bool = Field(alias="previousPrediabetes")

    class Config:
        populate_by_name = True

    @staticmethod
    def from_dict(data: dict) -> "SimilarPatientClinicalFeaturesResponse":
        return SimilarPatientClinicalFeaturesResponse(
            hba1cBaseline=data["hba1c_baseline"],
            diabetesDuration=data["diabetes_duration"],
            fastingGlucose=data["fasting_glucose"],
            cPeptide=data["c_peptide"],
            egfr=data["egfr"],
            bmi=data["bmi"],
            bpSystolic=data["bp_systolic"],
            bpDiastolic=data["bp_diastolic"],
            alt=data["alt"],
            ldl=data["ldl"],
            hdl=data["hdl"],
            triglycerides=data["triglycerides"],
            previousPrediabetes=data["previous_prediabetes"],
        )


class SimilarPatientClinicalCategoriesResponse(BaseModel):
    bmi_category: str = Field(alias="bmiCategory")
    hba1c_severity: str = Field(alias="hba1cSeverity")
    kidney_function: str = Field(alias="kidneyFunction")

    class Config:
        populate_by_name = True

    @staticmethod
    def from_dict(data: dict) -> "SimilarPatientClinicalCategoriesResponse":
        return SimilarPatientClinicalCategoriesResponse(
            bmiCategory=data["bmi_category"],
            hba1cSeverity=data["hba1c_severity"],
            kidneyFunction=data["kidney_function"],
        )


class SimilarPatientTreatmentResponse(BaseModel):
    drug_name: str = Field(alias="drugName")
    drug_class: str = Field(alias="drugClass")
    cost_category: str = Field(alias="costCategory")
    evidence_level: str = Field(alias="evidenceLevel")

    class Config:
        populate_by_name = True

    @staticmethod
    def from_dict(data: dict) -> "SimilarPatientTreatmentResponse":
        return SimilarPatientTreatmentResponse(
            drugName=data["drug_name"],
            drugClass=data["drug_class"],
            costCategory=data["cost_category"],
            evidenceLevel=data["evidence_level"],
        )


class SimilarPatientDetailResponse(BaseModel):
    patient_id: str = Field(alias="patientId")
    demographics: SimilarPatientDemographicsResponse
    clinical_features: SimilarPatientClinicalFeaturesResponse = Field(alias="clinicalFeatures")
    clinical_categories: SimilarPatientClinicalCategoriesResponse = Field(alias="clinicalCategories")
    comorbidities: List[str]
    treatment: Optional[SimilarPatientTreatmentResponse] = None
    outcome: Optional[SimilarPatientOutcomeResponse] = None

    class Config:
        populate_by_name = True

    @staticmethod
    def from_dict(data: dict) -> "SimilarPatientDetailResponse":
        return SimilarPatientDetailResponse(
            patientId=data["patient_id"],
            demographics=SimilarPatientDemographicsResponse.from_dict(data["demographics"]),
            clinicalFeatures=SimilarPatientClinicalFeaturesResponse.from_dict(data["clinical_features"]),
            clinicalCategories=SimilarPatientClinicalCategoriesResponse.from_dict(data["clinical_categories"]),
            comorbidities=data["comorbidities"],
            treatment=SimilarPatientTreatmentResponse.from_dict(data["treatment"]) if data.get("treatment") else None,
            outcome=SimilarPatientOutcomeResponse.from_dict(data["outcome"]) if data.get("outcome") else None,
        )