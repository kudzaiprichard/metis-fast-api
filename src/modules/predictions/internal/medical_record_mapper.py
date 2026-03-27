"""
Maps a MedicalRecord SQLAlchemy object to a validated PatientInput — the
schema that the inference engine expects at its boundary.

This is the single bridge between the database schema and the ML model.
All 16 clinical features flow through; the four G-14 safety flags default
to 0 because MedicalRecord does not yet carry them, and the audit fields
(gender, ethnicity, patient_id) are deliberately omitted per G-15
fairness posture.
"""

from src.shared.inference import PatientInput
from src.modules.patients.domain.models.medical_record import MedicalRecord


def to_context(record: MedicalRecord) -> PatientInput:
    return PatientInput.model_validate({
        "age": record.age,
        "bmi": record.bmi,
        "hba1c_baseline": record.hba1c_baseline,
        "egfr": record.egfr,
        "diabetes_duration": record.diabetes_duration,
        "fasting_glucose": record.fasting_glucose,
        "c_peptide": record.c_peptide,
        "cvd": record.cvd,
        "ckd": record.ckd,
        "nafld": record.nafld,
        "hypertension": record.hypertension,
        "bp_systolic": record.bp_systolic,
        "ldl": record.ldl,
        "hdl": record.hdl,
        "triglycerides": record.triglycerides,
        "alt": record.alt,
    })
