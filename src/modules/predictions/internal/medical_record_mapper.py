"""
Maps a MedicalRecord SQLAlchemy object to the context dict
that the inference engine expects.

This is the single bridge between the database schema and the ML model.
All 16 features are extracted by name — order doesn't matter.
"""

from src.modules.patients.domain.models.medical_record import MedicalRecord


def to_context(record: MedicalRecord) -> dict:
    """
    Convert a MedicalRecord to the 16-feature context dict.

    Returns:
        dict matching the keys expected by FeaturePipeline.transform_single()
    """
    return {
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
    }