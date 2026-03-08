"""
Treatment constants extracted from data_generator.py.

These are the only values the inference pipeline needs from the
ML project's data layer. Keeping them here avoids importing the
full data_generator (which pulls in pandas, numpy generators, etc).
"""

TREATMENTS = ["Metformin", "GLP-1", "SGLT-2", "DPP-4", "Insulin"]
N_TREATMENTS = len(TREATMENTS)
TREATMENT_TO_IDX = {t: i for i, t in enumerate(TREATMENTS)}
IDX_TO_TREATMENT = {i: t for i, t in enumerate(TREATMENTS)}

CONTEXT_FEATURES = [
    "age", "bmi", "hba1c_baseline", "egfr", "diabetes_duration",
    "fasting_glucose", "c_peptide", "cvd", "ckd", "nafld",
    "hypertension", "bp_systolic", "ldl", "hdl", "triglycerides", "alt",
]

CONTINUOUS_FEATURES = [
    "age", "bmi", "hba1c_baseline", "egfr", "diabetes_duration",
    "fasting_glucose", "c_peptide", "bp_systolic", "ldl", "hdl",
    "triglycerides", "alt",
]

BINARY_FEATURES = ["cvd", "ckd", "nafld", "hypertension"]

INTERACTION_FEATURES = [
    "age_x_ckd", "bmi_x_nafld", "cvd_x_egfr", "cpeptide_x_hba1c",
    "duration_x_hba1c", "fg_x_bmi", "tg_hdl_ratio", "renal_risk",
    "severity_score",
]

ALL_FEATURES = CONTINUOUS_FEATURES + BINARY_FEATURES + INTERACTION_FEATURES