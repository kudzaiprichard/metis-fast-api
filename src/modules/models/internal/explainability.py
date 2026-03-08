"""
Explainability Extraction for NeuralThompson
Copied from ML project — import paths updated for FastAPI integration.

Only change: src.data_generator → src.modules.models.internal.constants
"""

import numpy as np
from typing import Dict, List
import logging
logger = logging.getLogger(__name__)

from src.modules.models.internal.constants import TREATMENTS, N_TREATMENTS, IDX_TO_TREATMENT


# ─── Safety Rules ───

def check_contraindications(context: Dict, treatment: str) -> List[str]:
    contraindications = []
    egfr = context.get("egfr", 999)
    if treatment == "Metformin":
        if egfr < 30:
            contraindications.append(f"Metformin contraindicated: eGFR {egfr:.0f} < 30 mL/min/1.73m² (risk of lactic acidosis)")
    elif treatment == "SGLT-2":
        if egfr < 25:
            contraindications.append(f"SGLT-2 contraindicated: eGFR {egfr:.0f} < 25 mL/min/1.73m² (insufficient renal function for efficacy)")
    return contraindications


def check_warnings(context: Dict, treatment: str) -> List[str]:
    warnings = []
    age = context.get("age", 0)
    bmi = context.get("bmi", 0)
    egfr = context.get("egfr", 999)
    hba1c = context.get("hba1c_baseline", 0)
    c_peptide = context.get("c_peptide", 999)
    cvd = context.get("cvd", 0)
    ckd = context.get("ckd", 0)
    hypertension = context.get("hypertension", 0)
    duration = context.get("diabetes_duration", 0)

    if treatment == "Metformin":
        if 30 <= egfr < 45:
            warnings.append(f"Reduced efficacy: eGFR {egfr:.0f} is between 30-45 — dose reduction recommended, monitor renal function closely")
        elif 45 <= egfr < 60:
            warnings.append(f"eGFR {egfr:.0f} approaching threshold — periodic renal function monitoring advisable")
        if bmi > 35 and context.get("nafld", 0) == 1:
            warnings.append(f"BMI {bmi:.1f} with NAFLD — Metformin is appropriate first-line but consider GLP-1 receptor agonist for additional weight and hepatic benefits")
    elif treatment == "GLP-1":
        if age > 75:
            warnings.append(f"Patient age {age} > 75 — GLP-1 tolerability may be reduced, start with lowest dose and titrate slowly")
        if egfr < 30:
            warnings.append(f"eGFR {egfr:.0f} < 30 — limited data on GLP-1 safety in severe renal impairment, use with caution")
        if 30 <= egfr < 45:
            warnings.append(f"eGFR {egfr:.0f} between 30-45 — some GLP-1 agents may require dose adjustment, monitor renal function closely")
        if bmi < 25:
            warnings.append(f"BMI {bmi:.1f} is in normal range — GLP-1 associated weight loss may not be desirable, monitor weight closely")
    elif treatment == "SGLT-2":
        if 25 <= egfr < 45:
            warnings.append(f"Reduced glycaemic efficacy: eGFR {egfr:.0f} is between 25-45 — cardiorenal benefits may still apply, monitor closely")
        if age > 70:
            warnings.append(f"Patient age {age} > 70 — increased risk of volume depletion and hypotension with SGLT-2, ensure adequate hydration")
        if hypertension and age > 65:
            warnings.append(f"Hypertensive patient aged {age} — SGLT-2 has beneficial blood pressure lowering effect but monitor for excessive hypotension")
    elif treatment == "DPP-4":
        if egfr < 45:
            warnings.append(f"eGFR {egfr:.0f} < 45 — DPP-4 dose adjustment required (except linagliptin which is not renally cleared)")
        if age < 45:
            warnings.append(f"Patient age {age} < 45 — DPP-4 has modest efficacy, more potent options may be appropriate for younger patients")
        if hba1c > 9.5:
            warnings.append(f"HbA1c {hba1c:.1f}% > 9.5% — DPP-4 monotherapy may provide insufficient glycaemic control, consider combination therapy")
    elif treatment == "Insulin":
        if hba1c < 8.0 and c_peptide > 1.5:
            warnings.append(f"HbA1c {hba1c:.1f}% with preserved C-peptide {c_peptide:.2f} — insulin may not be necessary, risk of hypoglycaemia and weight gain")
        if hba1c < 9.0 and c_peptide > 1.0:
            warnings.append(f"HbA1c {hba1c:.1f}% with C-peptide {c_peptide:.2f} — consider whether oral agents could achieve adequate control before initiating insulin")
        if bmi > 30:
            warnings.append(f"BMI {bmi:.1f} > 30 — insulin-associated weight gain may worsen metabolic profile, consider weight-neutral alternatives")
        if age > 70:
            warnings.append(f"Patient age {age} > 70 — increased hypoglycaemia risk with insulin in elderly, conservative dosing recommended")
        if cvd == 1:
            warnings.append(f"Patient has cardiovascular disease — hypoglycaemic episodes may increase cardiovascular risk, close glucose monitoring essential")
        if ckd == 1 or egfr < 60:
            warnings.append(f"Renal impairment (eGFR {egfr:.0f}) — insulin clearance may be reduced, start with lower doses and titrate cautiously")
        if duration > 15 and c_peptide < 0.5:
            warnings.append(f"Long-standing diabetes ({duration:.0f} years) with severely depleted C-peptide ({c_peptide:.2f}) — ensure robust hypoglycaemia education")
    return warnings


def run_safety_checks(context: Dict) -> Dict:
    all_contraindications = {}
    all_warnings = {}
    for treatment in TREATMENTS:
        contras = check_contraindications(context, treatment)
        warns = check_warnings(context, treatment)
        if contras:
            all_contraindications[treatment] = contras
        if warns:
            all_warnings[treatment] = warns
    return {"contraindications": all_contraindications, "warnings": all_warnings}


def get_safety_for_recommended(safety_all: Dict, recommended: str) -> Dict:
    rec_contras = safety_all["contraindications"].get(recommended, [])
    rec_warns = safety_all["warnings"].get(recommended, [])
    excluded = {t: reasons for t, reasons in safety_all["contraindications"].items()}
    if rec_contras:
        status = "CONTRAINDICATION_FOUND"
    elif rec_warns:
        status = "WARNING"
    else:
        status = "CLEAR"
    return {
        "recommended_contraindications": rec_contras, "recommended_warnings": rec_warns,
        "excluded_treatments": excluded, "all_warnings": safety_all["warnings"], "status": status,
    }


# ─── Fairness ───

CLINICAL_FEATURES = ["bmi", "hba1c_baseline", "egfr", "diabetes_duration", "fasting_glucose", "c_peptide", "bp_systolic", "ldl", "hdl", "triglycerides", "alt", "cvd", "ckd", "nafld", "hypertension"]
DUAL_USE_FEATURES = {"age": "Age is used as a clinical risk factor (renal function decline, frailty risk, drug tolerability) — not as a demographic discriminator"}
EXCLUDED_PROTECTED = ["gender", "ethnicity", "socioeconomic_status"]


def check_fairness() -> Dict:
    return {
        "decision_features": CLINICAL_FEATURES, "dual_use_features": DUAL_USE_FEATURES,
        "excluded_protected_features": EXCLUDED_PROTECTED,
        "statement": "Recommendation based solely on clinical features. Gender, ethnicity, and socioeconomic status are not used in the model's decision-making process. Age is included as a clinical risk factor for drug tolerability and organ function, not as a demographic characteristic.",
    }


# ─── ExplainabilityExtractor ───

class ExplainabilityExtractor:
    def __init__(self, model, n_confidence_draws: int = 200):
        self.model = model
        self.n_confidence_draws = n_confidence_draws
        logger.info(f"ExplainabilityExtractor initialized — confidence method: posterior sampling ({n_confidence_draws} draws)")

    def extract_patient_context(self, context: Dict) -> Dict:
        binary_map = {0: "No", 1: "Yes"}
        return {
            "age": context["age"], "bmi": round(context["bmi"], 1),
            "hba1c_baseline": round(context["hba1c_baseline"], 1), "egfr": round(context["egfr"], 1),
            "diabetes_duration": round(context["diabetes_duration"], 1), "fasting_glucose": round(context["fasting_glucose"], 1),
            "c_peptide": round(context["c_peptide"], 2), "bp_systolic": round(context["bp_systolic"], 1),
            "ldl": round(context["ldl"], 1), "hdl": round(context["hdl"], 1),
            "triglycerides": round(context["triglycerides"], 1),
            "cvd": binary_map.get(context["cvd"], str(context["cvd"])),
            "ckd": binary_map.get(context["ckd"], str(context["ckd"])),
            "nafld": binary_map.get(context["nafld"], str(context["nafld"])),
            "hypertension": binary_map.get(context["hypertension"], str(context["hypertension"])),
        }

    def extract_model_decision(self, x: np.ndarray) -> Dict:
        confidence = self.model.compute_confidence(x, n_draws=self.n_confidence_draws)
        recommended = confidence["recommended"]
        win_rates = confidence["win_rates"]
        posterior_means = confidence["posterior_means"]
        sorted_treatments = sorted(win_rates.items(), key=lambda t: t[1], reverse=True)
        runner_up = sorted_treatments[1][0]
        runner_up_win_rate = sorted_treatments[1][1]
        sorted_means = sorted(posterior_means.items(), key=lambda t: t[1], reverse=True)
        mean_gap = sorted_means[0][1] - sorted_means[1][1]
        return {
            "recommended_treatment": recommended, "recommended_idx": confidence["recommended_idx"],
            "confidence_pct": confidence["confidence_pct"], "confidence_label": confidence["confidence_label"],
            "win_rates": win_rates, "posterior_means": posterior_means,
            "runner_up": runner_up, "runner_up_win_rate": round(runner_up_win_rate, 3),
            "mean_gap": round(mean_gap, 2), "n_draws": confidence["n_draws"],
        }

    def extract_safety_and_fairness(self, context: Dict, recommended: str) -> Dict:
        safety_all = run_safety_checks(context)
        safety = get_safety_for_recommended(safety_all, recommended)
        fairness = check_fairness()
        if safety["status"] == "CONTRAINDICATION_FOUND":
            logger.warning(f"SAFETY ALERT: {recommended} has contraindications: {safety['recommended_contraindications']}")
        elif safety["status"] == "WARNING":
            logger.info(f"Safety warnings for {recommended}: {safety['recommended_warnings']}")
        else:
            logger.info(f"Safety check CLEAR for {recommended}")
        return {"safety": safety, "fairness": fairness}

    def extract(self, context: Dict, x: np.ndarray) -> Dict:
        patient = self.extract_patient_context(context)
        decision = self.extract_model_decision(x)
        safety_fairness = self.extract_safety_and_fairness(context, decision["recommended_treatment"])
        payload = {"patient": patient, "decision": decision, "safety": safety_fairness["safety"], "fairness": safety_fairness["fairness"]}
        logger.info(
            f"Extracted payload: recommended={decision['recommended_treatment']}, "
            f"confidence={decision['confidence_pct']}% ({decision['confidence_label']}), "
            f"safety={safety_fairness['safety']['status']}"
        )
        return payload