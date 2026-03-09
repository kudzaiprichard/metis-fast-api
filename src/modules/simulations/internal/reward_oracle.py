"""
Reward oracle for simulation module.
Copied from ML project data_generator.py — only reward_oracle.
Import paths updated to use models module constants.
Implementation unchanged to preserve identical behaviour with notebook.
"""

import numpy as np
from typing import Dict

from src.modules.models.internal.constants import TREATMENTS

# Treatment-specific noise std devs
TREATMENT_NOISE = {
    "Metformin": 0.3,
    "GLP-1": 0.4,
    "SGLT-2": 0.4,
    "DPP-4": 0.3,
    "Insulin": 0.5,
}


def reward_oracle(context: Dict, treatment: str, noise: bool = True) -> float:
    age = context["age"]
    bmi = context["bmi"]
    hba1c = context["hba1c_baseline"]
    egfr = context["egfr"]
    duration = context["diabetes_duration"]
    fg = context["fasting_glucose"]
    cpep = context["c_peptide"]
    cvd = context["cvd"]
    ckd = context["ckd"]
    nafld = context["nafld"]

    base = 0.15 * (hba1c - 7.0)
    base = np.clip(base, -0.2, 0.8)

    if treatment == "Metformin":
        r = 1.5
        young = age < 60
        lean = bmi < 32
        early = duration < 7
        good_kidney = egfr > 60
        good_beta = cpep > 1.2
        niche_score = young + lean + early + good_kidney + good_beta
        if niche_score >= 4:
            r += 4.5
        elif niche_score == 3:
            r += 2.5
        elif niche_score == 2:
            r += 1.0
        r += 0.5 * (hba1c < 9.0)
        r -= 3.5 * (egfr < 30)
        r -= 2.0 * (egfr < 45) * (egfr >= 30)
        r -= 1.5 * (bmi > 37)
        r -= 1.5 * (duration > 12)
        r -= 1.0 * (age > 70)
        r -= 1.0 * (cpep < 0.7)
        r -= 0.8 * (hba1c > 10)

    elif treatment == "GLP-1":
        r = 0.5
        obese = bmi > 35
        very_obese = bmi > 39
        has_nafld = nafld == 1
        has_cvd = cvd == 1
        if very_obese and has_nafld:
            r += 5.5
        elif very_obese:
            r += 3.5
        elif obese and has_nafld and has_cvd:
            r += 4.5
        elif obese and has_nafld:
            r += 3.0
        elif obese and has_cvd:
            r += 2.5
        elif obese:
            r += 1.5
        r += 0.5 * (cpep > 1.0)
        r += 0.5 * (hba1c > 9)
        r -= 5.0 * (bmi < 28)
        r -= 3.0 * (bmi < 32) * (bmi >= 28)
        r -= 1.0 * (age > 78)
        r -= 0.5 * (cpep < 0.5)

    elif treatment == "SGLT-2":
        r = 0.5
        has_cvd = cvd == 1
        has_ckd = ckd == 1
        decent_kidney = egfr >= 30
        good_kidney = egfr >= 45
        if has_cvd and good_kidney:
            r += 5.0
        elif has_cvd and decent_kidney:
            r += 3.5
        elif has_cvd:
            r += 1.5
        if has_ckd and decent_kidney and not has_cvd:
            r += 3.0
        elif has_ckd and decent_kidney and has_cvd:
            r += 1.5
        r += 0.5 * (bmi > 30) * has_cvd
        r += 0.3 * (cpep > 1.0)
        r -= 4.0 * (1 - has_cvd) * (1 - has_ckd)
        r -= 2.5 * (egfr < 25)
        r -= 0.5 * (bmi < 27) * (1 - has_cvd)

    elif treatment == "DPP-4":
        r = 0.8
        elderly = age > 60
        very_elderly = age > 70
        has_ckd = ckd == 1
        low_egfr = egfr < 60
        mod_disease = 7.5 < hba1c < 10.0
        mod_cpep = 0.7 < cpep < 1.5
        if very_elderly and (has_ckd or low_egfr):
            r += 5.5
        elif very_elderly:
            r += 3.5
        elif elderly and (has_ckd or low_egfr):
            r += 4.0
        elif elderly:
            r += 2.0
        elif has_ckd or low_egfr:
            r += 2.0
        r += 1.0 * mod_disease
        r += 0.8 * mod_cpep
        r += 0.5 * (low_egfr and egfr >= 30)
        r -= 4.0 * (age < 45)
        r -= 2.5 * (age < 55) * (age >= 45)
        r -= 1.5 * (hba1c > 10.5)
        r -= 1.0 * (1 - has_ckd) * (age < 60) * (egfr >= 60)
        r -= 0.5 * (cpep < 0.5)

    elif treatment == "Insulin":
        r = 0.0
        severe_hba1c = hba1c > 10
        very_severe_hba1c = hba1c > 11.5
        low_cpep = cpep < 0.8
        very_low_cpep = cpep < 0.5
        long_duration = duration > 15
        high_fg = fg > 220
        if very_low_cpep and very_severe_hba1c:
            r += 7.0
        elif very_low_cpep and severe_hba1c:
            r += 5.5
        elif very_low_cpep:
            r += 4.5
        elif low_cpep and very_severe_hba1c:
            r += 5.0
        elif low_cpep and severe_hba1c:
            r += 4.0
        elif severe_hba1c and long_duration:
            r += 3.5
        elif severe_hba1c:
            r += 2.0
        elif low_cpep:
            r += 2.5
        r += 1.5 * long_duration
        r += 1.0 * high_fg
        r += 0.5 * (duration > 10) * (duration <= 15)
        r -= 4.5 * (hba1c < 8.0) * (cpep > 1.5)
        r -= 3.0 * (hba1c < 9.0) * (cpep > 1.2)
        r -= 2.0 * (duration < 3) * (cpep > 1.0)
        r -= 1.5 * (age > 75)
        r -= 0.5 * (cpep > 2.0)

    else:
        raise ValueError(f"Unknown treatment: {treatment}")

    reward = base + r

    if noise:
        sigma = TREATMENT_NOISE[treatment]
        reward += np.random.normal(0, sigma)

    return float(np.clip(reward, 0.0, 10.0))