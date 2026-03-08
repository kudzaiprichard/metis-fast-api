"""
LLM Explanation Generator for NeuralThompson
Copied from ML project — import paths updated for FastAPI integration.

Only change: src.data_generator → src.modules.models.internal.constants
"""

import os
import json
from typing import Dict, Optional
import logging
logger = logging.getLogger(__name__)

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("google-genai not installed — LLM explanations unavailable")

from src.modules.models.internal.constants import TREATMENTS


SYSTEM_PROMPT = """You are Dr. Sarah Chen, a senior endocrinologist with 40 years of \
clinical experience specialising in Type 2 Diabetes management. You have treated \
over 30,000 patients across diverse populations, from newly diagnosed cases to \
complex multi-comorbidity presentations. You are also trained in interpreting \
AI-assisted clinical decision support tools and translating their outputs into \
language that fellow clinicians can immediately understand and act upon.

You are reviewing the output of an AI treatment recommendation model that \
predicts expected HbA1c reduction (in percentage points) for each of five \
treatment options given a patient's clinical profile. The model also provides \
a confidence score — the percentage of times this treatment won when the model \
simulated the decision 200 times using its own uncertainty estimates.

Your task is to FAITHFULLY explain the model's recommendation in clear clinical \
language that a practising physician would find useful during a consultation.

CRITICAL TRANSLATION RULES — follow these exactly:

1. POSTERIOR MEANS = PREDICTED HbA1c REDUCTION. Always refer to these as \
   "predicted HbA1c reduction" — never say "posterior mean", "sampled score", or "reward".

2. CONFIDENCE = WIN RATE. Use the EXACT percentage provided.

3. WIN RATES = TREATMENT COMPARISON.

4. MEAN GAP = EXPECTED BENEFIT DIFFERENCE.

5. SAFETY = REPORT EXACTLY as provided.

6. NO ML JARGON.

CRITICAL: You MUST respond with ONLY a valid JSON object. No markdown, no \
backticks, no preamble, no explanation outside the JSON. Just the raw JSON."""


RESPONSE_FORMAT_INSTRUCTION = """RESPONSE FORMAT:

You MUST respond with ONLY a valid JSON object matching this exact structure:

{
    "recommendation_summary": "2-3 sentences explaining why this treatment is recommended.",
    "runner_up_analysis": "1-2 sentences on the next best alternative.",
    "confidence_statement": "1-2 sentences stating the model's confidence as a percentage.",
    "safety_assessment": "1-3 sentences reporting safety check results.",
    "monitoring_note": "1-2 sentences with specific monitoring recommendations.",
    "disclaimer": "This is an AI-assisted decision support tool. Final treatment decisions must be made by the treating physician."
}

REMEMBER: Output ONLY the JSON object. Nothing else."""


REQUIRED_KEYS = [
    "recommendation_summary", "runner_up_analysis", "confidence_statement",
    "safety_assessment", "monitoring_note", "disclaimer",
]


def build_prompt(payload: Dict) -> str:
    p = payload["patient"]
    d = payload["decision"]
    s = payload["safety"]
    f = payload["fairness"]

    patient_section = f"""PATIENT PROFILE:
  Age: {p['age']} years | BMI: {p['bmi']} kg/m² | HbA1c: {p['hba1c_baseline']}%
  eGFR: {p['egfr']} mL/min/1.73m² | Duration: {p['diabetes_duration']} years
  Fasting Glucose: {p['fasting_glucose']} mg/dL | C-Peptide: {p['c_peptide']} ng/mL
  BP: {p['bp_systolic']} mmHg | LDL/HDL/TG: {p['ldl']}/{p['hdl']}/{p['triglycerides']} mg/dL
  CVD: {p['cvd']} | CKD: {p['ckd']} | NAFLD: {p['nafld']} | Hypertension: {p['hypertension']}"""

    means = d["posterior_means"]
    win_rates = d["win_rates"]

    model_section = f"""MODEL PREDICTION:
  Recommended: {d['recommended_treatment']} | Confidence: {d['confidence_pct']}% ({d['confidence_label']})
  ({d['recommended_treatment']} won {d['confidence_pct']} out of {d['n_draws']} simulations)

  Predicted HbA1c Reduction (pp): Metformin→{means['Metformin']:.1f} | GLP-1→{means['GLP-1']:.1f} | SGLT-2→{means['SGLT-2']:.1f} | DPP-4→{means['DPP-4']:.1f} | Insulin→{means['Insulin']:.1f}
  Win Rates: Metformin→{win_rates['Metformin']:.1%} | GLP-1→{win_rates['GLP-1']:.1%} | SGLT-2→{win_rates['SGLT-2']:.1%} | DPP-4→{win_rates['DPP-4']:.1%} | Insulin→{win_rates['Insulin']:.1%}
  Runner-up: {d['runner_up']} ({d['runner_up_win_rate']:.1%}) | Benefit gap: {d['mean_gap']:.1f} pp"""

    safety_lines = [f"SAFETY: Status={s['status']}"]
    if s["recommended_contraindications"]:
        for c in s["recommended_contraindications"]:
            safety_lines.append(f"  CONTRA: {c}")
    if s["recommended_warnings"]:
        for w in s["recommended_warnings"]:
            safety_lines.append(f"  WARN: {w}")
    if s["excluded_treatments"]:
        for t, reasons in s["excluded_treatments"].items():
            for r in reasons:
                safety_lines.append(f"  EXCLUDED {t}: {r}")
    if s.get("all_warnings"):
        for t, warns in s["all_warnings"].items():
            if t != d["recommended_treatment"]:
                for w in warns:
                    safety_lines.append(f"  WARN {t}: {w}")
    safety_section = "\n".join(safety_lines)

    fairness_section = f"FAIRNESS: {f['statement']}"

    return f"{patient_section}\n\n---\n\n{model_section}\n\n---\n\n{safety_section}\n\n---\n\n{fairness_section}\n\n---\n\n{RESPONSE_FORMAT_INSTRUCTION}"


def parse_llm_response(raw_text: str) -> Dict[str, str]:
    text = raw_text.strip()
    text = text.replace('\uff5b', '{').replace('\uff5d', '}')
    text = text.replace('\u200b', '').replace('\ufeff', '')
    text = text.encode('ascii', 'ignore').decode('ascii')
    if text.startswith("```"):
        first_newline = text.index("\n")
        text = text[first_newline + 1:]
        if text.endswith("```"):
            text = text[:-3].strip()
        elif "```" in text:
            text = text[:text.rfind("```")].strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in LLM response: {raw_text[:200]}")
    json_str = text[start:end]
    try:
        result = json.loads(json_str)
    except json.JSONDecodeError as e:
        repaired = _attempt_json_repair(json_str)
        if repaired is not None:
            result = repaired
        else:
            raise ValueError(f"Invalid JSON from LLM: {e}\nRaw: {json_str[:300]}")
    missing = [k for k in REQUIRED_KEYS if k not in result]
    if missing:
        if missing == ["disclaimer"]:
            result["disclaimer"] = "This is an AI-assisted decision support tool. Final treatment decisions must be made by the treating physician based on the complete clinical picture, patient preferences, and current guidelines."
        else:
            raise ValueError(f"LLM response missing required keys: {missing}")
    for key in REQUIRED_KEYS:
        result[key] = str(result[key])
    return result


def _attempt_json_repair(json_str: str) -> Optional[Dict]:
    attempts = [json_str + '"}', json_str + '"}\n}', json_str + '..."}']
    for attempt in attempts:
        try:
            result = json.loads(attempt)
            if isinstance(result, dict):
                logger.warning("Repaired truncated JSON from LLM response")
                return result
        except json.JSONDecodeError:
            continue
    return None


class LLMExplainer:
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash", temperature: float = 0.3, max_retries: int = 2):
        if not GEMINI_AVAILABLE:
            raise ImportError("google-genai is required. Install with: pip install google-genai")
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key required. Pass api_key= or set GEMINI_API_KEY env var.")
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        logger.info(f"LLMExplainer initialized: model={model_name}, temp={temperature}")

    def explain(self, payload: Dict) -> Dict[str, str]:
        prompt = build_prompt(payload)
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[
                        types.Content(role="user", parts=[types.Part.from_text(text=SYSTEM_PROMPT)]),
                        types.Content(role="model", parts=[types.Part.from_text(text="Understood. I will respond as Dr. Sarah Chen with only a valid JSON object, using clinical language throughout and no ML terminology.")]),
                        types.Content(role="user", parts=[types.Part.from_text(text=prompt)]),
                    ],
                    config=types.GenerateContentConfig(temperature=self.temperature, max_output_tokens=4096),
                )
                result = parse_llm_response(response.text)
                logger.info(f"Explanation generated (attempt {attempt + 1}): treatment={payload['decision']['recommended_treatment']}")
                return result
            except ValueError as e:
                logger.warning(f"Parse failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries:
                    continue
                raise ValueError(f"Failed to parse LLM response after {self.max_retries + 1} attempts: {e}") from e
            except Exception as e:
                raise RuntimeError(f"Gemini API call failed: {e}") from e

    def explain_batch(self, payloads: list) -> list:
        explanations = []
        for i, payload in enumerate(payloads):
            logger.info(f"Generating explanation {i + 1}/{len(payloads)}...")
            explanation = self.explain(payload)
            explanations.append(explanation)
        return explanations