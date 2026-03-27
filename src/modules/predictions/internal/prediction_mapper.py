"""
Map a ``PredictionResult`` (the engine's flat output) into a ``Prediction``
ORM row ready for persistence.

Open-decision (audit §5.1) — Schema strategy
--------------------------------------------
**Choice: Option (a), default (fast).** Serialise the new structured
``PredictionResult`` into the existing JSONB columns on the ``predictions``
table. No Alembic migration in this phase.

Concretely:

- ``safety_details`` JSONB — repurposed as the safety envelope for new rows:
  ``{safety_findings, excluded_treatments, override}``. Old rows keep their
  legacy keys (``recommended_contraindications``, ``recommended_warnings``,
  ``all_warnings``) and remain readable via the response DTO's permissive
  passthrough.

- ``fairness`` JSONB — ``PredictionResult`` has no ``fairness`` block, so this
  column is reused as the engine-extras envelope for new rows:
  ``{model_top_treatment, attribution, contrast, uncertainty_drivers}``. The
  legacy fairness block on old rows (``decision_features`` etc.) stays intact
  and is still returned as-is. The naming is misleading; a follow-up should
  rename this column once Phase 3 settles (tracked under Phase 3 follow-ups
  in the audit).

- Every other scalar column (``recommended_treatment`` … ``runner_up_win_rate``,
  ``win_rates``, ``posterior_means``, ``safety_status``, the six
  ``explanation_*`` text columns) maps 1:1 from ``PredictionResult``.

``explain="require"`` guarantees a non-null ``explanation`` block (see
engine.apredict); callers must honour that contract — the mapper raises a
clear ValueError if the explanation is missing, rather than storing empty
strings into NOT NULL columns.
"""
from __future__ import annotations

from typing import Any, Dict
from uuid import UUID

from src.shared.inference import PredictionResult
from src.modules.predictions.domain.models.enums import DoctorDecision
from src.modules.predictions.domain.models.prediction import Prediction


def to_prediction(
    result: PredictionResult,
    *,
    medical_record_id: UUID,
    patient_id: UUID,
    doctor_id: UUID,
) -> Prediction:
    explanation = result.explanation
    if explanation is None:
        raise ValueError(
            "prediction_mapper.to_prediction requires a non-null explanation "
            "(call engine.apredict with explain=\"require\")"
        )

    safety_envelope: Dict[str, Any] = {
        "safety_findings": list(result.safety_findings),
        "excluded_treatments": dict(result.excluded_treatments),
    }
    if result.override is not None:
        safety_envelope["override"] = result.override

    engine_extras: Dict[str, Any] = {
        "model_top_treatment": result.model_top_treatment,
    }
    if result.attribution is not None:
        engine_extras["attribution"] = result.attribution
    if result.contrast is not None:
        engine_extras["contrast"] = result.contrast
    if result.uncertainty_drivers is not None:
        engine_extras["uncertainty_drivers"] = list(result.uncertainty_drivers)

    return Prediction(
        medical_record_id=medical_record_id,
        patient_id=patient_id,
        created_by=doctor_id,
        recommended_treatment=result.recommended,
        recommended_idx=result.recommended_idx,
        confidence_pct=result.confidence_pct,
        confidence_label=result.confidence_label,
        mean_gap=result.mean_gap,
        runner_up=result.runner_up,
        runner_up_win_rate=result.runner_up_win_rate,
        win_rates=dict(result.win_rates),
        posterior_means=dict(result.posterior_means),
        safety_status=result.safety_status,
        safety_details=safety_envelope,
        fairness=engine_extras,
        explanation_summary=explanation["recommendation_summary"],
        explanation_runner_up=explanation["runner_up_analysis"],
        explanation_confidence=explanation["confidence_statement"],
        explanation_safety=explanation["safety_assessment"],
        explanation_monitoring=explanation["monitoring_note"],
        explanation_disclaimer=explanation["disclaimer"],
        doctor_decision=DoctorDecision.PENDING,
    )


__all__ = ["to_prediction"]
