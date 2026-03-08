import uuid

from sqlalchemy import String, Integer, Float, Text, ForeignKey, Enum as SAEnum
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.shared.database import BaseModel
from src.modules.predictions.domain.models.enums import DoctorDecision


class Prediction(BaseModel):
    __tablename__ = "predictions"

    # ── Links ──
    medical_record_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("medical_records.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("patients.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    created_by: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True,
    )

    # ── Model recommendation ──
    recommended_treatment: Mapped[str] = mapped_column(String(50), nullable=False)
    recommended_idx: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence_pct: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence_label: Mapped[str] = mapped_column(String(20), nullable=False)
    mean_gap: Mapped[float] = mapped_column(Float, nullable=False)
    runner_up: Mapped[str] = mapped_column(String(50), nullable=False)
    runner_up_win_rate: Mapped[float] = mapped_column(Float, nullable=False)

    # ── Full payload (structured JSON for variable-length data) ──
    win_rates: Mapped[dict] = mapped_column(JSONB, nullable=False)
    posterior_means: Mapped[dict] = mapped_column(JSONB, nullable=False)

    # ── Safety ──
    safety_status: Mapped[str] = mapped_column(String(30), nullable=False)
    safety_details: Mapped[dict] = mapped_column(JSONB, nullable=False)

    # ── Fairness ──
    fairness: Mapped[dict] = mapped_column(JSONB, nullable=False)

    # ── LLM Explanation (structured, always present) ──
    explanation_summary: Mapped[str] = mapped_column(Text, nullable=False)
    explanation_runner_up: Mapped[str] = mapped_column(Text, nullable=False)
    explanation_confidence: Mapped[str] = mapped_column(Text, nullable=False)
    explanation_safety: Mapped[str] = mapped_column(Text, nullable=False)
    explanation_monitoring: Mapped[str] = mapped_column(Text, nullable=False)
    explanation_disclaimer: Mapped[str] = mapped_column(Text, nullable=False)

    # ── Doctor's decision ──
    doctor_decision: Mapped[DoctorDecision] = mapped_column(
        SAEnum(DoctorDecision, name="doctor_decision_enum"),
        nullable=False, default=DoctorDecision.PENDING,
    )
    final_treatment: Mapped[str | None] = mapped_column(String(50), nullable=True)
    doctor_notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    # ── Relationships ──
    medical_record = relationship("MedicalRecord", lazy="selectin")
    patient = relationship("Patient", lazy="selectin")
    doctor = relationship("User", lazy="selectin")