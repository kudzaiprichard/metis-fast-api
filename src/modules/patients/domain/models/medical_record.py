import uuid

from sqlalchemy import Integer, Float, Boolean, ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.shared.database import BaseModel


class MedicalRecord(BaseModel):
    __tablename__ = "medical_records"

    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("patients.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # ── Clinical context features (16 features for model input) ──
    age: Mapped[int] = mapped_column(Integer, nullable=False)
    bmi: Mapped[float] = mapped_column(Float, nullable=False)
    hba1c_baseline: Mapped[float] = mapped_column(Float, nullable=False)
    egfr: Mapped[float] = mapped_column(Float, nullable=False)
    diabetes_duration: Mapped[float] = mapped_column(Float, nullable=False)
    fasting_glucose: Mapped[float] = mapped_column(Float, nullable=False)
    c_peptide: Mapped[float] = mapped_column(Float, nullable=False)

    # Comorbidities (binary)
    cvd: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    ckd: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    nafld: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    hypertension: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Vitals and labs
    bp_systolic: Mapped[float] = mapped_column(Float, nullable=False)
    ldl: Mapped[float] = mapped_column(Float, nullable=False)
    hdl: Mapped[float] = mapped_column(Float, nullable=False)
    triglycerides: Mapped[float] = mapped_column(Float, nullable=False)
    alt: Mapped[float] = mapped_column(Float, nullable=False)

    # Optional notes from the doctor
    notes: Mapped[str | None] = mapped_column(String(1000), nullable=True)

    # Relationships
    patient = relationship("Patient", back_populates="medical_records")