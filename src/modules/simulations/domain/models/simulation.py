import uuid

from sqlalchemy import Integer, Float, String, Boolean, Enum as SAEnum, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.shared.database import BaseModel
from src.modules.simulations.domain.models.enums import SimulationStatus


class Simulation(BaseModel):
    __tablename__ = "simulations"

    # ── Config ──
    initial_epsilon: Mapped[float] = mapped_column(Float, nullable=False)
    epsilon_decay: Mapped[float] = mapped_column(Float, nullable=False)
    min_epsilon: Mapped[float] = mapped_column(Float, nullable=False)
    random_seed: Mapped[int] = mapped_column(Integer, nullable=False, default=42)
    reset_posterior: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    # ── Dataset ──
    dataset_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    dataset_row_count: Mapped[int] = mapped_column(Integer, nullable=False)

    # ── Status ──
    status: Mapped[SimulationStatus] = mapped_column(
        SAEnum(SimulationStatus), nullable=False, default=SimulationStatus.PENDING
    )
    current_step: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_message: Mapped[str | None] = mapped_column(String(1000), nullable=True)

    # ── Final aggregates (populated on completion) ──
    final_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    final_cumulative_reward: Mapped[float | None] = mapped_column(Float, nullable=True)
    final_cumulative_regret: Mapped[float | None] = mapped_column(Float, nullable=True)
    mean_reward: Mapped[float | None] = mapped_column(Float, nullable=True)
    mean_regret: Mapped[float | None] = mapped_column(Float, nullable=True)
    thompson_exploration_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    treatment_counts: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    confidence_distribution: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    safety_distribution: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # ── Relationships ──
    steps = relationship(
        "SimulationStep",
        back_populates="simulation",
        cascade="all, delete-orphan",
        order_by="SimulationStep.step_number.asc()",
        lazy="noload",
    )