import uuid

from sqlalchemy import Integer, Float, String, Boolean, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.shared.database import BaseModel


class SimulationStep(BaseModel):
    __tablename__ = "simulation_steps"

    simulation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("simulations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    step_number: Mapped[int] = mapped_column(Integer, nullable=False)
    epsilon: Mapped[float] = mapped_column(Float, nullable=False)

    # ── Patient context (raw 16 features) ──
    patient_context: Mapped[dict] = mapped_column(JSON, nullable=False)

    # ── Oracle ground truth ──
    oracle_rewards: Mapped[dict] = mapped_column(JSON, nullable=False)
    optimal_treatment: Mapped[str] = mapped_column(String(50), nullable=False)
    optimal_reward: Mapped[float] = mapped_column(Float, nullable=False)

    # ── Model decision ──
    selected_treatment: Mapped[str] = mapped_column(String(50), nullable=False)
    selected_idx: Mapped[int] = mapped_column(Integer, nullable=False)
    posterior_means: Mapped[dict] = mapped_column(JSON, nullable=False)
    win_rates: Mapped[dict] = mapped_column(JSON, nullable=False)
    confidence_pct: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence_label: Mapped[str] = mapped_column(String(20), nullable=False)
    sampled_values: Mapped[dict] = mapped_column(JSON, nullable=False)
    runner_up: Mapped[str] = mapped_column(String(50), nullable=False)
    runner_up_winrate: Mapped[float] = mapped_column(Float, nullable=False)
    mean_gap: Mapped[float] = mapped_column(Float, nullable=False)

    # ── Exploration ──
    thompson_explored: Mapped[bool] = mapped_column(Boolean, nullable=False)
    epsilon_explored: Mapped[bool] = mapped_column(Boolean, nullable=False)
    posterior_mean_best: Mapped[str] = mapped_column(String(50), nullable=False)

    # ── Outcome ──
    observed_reward: Mapped[float] = mapped_column(Float, nullable=False)
    instantaneous_regret: Mapped[float] = mapped_column(Float, nullable=False)
    matched_oracle: Mapped[bool] = mapped_column(Boolean, nullable=False)

    # ── Safety ──
    safety_status: Mapped[str] = mapped_column(String(30), nullable=False)
    safety_contraindications: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    safety_warnings: Mapped[list] = mapped_column(JSON, nullable=False, default=list)

    # ── Running aggregates ──
    cumulative_reward: Mapped[float] = mapped_column(Float, nullable=False)
    cumulative_regret: Mapped[float] = mapped_column(Float, nullable=False)
    running_accuracy: Mapped[float] = mapped_column(Float, nullable=False)
    treatment_counts: Mapped[dict] = mapped_column(JSON, nullable=False)
    running_estimates: Mapped[dict] = mapped_column(JSON, nullable=False)

    # ── Relationships ──
    simulation = relationship("Simulation", back_populates="steps")