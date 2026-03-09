from pydantic import BaseModel, Field


class StartSimulationRequest(BaseModel):
    """
    Optional simulation config sent as form fields alongside the CSV file upload.
    All fields have sensible defaults from the notebook.
    """
    initial_epsilon: float = Field(
        default=0.3, ge=0.0, le=1.0,
        description="Starting epsilon for decay schedule",
    )
    epsilon_decay: float = Field(
        default=0.997, ge=0.9, le=1.0,
        description="Multiplicative decay factor per step",
    )
    min_epsilon: float = Field(
        default=0.01, ge=0.0, le=1.0,
        description="Floor for epsilon",
    )
    random_seed: int = Field(
        default=42, ge=0, le=999999,
        description="Random seed for reproducibility",
    )
    reset_posterior: bool = Field(
        default=True,
        description="If true, model starts from prior. If false, keeps learned posterior.",
    )