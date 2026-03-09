from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from src.modules.simulations.domain.models.simulation import Simulation
from src.modules.simulations.domain.models.simulation_step import SimulationStep


# ──────────────────────────────────────────────
# Simulation
# ──────────────────────────────────────────────

class SimulationConfigResponse(BaseModel):
    initial_epsilon: float = Field(alias="initialEpsilon")
    epsilon_decay: float = Field(alias="epsilonDecay")
    min_epsilon: float = Field(alias="minEpsilon")
    random_seed: int = Field(alias="randomSeed")
    reset_posterior: bool = Field(alias="resetPosterior")
    dataset_filename: str = Field(alias="datasetFilename")
    dataset_row_count: int = Field(alias="datasetRowCount")

    class Config:
        populate_by_name = True


class SimulationAggregatesResponse(BaseModel):
    final_accuracy: Optional[float] = Field(None, alias="finalAccuracy")
    final_cumulative_reward: Optional[float] = Field(None, alias="finalCumulativeReward")
    final_cumulative_regret: Optional[float] = Field(None, alias="finalCumulativeRegret")
    mean_reward: Optional[float] = Field(None, alias="meanReward")
    mean_regret: Optional[float] = Field(None, alias="meanRegret")
    thompson_exploration_rate: Optional[float] = Field(None, alias="thompsonExplorationRate")
    treatment_counts: Optional[Dict[str, int]] = Field(None, alias="treatmentCounts")
    confidence_distribution: Optional[Dict[str, int]] = Field(None, alias="confidenceDistribution")
    safety_distribution: Optional[Dict[str, int]] = Field(None, alias="safetyDistribution")

    class Config:
        populate_by_name = True


class SimulationResponse(BaseModel):
    id: UUID
    status: str
    current_step: int = Field(alias="currentStep")
    error_message: Optional[str] = Field(None, alias="errorMessage")
    config: SimulationConfigResponse
    aggregates: SimulationAggregatesResponse
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")

    class Config:
        populate_by_name = True

    @staticmethod
    def from_entity(entity: Simulation) -> "SimulationResponse":
        return SimulationResponse(
            id=entity.id,
            status=entity.status.value,
            currentStep=entity.current_step,
            errorMessage=entity.error_message,
            config=SimulationConfigResponse(
                initialEpsilon=entity.initial_epsilon,
                epsilonDecay=entity.epsilon_decay,
                minEpsilon=entity.min_epsilon,
                randomSeed=entity.random_seed,
                resetPosterior=entity.reset_posterior,
                datasetFilename=entity.dataset_filename,
                datasetRowCount=entity.dataset_row_count,
            ),
            aggregates=SimulationAggregatesResponse(
                finalAccuracy=entity.final_accuracy,
                finalCumulativeReward=entity.final_cumulative_reward,
                finalCumulativeRegret=entity.final_cumulative_regret,
                meanReward=entity.mean_reward,
                meanRegret=entity.mean_regret,
                thompsonExplorationRate=entity.thompson_exploration_rate,
                treatmentCounts=entity.treatment_counts,
                confidenceDistribution=entity.confidence_distribution,
                safetyDistribution=entity.safety_distribution,
            ),
            createdAt=entity.created_at,
            updatedAt=entity.updated_at,
        )


# ──────────────────────────────────────────────
# Simulation Step
# ──────────────────────────────────────────────

class StepOracleResponse(BaseModel):
    rewards: Dict[str, float]
    optimal_treatment: str = Field(alias="optimalTreatment")
    optimal_reward: float = Field(alias="optimalReward")

    class Config:
        populate_by_name = True


class StepDecisionResponse(BaseModel):
    selected_treatment: str = Field(alias="selectedTreatment")
    selected_idx: int = Field(alias="selectedIdx")
    posterior_means: Dict[str, float] = Field(alias="posteriorMeans")
    win_rates: Dict[str, float] = Field(alias="winRates")
    confidence_pct: int = Field(alias="confidencePct")
    confidence_label: str = Field(alias="confidenceLabel")
    sampled_values: Dict[str, float] = Field(alias="sampledValues")
    runner_up: str = Field(alias="runnerUp")
    runner_up_winrate: float = Field(alias="runnerUpWinrate")
    mean_gap: float = Field(alias="meanGap")

    class Config:
        populate_by_name = True


class StepExplorationResponse(BaseModel):
    thompson_explored: bool = Field(alias="thompsonExplored")
    epsilon_explored: bool = Field(alias="epsilonExplored")
    posterior_mean_best: str = Field(alias="posteriorMeanBest")

    class Config:
        populate_by_name = True


class StepOutcomeResponse(BaseModel):
    observed_reward: float = Field(alias="observedReward")
    instantaneous_regret: float = Field(alias="instantaneousRegret")
    matched_oracle: bool = Field(alias="matchedOracle")

    class Config:
        populate_by_name = True


class StepSafetyResponse(BaseModel):
    status: str
    contraindications: List[str]
    warnings: List[str]

    class Config:
        populate_by_name = True


class StepAggregatesResponse(BaseModel):
    cumulative_reward: float = Field(alias="cumulativeReward")
    cumulative_regret: float = Field(alias="cumulativeRegret")
    running_accuracy: float = Field(alias="runningAccuracy")
    treatment_counts: Dict[str, int] = Field(alias="treatmentCounts")
    running_estimates: Dict[str, float] = Field(alias="runningEstimates")

    class Config:
        populate_by_name = True


class SimulationStepResponse(BaseModel):
    step: int
    epsilon: float
    patient: Dict[str, float]
    oracle: StepOracleResponse
    decision: StepDecisionResponse
    exploration: StepExplorationResponse
    outcome: StepOutcomeResponse
    safety: StepSafetyResponse
    aggregates: StepAggregatesResponse

    class Config:
        populate_by_name = True

    @staticmethod
    def from_entity(entity: SimulationStep) -> "SimulationStepResponse":
        return SimulationStepResponse(
            step=entity.step_number,
            epsilon=entity.epsilon,
            patient=entity.patient_context,
            oracle=StepOracleResponse(
                rewards=entity.oracle_rewards,
                optimalTreatment=entity.optimal_treatment,
                optimalReward=entity.optimal_reward,
            ),
            decision=StepDecisionResponse(
                selectedTreatment=entity.selected_treatment,
                selectedIdx=entity.selected_idx,
                posteriorMeans=entity.posterior_means,
                winRates=entity.win_rates,
                confidencePct=entity.confidence_pct,
                confidenceLabel=entity.confidence_label,
                sampledValues=entity.sampled_values,
                runnerUp=entity.runner_up,
                runnerUpWinrate=entity.runner_up_winrate,
                meanGap=entity.mean_gap,
            ),
            exploration=StepExplorationResponse(
                thompsonExplored=entity.thompson_explored,
                epsilonExplored=entity.epsilon_explored,
                posteriorMeanBest=entity.posterior_mean_best,
            ),
            outcome=StepOutcomeResponse(
                observedReward=entity.observed_reward,
                instantaneousRegret=entity.instantaneous_regret,
                matchedOracle=entity.matched_oracle,
            ),
            safety=StepSafetyResponse(
                status=entity.safety_status,
                contraindications=entity.safety_contraindications,
                warnings=entity.safety_warnings,
            ),
            aggregates=StepAggregatesResponse(
                cumulativeReward=entity.cumulative_reward,
                cumulativeRegret=entity.cumulative_regret,
                runningAccuracy=entity.running_accuracy,
                treatmentCounts=entity.treatment_counts,
                runningEstimates=entity.running_estimates,
            ),
        )