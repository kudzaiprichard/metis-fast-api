"""
Simulation Runner — async background task that executes the bandit loop.

Mirrors the notebook's Cell 3 (extract_step) and Cell 4 (simulation loop)
exactly. Pushes each step to the stream manager and persists to DB.
"""

import csv
import io
import logging
import asyncio
import numpy as np
from typing import Dict, List
from uuid import UUID
from collections import Counter

from src.modules.models.internal.constants import (
    TREATMENTS, N_TREATMENTS, IDX_TO_TREATMENT, CONTEXT_FEATURES,
)
from src.modules.models.internal.neural_bandit import NeuralThompson
from src.modules.models.internal.feature_engineering import FeaturePipeline
from src.modules.models.internal.explainability import (
    run_safety_checks,
    get_safety_for_recommended,
)
from src.modules.simulations.internal.reward_oracle import reward_oracle
from src.modules.simulations.internal.stream_manager import stream_manager
from src.modules.simulations.domain.models.simulation_step import SimulationStep
from src.modules.simulations.domain.models.enums import SimulationStatus

from src.shared.database import async_session

from src.modules.simulations.domain.repositories.simulation_repository import SimulationRepository
from src.modules.simulations.domain.repositories.simulation_step_repository import SimulationStepRepository

logger = logging.getLogger(__name__)

# How often to flush steps to DB (every N steps)
DB_FLUSH_INTERVAL = 50
# How often to update simulation.current_step in DB
PROGRESS_UPDATE_INTERVAL = 10
# Minimum number of patients required in CSV
MIN_PATIENTS = 100

# Validation ranges (same as PredictRequest)
FEATURE_RANGES = {
    "age": (18, 120),
    "bmi": (10.0, 80.0),
    "hba1c_baseline": (3.0, 20.0),
    "egfr": (5.0, 200.0),
    "diabetes_duration": (0.0, 60.0),
    "fasting_glucose": (50.0, 500.0),
    "c_peptide": (0.0, 10.0),
    "cvd": (0, 1),
    "ckd": (0, 1),
    "nafld": (0, 1),
    "hypertension": (0, 1),
    "bp_systolic": (60.0, 250.0),
    "ldl": (20.0, 400.0),
    "hdl": (10.0, 150.0),
    "triglycerides": (30.0, 800.0),
    "alt": (5.0, 500.0),
}

BINARY_FEATURES = {"cvd", "ckd", "nafld", "hypertension"}
INT_FEATURES = {"age"}


def parse_and_validate_csv(csv_content: str) -> List[Dict]:
    """
    Parse uploaded CSV into list of patient context dicts.
    Validates: column presence, data types, value ranges, minimum row count.
    """
    reader = csv.DictReader(io.StringIO(csv_content))

    if reader.fieldnames is None:
        raise ValueError("CSV file is empty or has no headers")

    # Strip whitespace from headers
    headers = [h.strip() for h in reader.fieldnames]

    # Check all 16 features are present
    missing = [f for f in CONTEXT_FEATURES if f not in headers]
    if missing:
        raise ValueError(f"CSV missing required feature columns: {', '.join(missing)}")

    patients = []
    errors = []

    for row_idx, raw_row in enumerate(reader, start=2):
        # Strip whitespace from keys
        row = {k.strip(): v.strip() if v else v for k, v in raw_row.items()}
        patient = {}

        for feat in CONTEXT_FEATURES:
            val = row.get(feat)

            if val is None or val == "":
                errors.append(f"Row {row_idx}: missing value for '{feat}'")
                continue

            try:
                if feat in BINARY_FEATURES:
                    parsed = int(float(val))
                    if parsed not in (0, 1):
                        errors.append(f"Row {row_idx}: '{feat}' must be 0 or 1, got {parsed}")
                        continue
                elif feat in INT_FEATURES:
                    parsed = int(float(val))
                else:
                    parsed = float(val)

                lo, hi = FEATURE_RANGES[feat]
                if not (lo <= parsed <= hi):
                    errors.append(
                        f"Row {row_idx}: '{feat}' value {parsed} out of range [{lo}, {hi}]"
                    )
                    continue

                patient[feat] = parsed

            except (ValueError, TypeError):
                errors.append(f"Row {row_idx}: '{feat}' has invalid value '{val}'")

        if len(patient) == len(CONTEXT_FEATURES):
            patients.append(patient)

    if errors:
        # Return first 20 errors to avoid overwhelming response
        raise ValueError(
            f"CSV validation failed with {len(errors)} error(s):\n"
            + "\n".join(errors[:20])
            + (f"\n... and {len(errors) - 20} more" if len(errors) > 20 else "")
        )

    if len(patients) < MIN_PATIENTS:
        raise ValueError(
            f"CSV must contain at least {MIN_PATIENTS} valid patient rows, "
            f"found {len(patients)}"
        )

    return patients


def extract_step(
    patient_context: Dict,
    model: NeuralThompson,
    pipeline: FeaturePipeline,
    step_number: int,
    epsilon: float,
) -> Dict:
    """
    Extract the full payload for one simulation step.
    Identical to notebook Cell 3 — not modified.
    """
    x = pipeline.transform_single(patient_context)

    # Oracle ground truth
    oracle_rewards = {
        t: reward_oracle(patient_context, t, noise=False)
        for t in TREATMENTS
    }
    oracle_rewards_list = [oracle_rewards[t] for t in TREATMENTS]
    optimal_idx = int(np.argmax(oracle_rewards_list))
    optimal_treatment = TREATMENTS[optimal_idx]
    optimal_reward = oracle_rewards_list[optimal_idx]

    # Model decision via Thompson Sampling
    selected_idx, sampled_values = model.select_action(x)
    selected_treatment = IDX_TO_TREATMENT[selected_idx]

    # Posterior means and confidence
    confidence = model.compute_confidence(x, n_draws=200)

    # Runner-up
    sorted_by_winrate = sorted(
        confidence["win_rates"].items(), key=lambda t: t[1], reverse=True
    )
    runner_up_treatment = sorted_by_winrate[1][0]
    runner_up_winrate = sorted_by_winrate[1][1]

    # Exploration vs exploitation
    posterior_mean_best = max(
        confidence["posterior_means"], key=confidence["posterior_means"].get
    )
    explored = selected_treatment != posterior_mean_best
    epsilon_explored = np.random.random() < epsilon

    # Observed reward
    observed_reward = reward_oracle(patient_context, selected_treatment, noise=True)
    instantaneous_regret = optimal_reward - reward_oracle(
        patient_context, selected_treatment, noise=False
    )
    matched_oracle = selected_idx == optimal_idx

    # Safety checks
    safety_all = run_safety_checks(patient_context)
    safety_for_selected = get_safety_for_recommended(safety_all, selected_treatment)

    payload = {
        "step": step_number,
        "epsilon": round(epsilon, 6),
        "patient": {feat: patient_context[feat] for feat in CONTEXT_FEATURES},
        "oracle": {
            "rewards": {t: round(r, 4) for t, r in oracle_rewards.items()},
            "optimal_treatment": optimal_treatment,
            "optimal_idx": optimal_idx,
            "optimal_reward": round(optimal_reward, 4),
        },
        "decision": {
            "selected_treatment": selected_treatment,
            "selected_idx": selected_idx,
            "sampled_values": {
                IDX_TO_TREATMENT[i]: round(float(sampled_values[i]), 4)
                for i in range(N_TREATMENTS)
            },
            "posterior_means": confidence["posterior_means"],
            "win_rates": confidence["win_rates"],
            "confidence_pct": confidence["confidence_pct"],
            "confidence_label": confidence["confidence_label"],
            "runner_up": runner_up_treatment,
            "runner_up_winrate": round(runner_up_winrate, 3),
            "mean_gap": confidence["mean_gap"],
        },
        "exploration": {
            "thompson_explored": explored,
            "epsilon_explored": epsilon_explored,
            "posterior_mean_best": posterior_mean_best,
        },
        "outcome": {
            "observed_reward": round(observed_reward, 4),
            "instantaneous_regret": round(instantaneous_regret, 4),
            "matched_oracle": matched_oracle,
        },
        "safety": {
            "status": safety_for_selected["status"],
            "contraindications": safety_for_selected["recommended_contraindications"],
            "warnings": safety_for_selected["recommended_warnings"],
            "excluded_treatments": {
                t: reasons
                for t, reasons in safety_for_selected["excluded_treatments"].items()
            },
        },
    }

    return payload


def _build_step_entity(
    simulation_id: UUID,
    payload: Dict,
    aggregates: Dict,
) -> SimulationStep:
    """Convert a step payload + running aggregates into a SimulationStep ORM entity."""
    return SimulationStep(
        simulation_id=simulation_id,
        step_number=payload["step"],
        epsilon=payload["epsilon"],
        patient_context=payload["patient"],
        oracle_rewards=payload["oracle"]["rewards"],
        optimal_treatment=payload["oracle"]["optimal_treatment"],
        optimal_reward=payload["oracle"]["optimal_reward"],
        selected_treatment=payload["decision"]["selected_treatment"],
        selected_idx=payload["decision"]["selected_idx"],
        posterior_means=payload["decision"]["posterior_means"],
        win_rates=payload["decision"]["win_rates"],
        confidence_pct=payload["decision"]["confidence_pct"],
        confidence_label=payload["decision"]["confidence_label"],
        sampled_values=payload["decision"]["sampled_values"],
        runner_up=payload["decision"]["runner_up"],
        runner_up_winrate=payload["decision"]["runner_up_winrate"],
        mean_gap=payload["decision"]["mean_gap"],
        thompson_explored=payload["exploration"]["thompson_explored"],
        epsilon_explored=payload["exploration"]["epsilon_explored"],
        posterior_mean_best=payload["exploration"]["posterior_mean_best"],
        observed_reward=payload["outcome"]["observed_reward"],
        instantaneous_regret=payload["outcome"]["instantaneous_regret"],
        matched_oracle=payload["outcome"]["matched_oracle"],
        safety_status=payload["safety"]["status"],
        safety_contraindications=payload["safety"]["contraindications"],
        safety_warnings=payload["safety"]["warnings"],
        cumulative_reward=aggregates["cumulative_reward"],
        cumulative_regret=aggregates["cumulative_regret"],
        running_accuracy=aggregates["running_accuracy"],
        treatment_counts=aggregates["treatment_counts"],
        running_estimates=aggregates["running_estimates"],
    )


async def run_simulation(
    simulation_id: UUID,
    patients: List[Dict],
    initial_epsilon: float = 0.3,
    epsilon_decay: float = 0.997,
    min_epsilon: float = 0.01,
    random_seed: int = 42,
    reset_posterior: bool = True,
) -> None:
    """
    Background task: runs the full bandit simulation loop.

    Mirrors notebook Cell 4 exactly:
    - Load model once, reset posterior
    - For each patient: extract_step → update aggregates → update posterior → stream
    - Save final aggregates to DB

    Args:
        simulation_id: UUID of the simulation record
        patients: list of validated patient context dicts from CSV
        initial_epsilon: starting epsilon for decay schedule
        epsilon_decay: multiplicative decay factor per step
        min_epsilon: floor for epsilon
        random_seed: seed for reproducibility
        reset_posterior: if True, start from prior; if False, keep learned posterior
    """
    from src.modules.models.internal.model_loader import registry

    n_patients = len(patients)

    logger.info("Simulation %s starting (n=%d, seed=%d)", simulation_id, n_patients, random_seed)

    # Register stream before anything else
    await stream_manager.register_simulation(simulation_id)

    try:
        # ── Load model (once for entire simulation) ──
        bundle = registry.get("default")
        pipeline = bundle.pipeline

        # Fresh instance — doesn't mutate the shared model
        model = registry.clone_fresh("default", reset_posterior=reset_posterior)

        # ── Seed for reproducibility ──
        np.random.seed(random_seed)

        # ── Running aggregates ──
        cumulative_reward = 0.0
        cumulative_regret = 0.0
        correct_count = 0
        thompson_explore_count = 0
        treatment_counts = {t: 0 for t in TREATMENTS}
        treatment_rewards = {t: 0.0 for t in TREATMENTS}
        confidence_labels = []
        safety_statuses = []

        step_buffer: List[SimulationStep] = []

        # ── Mark as running ──
        async with async_session() as session:
            async with session.begin():
                repo = SimulationRepository(session)
                await repo.update_status(simulation_id, SimulationStatus.RUNNING)

        # ── Main loop ──
        for i in range(n_patients):
            patient = patients[i]

            # Current epsilon
            epsilon = max(min_epsilon, initial_epsilon * (epsilon_decay ** (i + 1)))

            # Extract step payload (identical to notebook)
            payload = extract_step(patient, model, pipeline, step_number=i + 1, epsilon=epsilon)

            # Update running aggregates
            selected = payload["decision"]["selected_treatment"]
            reward = payload["outcome"]["observed_reward"]
            regret = payload["outcome"]["instantaneous_regret"]
            matched = payload["outcome"]["matched_oracle"]

            cumulative_reward += reward
            cumulative_regret += regret
            treatment_counts[selected] += 1
            treatment_rewards[selected] += reward
            if matched:
                correct_count += 1
            if payload["exploration"]["thompson_explored"]:
                thompson_explore_count += 1

            running_accuracy = correct_count / (i + 1)

            confidence_labels.append(payload["decision"]["confidence_label"])
            safety_statuses.append(payload["safety"]["status"])

            running_estimates = {
                t: round(treatment_rewards[t] / max(treatment_counts[t], 1), 4)
                for t in TREATMENTS
            }

            aggregates = {
                "cumulative_reward": round(cumulative_reward, 4),
                "cumulative_regret": round(cumulative_regret, 4),
                "running_accuracy": round(running_accuracy, 4),
                "treatment_counts": dict(treatment_counts),
                "running_estimates": running_estimates,
            }

            # Attach aggregates to payload for streaming
            payload["aggregates"] = aggregates

            # Update model posterior (model learns over time)
            x = pipeline.transform_single(patient)
            model.update_posterior(x, payload["decision"]["selected_idx"], reward)

            # Push to SSE stream
            await stream_manager.push_event(simulation_id, payload)

            # Buffer step for DB persistence
            step_entity = _build_step_entity(simulation_id, payload, aggregates)
            step_buffer.append(step_entity)

            # Periodic DB flush
            if len(step_buffer) >= DB_FLUSH_INTERVAL:
                async with async_session() as session:
                    async with session.begin():
                        step_repo = SimulationStepRepository(session)
                        await step_repo.create_batch(step_buffer)
                step_buffer.clear()

            # Periodic progress update
            if (i + 1) % PROGRESS_UPDATE_INTERVAL == 0:
                async with async_session() as session:
                    async with session.begin():
                        repo = SimulationRepository(session)
                        await repo.update_progress(simulation_id, i + 1)

            # Yield control to event loop
            await asyncio.sleep(0)

        # ── Flush remaining steps ──
        if step_buffer:
            async with async_session() as session:
                async with session.begin():
                    step_repo = SimulationStepRepository(session)
                    await step_repo.create_batch(step_buffer)
            step_buffer.clear()

        # ── Compute final aggregates ──
        confidence_dist = dict(Counter(confidence_labels))
        safety_dist = dict(Counter(safety_statuses))

        final_aggregates = {
            "final_accuracy": round(running_accuracy, 4),
            "final_cumulative_reward": round(cumulative_reward, 4),
            "final_cumulative_regret": round(cumulative_regret, 4),
            "mean_reward": round(cumulative_reward / n_patients, 4),
            "mean_regret": round(cumulative_regret / n_patients, 4),
            "thompson_exploration_rate": round(thompson_explore_count / n_patients, 4),
            "treatment_counts": dict(treatment_counts),
            "confidence_distribution": confidence_dist,
            "safety_distribution": safety_dist,
        }

        async with async_session() as session:
            async with session.begin():
                repo = SimulationRepository(session)
                await repo.save_final_aggregates(simulation_id, final_aggregates)

        logger.info(
            "Simulation %s completed: accuracy=%.4f, reward=%.2f, regret=%.2f",
            simulation_id, running_accuracy, cumulative_reward, cumulative_regret,
        )

        # Signal stream completion
        await stream_manager.push_complete(simulation_id)

    except Exception as e:
        logger.exception("Simulation %s failed: %s", simulation_id, e)

        # Mark failed in DB
        try:
            async with async_session() as session:
                async with session.begin():
                    repo = SimulationRepository(session)
                    await repo.update_status(
                        simulation_id, SimulationStatus.FAILED, error_message=str(e)
                    )
        except Exception:
            logger.exception("Failed to update simulation status to FAILED")

        # Notify stream subscribers
        await stream_manager.push_error(simulation_id, str(e))
        await stream_manager.push_complete(simulation_id)

    finally:
        # Cleanup stream after a delay (give clients time to receive final events)
        await asyncio.sleep(5)
        await stream_manager.cleanup(simulation_id)