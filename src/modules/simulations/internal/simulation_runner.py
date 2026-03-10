"""
Simulation Runner — runs simulation directly in the API process.

No arq worker, no Redis Streams. The simulation runs as a background
asyncio task, streams results via an in-memory asyncio.Queue, and
batch-persists steps to the database.

Architecture:
    POST /simulations → creates simulation + launches background task
    GET  /simulations/{id}/stream → subscribes to the queue via SSE
    POST /simulations/{id}/cancel → cancels a running simulation
    Background task → computes steps, pushes to queue, persists to DB

NOTE: The simulation registry is process-local. It will not survive
process restarts or scale across multiple workers. If horizontal scaling
is needed, replace with Redis pub/sub or similar distributed mechanism.
"""

import csv
import io
import json
import logging
import asyncio
import time
import warnings
import numpy as np
from typing import Dict, List, AsyncIterator
from uuid import UUID
from collections import Counter

# Suppress sklearn feature name warnings from transform_single
warnings.filterwarnings("ignore", message="X does not have valid feature names")

from src.modules.models.internal.constants import (
    TREATMENTS, N_TREATMENTS, IDX_TO_TREATMENT, CONTEXT_FEATURES,
)
from src.modules.models.internal.model_loader import registry
from src.modules.models.internal.explainability import (
    run_safety_checks, get_safety_for_recommended,
)
from src.modules.simulations.internal.reward_oracle import reward_oracle
from src.modules.simulations.domain.models.simulation_step import SimulationStep
from src.modules.simulations.domain.models.enums import SimulationStatus
from src.modules.simulations.presentation.dtos.responses import SSEStepResponse
from src.shared.database import async_session
from src.modules.simulations.domain.repositories.simulation_repository import SimulationRepository
from src.modules.simulations.domain.repositories.simulation_step_repository import SimulationStepRepository

logger = logging.getLogger(__name__)

MIN_PATIENTS = 100
DB_BATCH_SIZE = 100
PROGRESS_UPDATE_INTERVAL = 100
MAX_DB_FLUSH_FAILURES = 3

# Registry TTL: entries older than this are swept even if not cleaned up.
# Guards against leaked entries from crashed tasks.
REGISTRY_TTL_SECONDS = 3600  # 1 hour

FEATURE_RANGES = {
    "age": (18, 120), "bmi": (10.0, 80.0), "hba1c_baseline": (3.0, 20.0),
    "egfr": (5.0, 200.0), "diabetes_duration": (0.0, 60.0),
    "fasting_glucose": (50.0, 500.0), "c_peptide": (0.0, 10.0),
    "cvd": (0, 1), "ckd": (0, 1), "nafld": (0, 1), "hypertension": (0, 1),
    "bp_systolic": (60.0, 250.0), "ldl": (20.0, 400.0), "hdl": (10.0, 150.0),
    "triglycerides": (30.0, 800.0), "alt": (5.0, 500.0),
}
BINARY_FEATURES = {"cvd", "ckd", "nafld", "hypertension"}
INT_FEATURES = {"age"}


# ─────────────────────────────────────────────────────────────
# In-memory simulation registry — tracks running simulations
# ─────────────────────────────────────────────────────────────

class SimulationRegistry:
    """
    Tracks running simulations and their event queues.
    SSE clients subscribe by getting a queue for a simulation ID.
    Multiple clients can subscribe to the same simulation.

    NOTE: Process-local only. Will not work across multiple workers.
    For horizontal scaling, replace with Redis pub/sub or similar.
    """

    def __init__(self):
        self._simulations: Dict[UUID, Dict] = {}

    def register(self, simulation_id: UUID) -> None:
        self._simulations[simulation_id] = {
            "subscribers": [],
            "history": [],
            "completed": False,
            "cancelled": False,
            "task": None,
            "registered_at": time.monotonic(),
        }

    def set_task(self, simulation_id: UUID, task: asyncio.Task) -> None:
        """Store the asyncio.Task so we can cancel it later."""
        sim = self._simulations.get(simulation_id)
        if sim:
            sim["task"] = task

    def is_registered(self, simulation_id: UUID) -> bool:
        return simulation_id in self._simulations

    def is_completed(self, simulation_id: UUID) -> bool:
        sim = self._simulations.get(simulation_id)
        return sim["completed"] if sim else True

    def is_cancelled(self, simulation_id: UUID) -> bool:
        sim = self._simulations.get(simulation_id)
        return sim["cancelled"] if sim else False

    def cancel(self, simulation_id: UUID) -> bool:
        """
        Request cancellation of a running simulation.
        Sets the cancelled flag (checked by the run loop) and cancels the asyncio.Task.
        Returns True if the simulation was found and cancellation was requested.
        """
        sim = self._simulations.get(simulation_id)
        if not sim or sim["completed"]:
            return False
        sim["cancelled"] = True
        task = sim.get("task")
        if task and not task.done():
            task.cancel()
        return True

    def subscribe(self, simulation_id: UUID) -> asyncio.Queue | None:
        sim = self._simulations.get(simulation_id)
        if not sim:
            return None
        q: asyncio.Queue = asyncio.Queue(maxsize=1000)
        # Replay history for late-joining clients
        for event in sim["history"]:
            q.put_nowait(event)
        sim["subscribers"].append(q)
        return q

    def unsubscribe(self, simulation_id: UUID, queue: asyncio.Queue) -> None:
        sim = self._simulations.get(simulation_id)
        if sim and queue in sim["subscribers"]:
            sim["subscribers"].remove(queue)

    async def publish(self, simulation_id: UUID, event: Dict) -> None:
        sim = self._simulations.get(simulation_id)
        if not sim:
            return
        sim["history"].append(event)
        dead_queues = []
        for q in sim["subscribers"]:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                dead_queues.append(q)
        for q in dead_queues:
            sim["subscribers"].remove(q)

    def mark_completed(self, simulation_id: UUID) -> None:
        sim = self._simulations.get(simulation_id)
        if sim:
            sim["completed"] = True

    def cleanup(self, simulation_id: UUID) -> None:
        self._simulations.pop(simulation_id, None)

    def sweep_stale(self) -> int:
        """
        Remove registry entries older than REGISTRY_TTL_SECONDS.
        Call this periodically (e.g. from a background task or before register)
        to prevent memory leaks from orphaned entries.
        Returns the number of entries swept.
        """
        now = time.monotonic()
        stale = [
            sid for sid, data in self._simulations.items()
            if now - data["registered_at"] > REGISTRY_TTL_SECONDS
        ]
        for sid in stale:
            logger.warning("Sweeping stale simulation registry entry: %s", sid)
            self._simulations.pop(sid, None)
        return len(stale)


# Global singleton
simulation_registry = SimulationRegistry()


# ─────────────────────────────────────────────────────────────
# CSV Parsing
# ─────────────────────────────────────────────────────────────

def parse_and_validate_csv(csv_content: str) -> List[Dict]:
    reader = csv.DictReader(io.StringIO(csv_content))
    if reader.fieldnames is None:
        raise ValueError("CSV file is empty or has no headers")
    headers = [h.strip() for h in reader.fieldnames]
    missing = [f for f in CONTEXT_FEATURES if f not in headers]
    if missing:
        raise ValueError(f"CSV missing required feature columns: {', '.join(missing)}")
    patients, errors = [], []
    for row_idx, raw_row in enumerate(reader, start=2):
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
                    errors.append(f"Row {row_idx}: '{feat}' value {parsed} out of range [{lo}, {hi}]")
                    continue
                patient[feat] = parsed
            except (ValueError, TypeError):
                errors.append(f"Row {row_idx}: '{feat}' has invalid value '{val}'")
        if len(patient) == len(CONTEXT_FEATURES):
            patients.append(patient)
    if errors:
        raise ValueError(
            f"CSV validation failed with {len(errors)} error(s):\n"
            + "\n".join(errors[:20])
            + (f"\n... and {len(errors) - 20} more" if len(errors) > 20 else "")
        )
    if len(patients) < MIN_PATIENTS:
        raise ValueError(f"CSV must contain at least {MIN_PATIENTS} valid patient rows, found {len(patients)}")
    return patients


# ─────────────────────────────────────────────────────────────
# DB persistence helper
# ─────────────────────────────────────────────────────────────

def _build_step_entity(simulation_id: UUID, payload: Dict, aggregates: Dict) -> SimulationStep:
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


async def _flush_to_db(simulation_id: UUID, step_buffer: List[SimulationStep], step_number: int) -> None:
    """
    Flush a batch of steps to the database.
    Raises on failure so the caller can track it — no longer silently swallowed.
    """
    if not step_buffer:
        return
    async with async_session() as session:
        async with session.begin():
            step_repo = SimulationStepRepository(session)
            await step_repo.create_batch(step_buffer)
    if step_number % PROGRESS_UPDATE_INTERVAL == 0 or step_number > 0:
        async with async_session() as session:
            async with session.begin():
                repo = SimulationRepository(session)
                await repo.update_progress(simulation_id, step_number)


# ─────────────────────────────────────────────────────────────
# CPU-heavy step computation — runs in thread pool
# ─────────────────────────────────────────────────────────────

def _compute_step(patient, epsilon, model, pipeline, rng):
    """
    All CPU-heavy work for a single simulation step.
    Runs in a thread via asyncio.to_thread so the event loop stays free.
    Uses per-simulation rng instead of global np.random to avoid
    cross-contamination between concurrent simulations.

    IMPORTANT: This function mutates `model` via update_posterior().
    This is safe because steps are executed sequentially (one await per step).
    Do NOT parallelize step execution without adding a lock around model access.
    """
    x = pipeline.transform_single(patient)

    oracle_rewards = {t: reward_oracle(patient, t, noise=False) for t in TREATMENTS}
    oracle_rewards_list = [oracle_rewards[t] for t in TREATMENTS]
    optimal_idx = int(np.argmax(oracle_rewards_list))
    optimal_treatment = TREATMENTS[optimal_idx]
    optimal_reward = oracle_rewards_list[optimal_idx]

    selected_idx, sampled_values = model.select_action(x)
    selected_treatment = IDX_TO_TREATMENT[selected_idx]

    confidence = model.compute_confidence(x, n_draws=200)

    posterior_mean_best = max(
        confidence["posterior_means"], key=confidence["posterior_means"].get
    )
    explored = selected_treatment != posterior_mean_best
    epsilon_explored = bool(rng.random() < epsilon)

    observed_reward = reward_oracle(patient, selected_treatment, noise=True)
    instantaneous_regret = optimal_reward - reward_oracle(
        patient, selected_treatment, noise=False
    )
    matched_oracle = selected_idx == optimal_idx

    safety_all = run_safety_checks(patient)
    safety_for_selected = get_safety_for_recommended(safety_all, selected_treatment)

    # Update posterior inside the thread — safe because steps run sequentially.
    model.update_posterior(x, selected_idx, observed_reward)

    return {
        "oracle_rewards": oracle_rewards,
        "optimal_idx": optimal_idx,
        "optimal_treatment": optimal_treatment,
        "optimal_reward": optimal_reward,
        "selected_idx": selected_idx,
        "selected_treatment": selected_treatment,
        "sampled_values": sampled_values,
        "confidence": confidence,
        "posterior_mean_best": posterior_mean_best,
        "explored": explored,
        "epsilon_explored": epsilon_explored,
        "observed_reward": observed_reward,
        "instantaneous_regret": instantaneous_regret,
        "matched_oracle": matched_oracle,
        "safety_for_selected": safety_for_selected,
    }


# ─────────────────────────────────────────────────────────────
# Main simulation loop — runs as a background asyncio task
# ─────────────────────────────────────────────────────────────

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
    Run the full bandit simulation in-process.

    1. Mark simulation as RUNNING
    2. Register in the simulation registry for SSE subscribers
    3. For each patient: compute step in thread pool, publish to subscribers, buffer for DB
    4. Batch-persist to DB every DB_BATCH_SIZE steps
    5. On completion: save final aggregates, mark COMPLETED, cleanup registry
    6. On cancellation: flush remaining steps, mark CANCELLED, cleanup registry
    """
    logger.info("Simulation %s starting (n=%d)", simulation_id, len(patients))

    # Sweep stale entries before registering a new one
    swept = simulation_registry.sweep_stale()
    if swept:
        logger.info("Swept %d stale registry entries", swept)

    # Register for SSE subscribers
    simulation_registry.register(simulation_id)

    try:
        # Mark as running
        async with async_session() as session:
            async with session.begin():
                repo = SimulationRepository(session)
                await repo.update_status(simulation_id, SimulationStatus.RUNNING)

        # Load fresh model instance
        bundle = registry.get("default")
        model = registry.clone_fresh("default", reset_posterior=reset_posterior)
        pipeline = bundle.pipeline

        # Per-simulation RNG — avoids cross-contamination between
        # concurrent simulations sharing the default thread pool
        rng = np.random.RandomState(random_seed)

        n_patients = len(patients)
        cumulative_reward = 0.0
        cumulative_regret = 0.0
        correct_count = 0
        thompson_explore_count = 0
        treatment_counts = {t: 0 for t in TREATMENTS}
        treatment_rewards = {t: 0.0 for t in TREATMENTS}
        confidence_labels = []
        safety_statuses = []
        step_buffer: List[SimulationStep] = []
        db_flush_failures = 0

        for i in range(n_patients):
            # ── Check for cancellation ──
            if simulation_registry.is_cancelled(simulation_id):
                logger.info("Simulation %s cancelled at step %d/%d", simulation_id, i + 1, n_patients)

                # Flush any buffered steps before marking cancelled
                if step_buffer:
                    try:
                        await _flush_to_db(simulation_id, step_buffer, i)
                        step_buffer.clear()
                    except Exception as e:
                        logger.error("Failed to flush steps on cancel: %s", e)

                # Mark cancelled in DB
                async with async_session() as session:
                    async with session.begin():
                        repo = SimulationRepository(session)
                        await repo.update_status(simulation_id, SimulationStatus.CANCELLED)

                # Notify subscribers
                await simulation_registry.publish(simulation_id, {
                    "type": "complete",
                    "data": {"status": "CANCELLED", "cancelled_at_step": i},
                })
                simulation_registry.mark_completed(simulation_id)
                simulation_registry.cleanup(simulation_id)
                return

            patient = patients[i]
            epsilon = max(min_epsilon, initial_epsilon * (epsilon_decay ** (i + 1)))

            # ── Compute step in thread pool (non-blocking) ──
            result = await asyncio.to_thread(
                _compute_step, patient, epsilon, model, pipeline, rng
            )

            # ── Unpack results ──
            oracle_rewards = result["oracle_rewards"]
            optimal_treatment = result["optimal_treatment"]
            optimal_reward = result["optimal_reward"]
            selected_idx = result["selected_idx"]
            selected_treatment = result["selected_treatment"]
            sampled_values = result["sampled_values"]
            confidence = result["confidence"]
            posterior_mean_best = result["posterior_mean_best"]
            explored = result["explored"]
            epsilon_explored = result["epsilon_explored"]
            observed_reward = result["observed_reward"]
            instantaneous_regret = result["instantaneous_regret"]
            matched_oracle = result["matched_oracle"]
            safety_for_selected = result["safety_for_selected"]

            # ── Update aggregates ──
            cumulative_reward += observed_reward
            cumulative_regret += instantaneous_regret
            treatment_counts[selected_treatment] += 1
            treatment_rewards[selected_treatment] += observed_reward
            if matched_oracle:
                correct_count += 1
            if explored:
                thompson_explore_count += 1

            running_accuracy = correct_count / (i + 1)
            confidence_labels.append(confidence["confidence_label"])
            safety_statuses.append(safety_for_selected["status"])

            running_estimates = {
                t: round(treatment_rewards[t] / max(treatment_counts[t], 1), 4)
                for t in TREATMENTS
            }

            # ── Build full payload for DB persistence ──
            payload = {
                "step": i + 1,
                "total_steps": n_patients,
                "epsilon": round(epsilon, 6),
                "patient": {feat: patient[feat] for feat in CONTEXT_FEATURES},
                "oracle": {
                    "rewards": {t: round(r, 4) for t, r in oracle_rewards.items()},
                    "optimal_treatment": optimal_treatment,
                    "optimal_idx": result["optimal_idx"],
                    "optimal_reward": round(optimal_reward, 4),
                },
                "decision": {
                    "selected_treatment": selected_treatment,
                    "selected_idx": selected_idx,
                    "sampled_values": {
                        IDX_TO_TREATMENT[j]: round(float(sampled_values[j]), 4)
                        for j in range(N_TREATMENTS)
                    },
                    "posterior_means": confidence["posterior_means"],
                    "win_rates": confidence["win_rates"],
                    "confidence_pct": confidence["confidence_pct"],
                    "confidence_label": confidence["confidence_label"],
                    "runner_up": confidence["runner_up"],
                    "runner_up_winrate": confidence["runner_up_win_rate"],
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
                        t: reasons for t, reasons
                        in safety_for_selected["excluded_treatments"].items()
                    },
                },
                "aggregates": {
                    "cumulative_reward": round(cumulative_reward, 4),
                    "cumulative_regret": round(cumulative_regret, 4),
                    "running_accuracy": round(running_accuracy, 4),
                    "treatment_counts": dict(treatment_counts),
                    "running_estimates": running_estimates,
                },
            }

            # ── Log progress ──
            if (i + 1) % 10 == 0 or (i + 1) <= 5:
                logger.info(
                    "Step %d/%d | %s -> %s (oracle: %s) | reward=%.2f | regret=%.2f | acc=%.2f%% | confidence=%s",
                    i + 1, n_patients,
                    selected_treatment,
                    "Y" if matched_oracle else "N",
                    optimal_treatment,
                    observed_reward,
                    instantaneous_regret,
                    running_accuracy * 100,
                    confidence["confidence_label"],
                )

            # ── Publish lean SSE payload via DTO ──
            sse_step = SSEStepResponse.from_runner({
                "step": i + 1,
                "total_steps": n_patients,
                "selected_idx": selected_idx,
                "selected_treatment": selected_treatment,
                "explored": explored,
                "observed_reward": round(observed_reward, 4),
                "epsilon": round(epsilon, 6),
                "running_estimates": running_estimates,
                "running_accuracy": round(running_accuracy, 4),
                "cumulative_reward": round(cumulative_reward, 4),
                "cumulative_regret": round(cumulative_regret, 4),
                "treatment_counts": dict(treatment_counts),
            })

            await simulation_registry.publish(simulation_id, {
                "type": "step",
                "data": sse_step.model_dump_json(by_alias=True),
            })

            # ── Buffer for DB (full payload) ──
            step_entity = _build_step_entity(simulation_id, payload, payload["aggregates"])
            step_buffer.append(step_entity)

            # ── Batch flush to DB ──
            if len(step_buffer) >= DB_BATCH_SIZE:
                try:
                    await _flush_to_db(simulation_id, step_buffer, i + 1)
                    step_buffer.clear()
                    db_flush_failures = 0
                except Exception as e:
                    db_flush_failures += 1
                    logger.error(
                        "DB flush failed for %s at step %d (%d/%d failures): %s",
                        simulation_id, i + 1, db_flush_failures, MAX_DB_FLUSH_FAILURES, e,
                    )
                    if db_flush_failures >= MAX_DB_FLUSH_FAILURES:
                        raise RuntimeError(
                            f"Simulation aborted: {db_flush_failures} consecutive DB flush failures. "
                            f"Last error: {e}"
                        )
                    # Keep buffer intact so next flush retries these steps

        # ── Flush remaining steps ──
        if step_buffer:
            try:
                await _flush_to_db(simulation_id, step_buffer, n_patients)
                step_buffer.clear()
            except Exception as e:
                raise RuntimeError(f"Final DB flush failed: {e}")

        # ── Final aggregates ──
        final = {
            "final_accuracy": round(running_accuracy, 4),
            "final_cumulative_reward": round(cumulative_reward, 4),
            "final_cumulative_regret": round(cumulative_regret, 4),
            "mean_reward": round(cumulative_reward / n_patients, 4),
            "mean_regret": round(cumulative_regret / n_patients, 4),
            "thompson_exploration_rate": round(thompson_explore_count / n_patients, 4),
            "treatment_counts": dict(treatment_counts),
            "confidence_distribution": dict(Counter(confidence_labels)),
            "safety_distribution": dict(Counter(safety_statuses)),
        }

        async with async_session() as session:
            async with session.begin():
                repo = SimulationRepository(session)
                await repo.save_final_aggregates(simulation_id, final)

        # ── Notify subscribers of completion ──
        await simulation_registry.publish(simulation_id, {
            "type": "complete",
            "data": {"status": "COMPLETED", **final},
        })
        simulation_registry.mark_completed(simulation_id)
        simulation_registry.cleanup(simulation_id)

        logger.info(
            "Simulation %s completed: accuracy=%.4f, reward=%.4f, regret=%.4f",
            simulation_id, running_accuracy, cumulative_reward, cumulative_regret,
        )

    except asyncio.CancelledError:
        # Task was cancelled externally (e.g. via cancel endpoint)
        logger.info("Simulation %s task cancelled via asyncio", simulation_id)

        await simulation_registry.publish(simulation_id, {
            "type": "complete",
            "data": {"status": "CANCELLED"},
        })
        simulation_registry.mark_completed(simulation_id)
        simulation_registry.cleanup(simulation_id)

        try:
            async with async_session() as session:
                async with session.begin():
                    repo = SimulationRepository(session)
                    await repo.update_status(simulation_id, SimulationStatus.CANCELLED)
        except Exception:
            logger.exception("Failed to update simulation status on cancel")

    except Exception as e:
        logger.exception("Simulation %s failed: %s", simulation_id, e)

        # Notify subscribers of error
        await simulation_registry.publish(simulation_id, {
            "type": "error",
            "data": {"error": str(e)},
        })
        simulation_registry.mark_completed(simulation_id)
        simulation_registry.cleanup(simulation_id)

        try:
            async with async_session() as session:
                async with session.begin():
                    repo = SimulationRepository(session)
                    await repo.update_status(
                        simulation_id, SimulationStatus.FAILED, error_message=str(e),
                    )
        except Exception:
            logger.exception("Failed to update simulation status")