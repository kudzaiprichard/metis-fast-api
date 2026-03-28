"""
Simulation Runner — runs a bandit simulation directly in the API process.

Phase 4 cutover: the runner no longer reaches into ``NeuralThompson`` /
``FeaturePipeline`` / safety checkers. It drives a per-simulation
``InferenceEngine.alearning_stream`` and projects the emitted
``LearningStepEvent`` + a small amount of runner-local state into the
existing ``SimulationStep`` row shape and ``SSEStepResponse`` SSE frames.

Phase-4 open-decision resolutions (§5 of the audit)
---------------------------------------------------
ε-greedy handling — DROP.
    ``LearningStream`` owns action selection atomically (Thompson-draw →
    argmax → observe → online-update). There is no public hook for
    injecting a pre-chosen action into a step. "Wrap externally" (option
    (a) in the audit) would require a second posterior update per step
    with a random arm, polluting the Bayesian linear regression and
    invalidating regret analysis. Thompson sampling already explores —
    the event exposes this via ``explored = selected_idx !=
    posterior_mean_argmax``, which we persist as ``thompson_explored``.
    We still compute and log the decayed ε value each step (so the
    frontend's epsilon-decay chart keeps working); ``epsilon_explored``
    is always False. If product wants ε-greedy back, it has to land in
    ``shared/inference`` as a first-class API.

Per-simulation posterior reset — build via ``from_config`` + optional
``model.reset_posterior()``.
    ``InferenceEngine.from_config`` loads the saved ``.pt`` checkpoint,
    which includes the trained posterior (A, A_inv, b, mu). To honour
    ``reset_posterior=True`` per simulation, we call
    ``sim_engine.model.reset_posterior()`` after construction — this
    reverts the posterior to the prior while keeping the trained network
    backbone intact. ``InferenceConfig`` exposes no ``reset_posterior``
    flag, so a post-construction call is the only route that doesn't
    require editing the frozen shared library. Each simulation gets its
    own engine, so the in-place reset does not touch the app-wide engine
    on ``app.state.engine``.

Oracle noise
------------
The legacy runner called ``reward_oracle(..., noise=True)`` to obtain
``observed_reward`` and ``noise=False`` for regret accounting. The new
``LearningStream.astep`` takes a single oracle vector and derives both
observed reward and regret from it internally — so we pass the
noise-free vector. Consequence: ``observed_reward`` and
``instantaneous_regret`` are now both noise-free in persisted rows and
SSE frames. This is a deliberate behaviour change and cleaner for
regret analysis.

NOTE: the simulation registry is process-local. It will not survive
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
from typing import Dict, List
from uuid import UUID
from collections import Counter

# Suppress sklearn feature name warnings from transform_single
warnings.filterwarnings("ignore", message="X does not have valid feature names")

from pydantic import ValidationError as PydanticValidationError

from src.shared.inference import InferenceEngine, LearningStepEvent, PatientInput
from src.shared.inference_bootstrap import build_inference_config
from src.modules.simulations.internal.reward_oracle import reward_oracle, TREATMENTS
from src.modules.simulations.domain.models.simulation_step import SimulationStep
from src.modules.simulations.domain.models.enums import SimulationStatus
from src.modules.simulations.presentation.dtos.responses import SSEStepResponse
from src.shared.database import async_session
from src.modules.simulations.domain.repositories.simulation_repository import SimulationRepository
from src.modules.simulations.domain.repositories.simulation_step_repository import SimulationStepRepository

logger = logging.getLogger(__name__)

# Treatment-index helpers derived locally (do not reach into shared/inference
# internals). Ordering matches the engine's arm indexing contract.
N_TREATMENTS = len(TREATMENTS)
IDX_TO_TREATMENT = dict(enumerate(TREATMENTS))

# The 16 clinical features the simulation CSV must carry. Matches
# ``PatientInput.feature_dict()`` keys by contract; kept local so the
# simulation's CSV schema is not coupled to the engine's private constants.
CONTEXT_FEATURES = [
    "age", "bmi", "hba1c_baseline", "egfr", "diabetes_duration",
    "fasting_glucose", "c_peptide", "cvd", "ckd", "nafld", "hypertension",
    "bp_systolic", "ldl", "hdl", "triglycerides", "alt",
]

MIN_PATIENTS = 100
MAX_PATIENTS = 50_000
MAX_CSV_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB
DB_BATCH_SIZE = 100
PROGRESS_UPDATE_INTERVAL = 100
MAX_DB_FLUSH_FAILURES = 3
MAX_DB_FLUSH_RETRIES = 3  # retries for the final flush
MAX_SUBSCRIBERS = 20
MAX_HISTORY_SIZE = 5_000  # cap in-memory history; beyond this, rely on DB replay

# Registry TTL: entries older than this are swept even if not cleaned up.
# Guards against leaked entries from crashed tasks.
REGISTRY_TTL_SECONDS = 3600  # 1 hour

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
        """
        Atomically check registration and subscribe.
        Returns None if the simulation is not registered (caller should
        fall back to DB replay). Returns a Queue otherwise.
        """
        sim = self._simulations.get(simulation_id)
        if not sim:
            return None
        if len(sim["subscribers"]) >= MAX_SUBSCRIBERS:
            return None
        q: asyncio.Queue = asyncio.Queue(maxsize=1000)
        # Replay history for late-joining clients
        for event in sim["history"]:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                break
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
        # Cap in-memory history to prevent unbounded growth.
        # Late-joining clients beyond this point fall back to DB replay.
        if len(sim["history"]) < MAX_HISTORY_SIZE:
            sim["history"].append(event)
        dead_queues = []
        for q in sim["subscribers"]:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # Send a sentinel error event before dropping the subscriber
                try:
                    q.put_nowait({
                        "type": "error",
                        "data": {"error": "Subscriber queue full — connection dropped"},
                    })
                except asyncio.QueueFull:
                    pass
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
    # PatientInput (shared/inference) owns per-field validation. We only
    # handle CSV-shape concerns here (headers, row count, file size) and
    # per-row numeric coercion, then defer to PatientInput for ranges /
    # binary literals / type coercion. This guarantees any row that passes
    # upload validation will also pass the engine's per-step _validate_patient
    # inside LearningStream.astep — no silent divergence, no mid-run failures.
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
        row_values: Dict[str, object] = {}
        row_failed = False
        for feat in CONTEXT_FEATURES:
            val = row.get(feat)
            if val is None or val == "":
                errors.append(f"Row {row_idx}: missing value for '{feat}'")
                row_failed = True
                continue
            try:
                parsed = float(val)
            except (ValueError, TypeError):
                errors.append(f"Row {row_idx}: '{feat}' has invalid value '{val}'")
                row_failed = True
                continue
            row_values[feat] = int(parsed) if parsed.is_integer() else parsed
        if row_failed:
            continue
        try:
            pi = PatientInput.model_validate(row_values)
        except PydanticValidationError as e:
            for err in e.errors():
                field = ".".join(str(p) for p in err.get("loc", ())) or "_"
                errors.append(
                    f"Row {row_idx}: '{field}' {err.get('msg', 'invalid value')}"
                )
            continue
        patients.append(pi.feature_dict())
    if errors:
        raise ValueError(
            f"CSV validation failed with {len(errors)} error(s):\n"
            + "\n".join(errors[:20])
            + (f"\n... and {len(errors) - 20} more" if len(errors) > 20 else "")
        )
    if len(patients) < MIN_PATIENTS:
        raise ValueError(f"CSV must contain at least {MIN_PATIENTS} valid patient rows, found {len(patients)}")
    if len(patients) > MAX_PATIENTS:
        raise ValueError(
            f"CSV exceeds maximum of {MAX_PATIENTS:,} patient rows (found {len(patients):,}). "
            f"Reduce the dataset size and try again."
        )
    return patients


# ─────────────────────────────────────────────────────────────
# Projection helpers — LearningStepEvent → DB row
# ─────────────────────────────────────────────────────────────

def _runner_up_from_win_rates(win_rates: Dict[str, float]) -> tuple[str, float]:
    """Second-place arm by win rate (matches legacy confidence output)."""
    ordered = sorted(win_rates.items(), key=lambda kv: kv[1], reverse=True)
    if len(ordered) < 2:
        return ordered[0] if ordered else ("", 0.0)
    name, rate = ordered[1]
    return name, float(rate)


def _build_step_entity(
    simulation_id: UUID,
    event: LearningStepEvent,
    patient: Dict,
    epsilon: float,
    oracle_rewards: Dict[str, float],
    optimal_treatment: str,
    optimal_reward: float,
    selected_treatment: str,
    posterior_mean_best: str,
    runner_up_name: str,
    runner_up_wr: float,
    running_estimates: Dict[str, float],
    treatment_counts: Dict[str, int],
) -> SimulationStep:
    return SimulationStep(
        simulation_id=simulation_id,
        step_number=event.step,
        epsilon=epsilon,
        patient_context={feat: patient[feat] for feat in CONTEXT_FEATURES},
        oracle_rewards={t: round(r, 4) for t, r in oracle_rewards.items()},
        optimal_treatment=optimal_treatment,
        optimal_reward=round(float(optimal_reward), 4),
        selected_treatment=selected_treatment,
        selected_idx=event.selectedIdx,
        posterior_means={k: float(v) for k, v in event.posteriorMeans.items()},
        win_rates={k: float(v) for k, v in event.winRates.items()},
        confidence_pct=int(event.confidencePct),
        confidence_label=event.confidenceLabel,
        sampled_values={
            k: round(float(v), 4) for k, v in event.thompsonSamples.items()
        },
        runner_up=runner_up_name,
        runner_up_winrate=runner_up_wr,
        mean_gap=float(event.meanGap),
        thompson_explored=bool(event.explored),
        # ε-greedy dropped (see module docstring); persisted False for
        # schema compatibility and so the UI can tell the two exploration
        # channels apart historically.
        epsilon_explored=False,
        posterior_mean_best=posterior_mean_best,
        observed_reward=round(float(event.observedReward), 4),
        instantaneous_regret=round(float(event.regret), 4),
        matched_oracle=bool(event.selectedIdx == event.oracleOptimalIdx),
        safety_status=event.safetyStatus,
        # shared/inference surfaces only the aggregate status per-step in
        # the event; detailed contraindication/warning lists are not part
        # of the LearningStepEvent schema. Stored empty for the frozen
        # API contract; full safety detail remains available via
        # ``engine.apredict(patient, explain=False)`` for drill-down.
        safety_contraindications=[],
        safety_warnings=[],
        cumulative_reward=round(float(event.cumulativeReward), 4),
        cumulative_regret=round(float(event.cumulativeRegret), 4),
        running_accuracy=round(float(event.runningAccuracy), 4),
        treatment_counts=treatment_counts,
        running_estimates=running_estimates,
    )


async def _flush_to_db(simulation_id: UUID, step_buffer: List[SimulationStep], step_number: int) -> None:
    """
    Flush a batch of steps to the database and update progress atomically.
    Raises on failure so the caller can track it — no longer silently swallowed.
    """
    if not step_buffer:
        return
    async with async_session() as session:
        async with session.begin():
            step_repo = SimulationStepRepository(session)
            await step_repo.create_batch(step_buffer)
            repo = SimulationRepository(session)
            await repo.update_progress(simulation_id, step_number)


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

    1. Build a per-simulation InferenceEngine (fresh posterior if requested)
    2. Mark simulation as RUNNING
    3. Register in the simulation registry for SSE subscribers
    4. Drive engine.alearning_stream; for each patient compute the oracle
       vector, call stream.astep, project the LearningStepEvent into a
       SimulationStep row and an SSEStepResponse frame
    5. Batch-persist to DB every DB_BATCH_SIZE steps
    6. On completion: save final aggregates, mark COMPLETED, cleanup registry
    7. On cancellation: flush remaining steps, mark CANCELLED, cleanup registry
    """
    logger.info("Simulation %s starting (n=%d)", simulation_id, len(patients))

    # Sweep stale entries before registering a new one
    swept = simulation_registry.sweep_stale()
    if swept:
        logger.info("Swept %d stale registry entries", swept)

    # Register for SSE subscribers
    simulation_registry.register(simulation_id)

    # Per-simulation RNG — feeds the LearningStream's Thompson draws so
    # concurrent simulations do not share the default generator.
    rng = np.random.default_rng(random_seed)

    # Per-simulation engine — isolates the sim's posterior from the
    # app-wide engine on ``app.state.engine``. Honour reset_posterior
    # via a post-construction call (see module docstring).
    sim_engine = InferenceEngine.from_config(build_inference_config())
    if reset_posterior:
        sim_engine.model.reset_posterior()

    try:
        # Mark as running
        async with async_session() as session:
            async with session.begin():
                repo = SimulationRepository(session)
                await repo.update_status(simulation_id, SimulationStatus.RUNNING)

        n_patients = len(patients)
        confidence_labels: List[str] = []
        safety_statuses: List[str] = []
        step_buffer: List[SimulationStep] = []
        db_flush_failures = 0
        thompson_explore_count = 0

        # Aggregates tracked for the final frame + progress logging. The
        # stream also emits running aggregates in the event, but some
        # end-of-run metrics (thompson exploration rate, confidence /
        # safety distributions) are not in the event schema, so we keep
        # our own counters here.
        final_event: LearningStepEvent | None = None

        async with sim_engine.alearning_stream(
            total_steps=n_patients, rng=rng,
        ) as stream:
            for i in range(n_patients):
                # ── Check for cancellation ──
                if simulation_registry.is_cancelled(simulation_id):
                    logger.info(
                        "Simulation %s cancelled at step %d/%d",
                        simulation_id, i + 1, n_patients,
                    )

                    if step_buffer:
                        try:
                            await _flush_to_db(simulation_id, step_buffer, i)
                            step_buffer.clear()
                        except Exception as e:
                            logger.error("Failed to flush steps on cancel: %s", e)

                    async with async_session() as session:
                        async with session.begin():
                            repo = SimulationRepository(session)
                            await repo.update_status(
                                simulation_id, SimulationStatus.CANCELLED,
                            )

                    await simulation_registry.publish(simulation_id, {
                        "type": "complete",
                        "data": {"status": "CANCELLED", "cancelled_at_step": i},
                    })
                    simulation_registry.mark_completed(simulation_id)
                    simulation_registry.cleanup(simulation_id)
                    return

                patient = patients[i]
                epsilon = max(
                    min_epsilon, initial_epsilon * (epsilon_decay ** (i + 1))
                )

                # Oracle vector — noise-free; see module docstring for
                # the rationale behind dropping observation noise.
                oracle_rewards = {
                    t: reward_oracle(patient, t, noise=False) for t in TREATMENTS
                }
                oracle_vector = np.array(
                    [oracle_rewards[t] for t in TREATMENTS], dtype=float,
                )
                optimal_idx = int(np.argmax(oracle_vector))
                optimal_treatment = TREATMENTS[optimal_idx]
                optimal_reward = float(oracle_vector[optimal_idx])

                # ── Drive the engine for one step ──
                event: LearningStepEvent = await stream.astep(patient, oracle_vector)
                final_event = event

                # ── Derive runner-local projection fields ──
                selected_treatment = IDX_TO_TREATMENT[event.selectedIdx]
                posterior_mean_best = IDX_TO_TREATMENT[event.bestTreatmentIdx]
                runner_up_name, runner_up_wr = _runner_up_from_win_rates(
                    event.winRates,
                )

                if event.explored:
                    thompson_explore_count += 1
                confidence_labels.append(event.confidenceLabel)
                safety_statuses.append(event.safetyStatus)

                running_estimates = {
                    t: round(float(event.runningMeanRewardPerArm.get(t, 0.0)), 4)
                    for t in TREATMENTS
                }
                treatment_counts = {
                    t: int(event.nUpdatesPerArm.get(t, 0)) for t in TREATMENTS
                }

                # ── Log progress ──
                if (i + 1) % 10 == 0 or (i + 1) <= 5:
                    logger.info(
                        "Step %d/%d | %s -> %s (oracle: %s) | "
                        "reward=%.2f | regret=%.2f | acc=%.2f%% | confidence=%s",
                        i + 1, n_patients,
                        selected_treatment,
                        "Y" if event.selectedIdx == event.oracleOptimalIdx else "N",
                        optimal_treatment,
                        event.observedReward,
                        event.regret,
                        event.runningAccuracy * 100,
                        event.confidenceLabel,
                    )

                # ── Publish lean SSE payload via DTO ──
                sse_step = SSEStepResponse.from_event(
                    event,
                    extras={
                        "total_steps": n_patients,
                        "selected_treatment": selected_treatment,
                        "epsilon": round(epsilon, 6),
                        "running_estimates": running_estimates,
                        "treatment_counts": treatment_counts,
                    },
                )

                await simulation_registry.publish(simulation_id, {
                    "type": "step",
                    "data": sse_step.model_dump_json(by_alias=True),
                })

                # ── Buffer for DB (full payload) ──
                step_entity = _build_step_entity(
                    simulation_id=simulation_id,
                    event=event,
                    patient=patient,
                    epsilon=round(epsilon, 6),
                    oracle_rewards=oracle_rewards,
                    optimal_treatment=optimal_treatment,
                    optimal_reward=optimal_reward,
                    selected_treatment=selected_treatment,
                    posterior_mean_best=posterior_mean_best,
                    runner_up_name=runner_up_name,
                    runner_up_wr=runner_up_wr,
                    running_estimates=running_estimates,
                    treatment_counts=treatment_counts,
                )
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
                            simulation_id, i + 1, db_flush_failures,
                            MAX_DB_FLUSH_FAILURES, e,
                        )
                        if db_flush_failures >= MAX_DB_FLUSH_FAILURES:
                            raise RuntimeError(
                                f"Simulation aborted: {db_flush_failures} "
                                f"consecutive DB flush failures. "
                                f"Last error: {e}"
                            )
                        # Keep buffer intact so next flush retries these steps

        # ── Flush remaining steps (with retries) ──
        if step_buffer:
            last_error = None
            for attempt in range(1, MAX_DB_FLUSH_RETRIES + 1):
                try:
                    await _flush_to_db(simulation_id, step_buffer, n_patients)
                    step_buffer.clear()
                    last_error = None
                    break
                except Exception as e:
                    last_error = e
                    logger.error(
                        "Final DB flush attempt %d/%d failed for %s: %s",
                        attempt, MAX_DB_FLUSH_RETRIES, simulation_id, e,
                    )
                    if attempt < MAX_DB_FLUSH_RETRIES:
                        await asyncio.sleep(2 ** attempt)  # exponential backoff
            if last_error is not None:
                lost_start = n_patients - len(step_buffer) + 1
                logger.error(
                    "Lost steps %d-%d for simulation %s after %d retries",
                    lost_start, n_patients, simulation_id, MAX_DB_FLUSH_RETRIES,
                )
                raise RuntimeError(
                    f"Final DB flush failed after {MAX_DB_FLUSH_RETRIES} "
                    f"retries: {last_error}"
                )

        # ── Final aggregates ──
        if final_event is None:
            raise RuntimeError("Simulation finished with zero steps")

        final = {
            "final_accuracy": round(float(final_event.runningAccuracy), 4),
            "final_cumulative_reward": round(float(final_event.cumulativeReward), 4),
            "final_cumulative_regret": round(float(final_event.cumulativeRegret), 4),
            "mean_reward": round(float(final_event.cumulativeReward) / n_patients, 4),
            "mean_regret": round(float(final_event.cumulativeRegret) / n_patients, 4),
            "thompson_exploration_rate": round(thompson_explore_count / n_patients, 4),
            "treatment_counts": {
                t: int(final_event.nUpdatesPerArm.get(t, 0)) for t in TREATMENTS
            },
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
            simulation_id,
            final_event.runningAccuracy,
            final_event.cumulativeReward,
            final_event.cumulativeRegret,
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
