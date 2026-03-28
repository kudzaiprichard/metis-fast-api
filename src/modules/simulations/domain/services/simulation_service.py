import asyncio
import logging
from typing import Sequence, Tuple
from uuid import UUID

from src.modules.simulations.domain.models.simulation import Simulation
from src.modules.simulations.domain.models.simulation_step import SimulationStep
from src.modules.simulations.domain.models.enums import SimulationStatus
from src.modules.simulations.domain.repositories.simulation_repository import SimulationRepository
from src.modules.simulations.domain.repositories.simulation_step_repository import SimulationStepRepository
from src.shared.exceptions import NotFoundException, ConflictException
from src.shared.responses import ErrorDetail

logger = logging.getLogger(__name__)

MAX_CONCURRENT_SIMULATIONS = 3

# Process-level lock to prevent TOCTOU race on the concurrency check.
# Sufficient for single-worker deployments; for multi-worker, use a
# database advisory lock instead.
_create_simulation_lock = asyncio.Lock()


class SimulationService:
    def __init__(
        self,
        simulation_repo: SimulationRepository,
        step_repo: SimulationStepRepository,
    ):
        self.simulation_repo = simulation_repo
        self.step_repo = step_repo

    # ── Create ──

    async def create_simulation(
        self,
        dataset_filename: str,
        dataset_row_count: int,
        initial_epsilon: float = 0.3,
        epsilon_decay: float = 0.997,
        min_epsilon: float = 0.01,
        random_seed: int = 42,
        reset_posterior: bool = True,
    ) -> Simulation:
        async with _create_simulation_lock:
            running_count = await self.simulation_repo.count_running()
            if running_count >= MAX_CONCURRENT_SIMULATIONS:
                raise ConflictException(
                    message=f"Maximum of {MAX_CONCURRENT_SIMULATIONS} concurrent simulations reached",
                    error_detail=ErrorDetail(
                        title="Concurrency Limit",
                        code="MAX_SIMULATIONS_REACHED",
                        status=409,
                        details=[f"{running_count} simulations are currently running"],
                    ),
                )

            simulation = Simulation(
                initial_epsilon=initial_epsilon,
                epsilon_decay=epsilon_decay,
                min_epsilon=min_epsilon,
                random_seed=random_seed,
                reset_posterior=reset_posterior,
                dataset_filename=dataset_filename,
                dataset_row_count=dataset_row_count,
                status=SimulationStatus.PENDING,
                current_step=0,
            )

            simulation = await self.simulation_repo.create(simulation)
            logger.info(
                "Simulation created: %s (file=%s, rows=%d)",
                simulation.id, dataset_filename, dataset_row_count,
            )
            return simulation

    # ── Read ──

    async def get_simulation(self, simulation_id: UUID) -> Simulation:
        simulation = await self.simulation_repo.get_by_id(simulation_id)
        if not simulation:
            raise NotFoundException(
                message="Simulation not found",
                error_detail=ErrorDetail(
                    title="Not Found",
                    code="SIMULATION_NOT_FOUND",
                    status=404,
                    details=[f"No simulation found with id {simulation_id}"],
                ),
            )
        return simulation

    async def get_simulations(
        self,
        page: int = 1,
        page_size: int = 20,
    ) -> Tuple[Sequence[Simulation], int]:
        return await self.simulation_repo.paginate(
            page=page,
            page_size=page_size,
            order_by="created_at",
            descending=True,
        )

    async def get_simulation_steps(
        self,
        simulation_id: UUID,
        skip: int = 0,
        limit: int = 100,
    ) -> Sequence[SimulationStep]:
        await self.get_simulation(simulation_id)
        return await self.step_repo.get_by_simulation(simulation_id, skip, limit)

    async def get_simulation_steps_paginated(
        self,
        simulation_id: UUID,
        page: int = 1,
        page_size: int = 20,
    ) -> Tuple[Sequence[SimulationStep], int]:
        await self.get_simulation(simulation_id)
        total = await self.step_repo.count_by_simulation(simulation_id)
        skip = (page - 1) * page_size
        steps = await self.step_repo.get_by_simulation(simulation_id, skip, page_size)
        return steps, total

    async def get_steps_from(
        self,
        simulation_id: UUID,
        from_step: int,
        limit: int = 100,
    ) -> Sequence[SimulationStep]:
        """Get steps from a given step number (for SSE reconnection)."""
        await self.get_simulation(simulation_id)
        return await self.step_repo.get_steps_from(simulation_id, from_step, limit)

    # ── Status updates (called by runner) ──

    async def mark_running(self, simulation_id: UUID) -> None:
        await self.simulation_repo.update_status(simulation_id, SimulationStatus.RUNNING)
        logger.info("Simulation %s → RUNNING", simulation_id)

    async def mark_completed(self, simulation_id: UUID, aggregates: dict) -> None:
        await self.simulation_repo.save_final_aggregates(simulation_id, aggregates)
        logger.info("Simulation %s → COMPLETED", simulation_id)

    async def mark_failed(self, simulation_id: UUID, error: str) -> None:
        await self.simulation_repo.update_status(
            simulation_id, SimulationStatus.FAILED, error_message=error
        )
        logger.error("Simulation %s → FAILED: %s", simulation_id, error)

    async def mark_cancelled(self, simulation_id: UUID) -> None:
        """
        Mark a simulation as cancelled in the database.
        Called as a fallback when the simulation is no longer in the
        in-memory registry (e.g. it finished between the status check
        and the cancel request).
        """
        simulation = await self.get_simulation(simulation_id)

        if simulation.status not in (SimulationStatus.RUNNING, SimulationStatus.PENDING):
            raise ConflictException(
                message="Only running or pending simulations can be cancelled",
                error_detail=ErrorDetail(
                    title="Conflict",
                    code="SIMULATION_NOT_CANCELLABLE",
                    status=409,
                    details=[f"Simulation status is '{simulation.status.value}'"],
                ),
            )

        await self.simulation_repo.update_status(simulation_id, SimulationStatus.CANCELLED)
        logger.info("Simulation %s → CANCELLED", simulation_id)

    async def update_progress(self, simulation_id: UUID, current_step: int) -> None:
        await self.simulation_repo.update_progress(simulation_id, current_step)

    # ── Delete ──

    async def delete_simulation(self, simulation_id: UUID) -> None:
        simulation = await self.get_simulation(simulation_id)

        if simulation.status == SimulationStatus.RUNNING:
            raise ConflictException(
                message="Cannot delete a running simulation",
                error_detail=ErrorDetail(
                    title="Conflict",
                    code="SIMULATION_RUNNING",
                    status=409,
                    details=["Cancel the simulation before deleting"],
                ),
            )

        await self.simulation_repo.delete(simulation)
        logger.info("Simulation %s deleted", simulation_id)