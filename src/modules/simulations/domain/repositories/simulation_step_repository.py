from typing import Sequence
from uuid import UUID

from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from src.shared.database import BaseRepository
from src.modules.simulations.domain.models.simulation_step import SimulationStep


class SimulationStepRepository(BaseRepository[SimulationStep]):
    def __init__(self, session: AsyncSession):
        super().__init__(SimulationStep, session)

    async def get_by_simulation(
        self,
        simulation_id: UUID,
        skip: int = 0,
        limit: int = 100,
    ) -> Sequence[SimulationStep]:
        stmt = (
            select(SimulationStep)
            .where(SimulationStep.simulation_id == simulation_id)
            .order_by(SimulationStep.step_number.asc())
            .offset(skip)
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_steps_from(
        self,
        simulation_id: UUID,
        from_step: int,
        limit: int = 100,
    ) -> Sequence[SimulationStep]:
        """Get steps starting from a given step number (for reconnection replay)."""
        stmt = (
            select(SimulationStep)
            .where(
                SimulationStep.simulation_id == simulation_id,
                SimulationStep.step_number >= from_step,
            )
            .order_by(SimulationStep.step_number.asc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def count_by_simulation(self, simulation_id: UUID) -> int:
        return await self.count(simulation_id=simulation_id)

    async def delete_by_simulation(self, simulation_id: UUID) -> None:
        stmt = delete(SimulationStep).where(
            SimulationStep.simulation_id == simulation_id
        )
        await self.session.execute(stmt)
        await self.session.flush()

    async def create_batch(self, steps: Sequence[SimulationStep]) -> None:
        """Bulk insert steps (used by runner to flush in batches)."""
        self.session.add_all(steps)
        await self.session.flush()