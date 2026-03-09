from typing import Sequence
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.shared.database import BaseRepository
from src.modules.simulations.domain.models.simulation import Simulation
from src.modules.simulations.domain.models.enums import SimulationStatus


class SimulationRepository(BaseRepository[Simulation]):
    def __init__(self, session: AsyncSession):
        super().__init__(Simulation, session)

    async def get_by_status(self, status: SimulationStatus) -> Sequence[Simulation]:
        stmt = (
            select(Simulation)
            .where(Simulation.status == status)
            .order_by(Simulation.created_at.desc())
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def count_running(self) -> int:
        return await self.count(status=SimulationStatus.RUNNING)

    async def update_status(
        self,
        simulation_id: UUID,
        status: SimulationStatus,
        error_message: str | None = None,
    ) -> None:
        values = {"status": status}
        if error_message is not None:
            values["error_message"] = error_message
        stmt = (
            update(Simulation)
            .where(Simulation.id == simulation_id)
            .values(**values)
        )
        await self.session.execute(stmt)
        await self.session.flush()

    async def update_progress(self, simulation_id: UUID, current_step: int) -> None:
        stmt = (
            update(Simulation)
            .where(Simulation.id == simulation_id)
            .values(current_step=current_step)
        )
        await self.session.execute(stmt)
        await self.session.flush()

    async def save_final_aggregates(self, simulation_id: UUID, aggregates: dict) -> None:
        stmt = (
            update(Simulation)
            .where(Simulation.id == simulation_id)
            .values(
                status=SimulationStatus.COMPLETED,
                final_accuracy=aggregates["final_accuracy"],
                final_cumulative_reward=aggregates["final_cumulative_reward"],
                final_cumulative_regret=aggregates["final_cumulative_regret"],
                mean_reward=aggregates["mean_reward"],
                mean_regret=aggregates["mean_regret"],
                thompson_exploration_rate=aggregates["thompson_exploration_rate"],
                treatment_counts=aggregates["treatment_counts"],
                confidence_distribution=aggregates["confidence_distribution"],
                safety_distribution=aggregates["safety_distribution"],
            )
        )
        await self.session.execute(stmt)
        await self.session.flush()