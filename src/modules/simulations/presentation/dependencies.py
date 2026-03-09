from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.shared.database import get_db
from src.modules.auth.presentation.dependencies import get_current_user, require_role
from src.modules.auth.domain.models.enums import Role
from src.modules.simulations.domain.repositories.simulation_repository import SimulationRepository
from src.modules.simulations.domain.repositories.simulation_step_repository import SimulationStepRepository
from src.modules.simulations.domain.services.simulation_service import SimulationService


require_admin = require_role(Role.ADMIN)


async def get_simulation_service(
    session: AsyncSession = Depends(get_db),
) -> SimulationService:
    simulation_repo = SimulationRepository(session)
    step_repo = SimulationStepRepository(session)
    return SimulationService(simulation_repo, step_repo)