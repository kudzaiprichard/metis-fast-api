from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.shared.database import BaseRepository
from src.modules.patients.domain.models.patient import Patient


class PatientRepository(BaseRepository[Patient]):
    def __init__(self, session: AsyncSession):
        super().__init__(Patient, session)

    async def get_by_email(self, email: str) -> Patient | None:
        stmt = select(Patient).where(Patient.email == email)
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def email_exists(self, email: str) -> bool:
        return await self.exists(email=email)

    async def search_by_name(self, name: str, skip: int = 0, limit: int = 20):
        pattern = f"%{name}%"
        stmt = (
            select(Patient)
            .where(
                Patient.first_name.ilike(pattern) | Patient.last_name.ilike(pattern)
            )
            .offset(skip)
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()