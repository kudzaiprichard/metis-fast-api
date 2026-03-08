from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.shared.database import BaseRepository
from src.modules.patients.domain.models.medical_record import MedicalRecord


class MedicalRecordRepository(BaseRepository[MedicalRecord]):
    def __init__(self, session: AsyncSession):
        super().__init__(MedicalRecord, session)

    async def get_by_patient(
        self, patient_id: UUID, skip: int = 0, limit: int = 50
    ):
        stmt = (
            select(MedicalRecord)
            .where(MedicalRecord.patient_id == patient_id)
            .order_by(MedicalRecord.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_latest_for_patient(self, patient_id: UUID) -> MedicalRecord | None:
        stmt = (
            select(MedicalRecord)
            .where(MedicalRecord.patient_id == patient_id)
            .order_by(MedicalRecord.created_at.desc())
            .limit(1)
        )
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def count_by_patient(self, patient_id: UUID) -> int:
        return await self.count(patient_id=patient_id)