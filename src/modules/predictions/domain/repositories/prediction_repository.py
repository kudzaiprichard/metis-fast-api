from uuid import UUID
from typing import Sequence, Tuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.shared.database import BaseRepository
from src.modules.predictions.domain.models.prediction import Prediction


class PredictionRepository(BaseRepository[Prediction]):
    def __init__(self, session: AsyncSession):
        super().__init__(Prediction, session)

    async def get_by_patient(
        self, patient_id: UUID, skip: int = 0, limit: int = 50,
    ) -> Sequence[Prediction]:
        stmt = (
            select(Prediction)
            .where(Prediction.patient_id == patient_id)
            .order_by(Prediction.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_by_medical_record(self, medical_record_id: UUID) -> Sequence[Prediction]:
        stmt = (
            select(Prediction)
            .where(Prediction.medical_record_id == medical_record_id)
            .order_by(Prediction.created_at.desc())
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_by_doctor(
        self, doctor_id: UUID, skip: int = 0, limit: int = 50,
    ) -> Sequence[Prediction]:
        stmt = (
            select(Prediction)
            .where(Prediction.created_by == doctor_id)
            .order_by(Prediction.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_paginated_by_patient(
        self, patient_id: UUID, page: int = 1, page_size: int = 20,
    ) -> Tuple[Sequence[Prediction], int]:
        return await self.paginate(
            page=page, page_size=page_size,
            order_by="created_at", descending=True,
            patient_id=patient_id,
        )