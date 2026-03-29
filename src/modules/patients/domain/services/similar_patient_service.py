"""
Similar patient search service.

Finds clinically similar patient cases from Neo4j using normalised
feature matching and comorbidity overlap scoring.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from uuid import UUID

from src.modules.patients.domain.repositories.patient_repository import PatientRepository
from src.modules.patients.domain.repositories.medical_record_repository import MedicalRecordRepository
from src.modules.patients.domain.models.medical_record import MedicalRecord
from src.shared.neo4j.neo4j_graph_database import Neo4jGraphDatabase
from src.shared.exceptions import (
    NotFoundException,
    BadRequestException,
    ServiceUnavailableException,
)
from src.shared.responses import ErrorDetail

logger = logging.getLogger(__name__)


class SimilarPatientService:
    """
    Async service wrapping sync Neo4j calls via asyncio.to_thread.
    Injected with repos + Neo4j instance through constructor.
    """

    def __init__(
        self,
        patient_repo: PatientRepository,
        medical_record_repo: MedicalRecordRepository,
        neo4j_db: Neo4jGraphDatabase,
    ):
        self.patient_repo = patient_repo
        self.record_repo = medical_record_repo
        self.neo4j_db = neo4j_db

    # ─── Public methods ───

    async def find_similar_patients(
        self,
        patient_id: Optional[UUID] = None,
        medical_record_id: Optional[UUID] = None,
        limit: int = 20,
        treatment_filter: Optional[List[str]] = None,
        min_similarity: float = 0.5,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Find similar patients from Neo4j.

        Returns:
            Tuple of (all matching cases as dicts, resolved patient_id as string).
            Pagination is handled by the controller.
        """
        record, resolved_patient_id = await self._resolve_medical_record(
            patient_id, medical_record_id
        )
        self._check_neo4j()
        profile = self._build_profile(record)

        try:
            similar_cases = await asyncio.to_thread(
                self.neo4j_db.find_similar_patients,
                patient_profile=profile,
                limit=limit,
                treatment_filter=treatment_filter,
                min_similarity=min_similarity,
            )
        except Exception as e:
            logger.error("Error finding similar patients: %s", e)
            raise ServiceUnavailableException(
                message="Similar patient search failed. Please try again later",
                error_detail=ErrorDetail(
                    title="Search Failed",
                    code="SIMILAR_PATIENTS_ERROR",
                    status=503,
                    details=["Failed to search for similar patients"],
                ),
            )

        return similar_cases, str(resolved_patient_id)

    async def find_similar_patients_graph(
        self,
        patient_id: Optional[UUID] = None,
        medical_record_id: Optional[UUID] = None,
        limit: int = 5,
        treatment_filter: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Find similar patients in graph format for visualization.

        Returns:
            Dict with patient_id, nodes, edges, metadata.
        """
        record, resolved_patient_id = await self._resolve_medical_record(
            patient_id, medical_record_id
        )
        self._check_neo4j()
        # Fetch the patient so the graph metadata can ship gender alongside
        # the numeric vitals. Gender lives on Patient (not MedicalRecord);
        # the graph view's context strip uses it to render the "49F" pill.
        # If the patient row went missing (e.g. record points at a deleted
        # patient), fail loudly rather than silently shipping a profile
        # without gender.
        patient = await self.patient_repo.get_by_id(resolved_patient_id)
        if not patient:
            raise NotFoundException(
                message="The patient linked to this record no longer exists",
                error_detail=ErrorDetail(
                    title="Patient Not Found",
                    code="PATIENT_NOT_FOUND",
                    status=404,
                    details=[f"Patient with ID {resolved_patient_id} does not exist"],
                ),
            )
        profile = self._build_profile(record, patient_gender=patient.gender)

        try:
            graph_data = await asyncio.to_thread(
                self.neo4j_db.find_similar_cases_graph,
                patient_profile=profile,
                limit=limit,
                treatment_filter=treatment_filter,
            )
        except Exception as e:
            logger.error("Error finding similar patients (graph): %s", e)
            raise ServiceUnavailableException(
                message="Similar patient graph generation failed. Please try again later",
                error_detail=ErrorDetail(
                    title="Search Failed",
                    code="SIMILAR_PATIENTS_GRAPH_ERROR",
                    status=503,
                    details=["Failed to generate graph for similar patients"],
                ),
            )

        return {"patient_id": str(resolved_patient_id), **graph_data}

    async def get_similar_patient_detail(self, case_id: str) -> Dict[str, Any]:
        """
        Get complete details of a similar patient case from Neo4j.

        Returns:
            Dict with patient details, demographics, clinical features, etc.
        """
        self._check_neo4j()

        try:
            patient_data = await asyncio.to_thread(
                self.neo4j_db.get_patient_by_id, case_id
            )
        except Exception as e:
            logger.error("Error retrieving patient case %s: %s", case_id, e)
            raise ServiceUnavailableException(
                message="Patient case lookup failed. Please try again later",
                error_detail=ErrorDetail(
                    title="Lookup Failed",
                    code="PATIENT_CASE_LOOKUP_ERROR",
                    status=503,
                    details=["Failed to retrieve patient case details"],
                ),
            )

        if not patient_data:
            raise NotFoundException(
                message="The patient case you're looking for doesn't exist",
                error_detail=ErrorDetail(
                    title="Patient Case Not Found",
                    code="CASE_NOT_FOUND",
                    status=404,
                    details=[f"Patient case with ID {case_id} does not exist in historical dataset"],
                ),
            )

        return patient_data

    # ─── Private helpers ───

    async def _resolve_medical_record(
        self,
        patient_id: Optional[UUID],
        medical_record_id: Optional[UUID],
    ) -> Tuple[MedicalRecord, UUID]:
        """
        Resolve medical record from either medical_record_id or patient_id.
        medical_record_id takes priority when both are provided.
        """
        if medical_record_id:
            record = await self.record_repo.get_by_id(medical_record_id)
            if not record:
                raise NotFoundException(
                    message="The medical record you're searching with doesn't exist",
                    error_detail=ErrorDetail(
                        title="Medical Record Not Found",
                        code="RECORD_NOT_FOUND",
                        status=404,
                        details=[f"Medical record with ID {medical_record_id} does not exist"],
                    ),
                )
            return record, record.patient_id

        patient = await self.patient_repo.get_by_id(patient_id)
        if not patient:
            raise NotFoundException(
                message="The patient you're searching for doesn't exist",
                error_detail=ErrorDetail(
                    title="Patient Not Found",
                    code="PATIENT_NOT_FOUND",
                    status=404,
                    details=[f"Patient with ID {patient_id} does not exist"],
                ),
            )

        record = await self.record_repo.get_latest_for_patient(patient_id)
        if not record:
            raise BadRequestException(
                message="Cannot find similar cases without patient medical data",
                error_detail=ErrorDetail(
                    title="Medical Data Missing",
                    code="NO_MEDICAL_DATA",
                    status=400,
                    details=["Patient must have medical data to find similar cases"],
                ),
            )

        return record, patient_id

    def _check_neo4j(self) -> None:
        if not self.neo4j_db.is_connected():
            raise ServiceUnavailableException(
                message="Similar patient search is temporarily unavailable",
                error_detail=ErrorDetail(
                    title="Service Unavailable",
                    code="NEO4J_UNAVAILABLE",
                    status=503,
                    details=["Graph database is currently unavailable"],
                ),
            )

    @staticmethod
    def _build_profile(
        record: MedicalRecord,
        patient_gender: Optional[str] = None,
    ) -> Dict[str, Any]:
        profile: Dict[str, Any] = {
            "age": record.age,
            "hba1c_baseline": float(record.hba1c_baseline),
            "diabetes_duration": float(record.diabetes_duration),
            "fasting_glucose": float(record.fasting_glucose),
            "c_peptide": float(record.c_peptide),
            "egfr": float(record.egfr),
            "bmi": float(record.bmi),
            "bp_systolic": int(record.bp_systolic),
            "ldl": float(record.ldl),
            "hdl": float(record.hdl),
            "triglycerides": float(record.triglycerides),
            "alt": float(record.alt),
            "cvd": record.cvd,
            "ckd": record.ckd,
            "nafld": record.nafld,
            "hypertension": record.hypertension,
        }
        # `gender` is only attached by the graph endpoint (the tabular flow
        # exposes per-result gender on each row already). When the graph
        # endpoint passes `patient_gender`, attach it; absence means the
        # caller deliberately didn't request gender enrichment.
        if patient_gender is not None:
            profile["gender"] = patient_gender
        return profile