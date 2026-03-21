"""
Similar Patient API controller.
Endpoints for finding and viewing similar patient cases from Neo4j.
"""

from fastapi import APIRouter, Depends

from src.shared.responses import ApiResponse, PaginatedResponse
from src.shared.database.pagination import PaginationParams, get_pagination
from src.modules.auth.presentation.dependencies import require_role
from src.modules.auth.domain.models.enums import Role
from src.modules.patients.domain.services.similar_patient_service import SimilarPatientService
from src.modules.patients.presentation.dependencies import get_similar_patient_service
from src.modules.patients.presentation.dtos.requests import (
    FindSimilarPatientsRequest,
    FindSimilarPatientsGraphRequest,
)
from src.modules.patients.presentation.dtos.responses import (
    SimilarPatientCaseResponse,
    SimilarPatientsGraphResponse,
    SimilarPatientDetailResponse,
)

router = APIRouter(dependencies=[Depends(require_role(Role.DOCTOR))])


@router.post("/search", status_code=200)
async def find_similar_patients(
    body: FindSimilarPatientsRequest,
    pagination: PaginationParams = Depends(get_pagination),
    service: SimilarPatientService = Depends(get_similar_patient_service),
):
    """
    Find similar patient cases in tabular format.
    Search criteria in body, pagination via query params (?page=1&pageSize=10).
    """
    all_cases, patient_id = await service.find_similar_patients(
        patient_id=body.patient_id,
        medical_record_id=body.medical_record_id,
        limit=body.limit,
        treatment_filter=body.treatment_filter,
        min_similarity=body.min_similarity,
    )

    total = len(all_cases)
    start = pagination.skip
    end = start + pagination.page_size
    page_cases = all_cases[start:end]

    return PaginatedResponse.ok(
        value=[SimilarPatientCaseResponse.from_dict(c) for c in page_cases],
        page=pagination.page,
        total=total,
        page_size=pagination.page_size,
        message="Similar patients retrieved successfully",
    )


@router.post("/search/graph", status_code=200)
async def find_similar_patients_graph(
    body: FindSimilarPatientsGraphRequest,
    service: SimilarPatientService = Depends(get_similar_patient_service),
):
    """
    Find similar patient cases in graph format for visualization.
    """
    graph_data = await service.find_similar_patients_graph(
        patient_id=body.patient_id,
        medical_record_id=body.medical_record_id,
        limit=body.limit,
        treatment_filter=body.treatment_filter,
    )
    return ApiResponse.ok(
        value=SimilarPatientsGraphResponse.from_dict(graph_data),
        message="Similar patients graph retrieved successfully",
    )


@router.get("/{case_id}", status_code=200)
async def get_similar_patient_detail(
    case_id: str,
    service: SimilarPatientService = Depends(get_similar_patient_service),
):
    """
    Get complete details of a similar patient case from the Neo4j
    historical dataset.
    """
    patient_data = await service.get_similar_patient_detail(case_id)
    return ApiResponse.ok(
        value=SimilarPatientDetailResponse.from_dict(patient_data),
        message="Patient case details retrieved successfully",
    )