from uuid import UUID

from fastapi import APIRouter, Depends, Query

from src.shared.responses import ApiResponse, PaginatedResponse
from src.shared.database.pagination import PaginationParams, get_pagination
from src.modules.auth.presentation.dependencies import get_current_user, require_role
from src.modules.auth.domain.models.enums import Role
from src.modules.auth.domain.models.user import User
from src.modules.patients.domain.services.patient_service import PatientService
from src.modules.patients.presentation.dependencies import get_patient_service
from src.modules.patients.presentation.dtos.requests import (
    CreatePatientRequest,
    UpdatePatientRequest,
    CreateMedicalRecordRequest,
)
from src.modules.patients.presentation.dtos.responses import (
    PatientResponse,
    PatientDetailResponse,
    MedicalRecordResponse,
)

router = APIRouter(dependencies=[Depends(require_role(Role.ADMIN, Role.DOCTOR))])


# ── Patient endpoints ──

@router.post("", status_code=201)
async def create_patient(
    body: CreatePatientRequest,
    service: PatientService = Depends(get_patient_service),
):
    patient = await service.create_patient(
        first_name=body.first_name,
        last_name=body.last_name,
        date_of_birth=body.date_of_birth,
        gender=body.gender,
        email=body.email,
        phone=body.phone,
        address=body.address,
    )
    return ApiResponse.ok(value=PatientResponse.from_patient(patient), message="Patient created")


@router.get("")
async def get_patients(
    pagination: PaginationParams = Depends(get_pagination),
    service: PatientService = Depends(get_patient_service),
):
    patients, total = await service.get_patients(page=pagination.page, page_size=pagination.page_size)
    return PaginatedResponse.ok(
        value=[PatientResponse.from_patient(p) for p in patients],
        page=pagination.page,
        total=total,
        page_size=pagination.page_size,
    )


@router.get("/{patient_id}")
async def get_patient(
    patient_id: UUID,
    service: PatientService = Depends(get_patient_service),
):
    patient = await service.get_patient(patient_id)
    return ApiResponse.ok(value=PatientDetailResponse.from_patient(patient))


@router.patch("/{patient_id}")
async def update_patient(
    patient_id: UUID,
    body: UpdatePatientRequest,
    service: PatientService = Depends(get_patient_service),
):
    patient = await service.update_patient(
        patient_id=patient_id,
        first_name=body.first_name,
        last_name=body.last_name,
        date_of_birth=body.date_of_birth,
        gender=body.gender,
        email=body.email,
        phone=body.phone,
        address=body.address,
    )
    return ApiResponse.ok(value=PatientResponse.from_patient(patient), message="Patient updated")


@router.delete("/{patient_id}", status_code=200)
async def delete_patient(
    patient_id: UUID,
    service: PatientService = Depends(get_patient_service),
):
    await service.delete_patient(patient_id)
    return ApiResponse.ok(value=None, message="Patient deleted")


# ── Medical Record endpoints ──

@router.post("/{patient_id}/medical-records", status_code=201)
async def add_medical_record(
    patient_id: UUID,
    body: CreateMedicalRecordRequest,
    service: PatientService = Depends(get_patient_service),
):
    record = await service.add_medical_record(
        patient_id=patient_id,
        age=body.age,
        bmi=body.bmi,
        hba1c_baseline=body.hba1c_baseline,
        egfr=body.egfr,
        diabetes_duration=body.diabetes_duration,
        fasting_glucose=body.fasting_glucose,
        c_peptide=body.c_peptide,
        cvd=body.cvd,
        ckd=body.ckd,
        nafld=body.nafld,
        hypertension=body.hypertension,
        bp_systolic=body.bp_systolic,
        ldl=body.ldl,
        hdl=body.hdl,
        triglycerides=body.triglycerides,
        alt=body.alt,
        notes=body.notes,
    )
    return ApiResponse.ok(value=MedicalRecordResponse.from_record(record), message="Medical record added")


@router.get("/{patient_id}/medical-records")
async def get_medical_records(
    patient_id: UUID,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    service: PatientService = Depends(get_patient_service),
):
    records = await service.get_medical_records(patient_id, skip, limit)
    return ApiResponse.ok(value=[MedicalRecordResponse.from_record(r) for r in records])


@router.get("/{patient_id}/medical-records/{record_id}")
async def get_medical_record(
    patient_id: UUID,
    record_id: UUID,
    service: PatientService = Depends(get_patient_service),
):
    record = await service.get_medical_record(patient_id, record_id)
    return ApiResponse.ok(value=MedicalRecordResponse.from_record(record))