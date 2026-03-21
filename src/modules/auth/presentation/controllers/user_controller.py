from uuid import UUID
from typing import Optional

from fastapi import APIRouter, Depends, Query

from src.shared.responses import ApiResponse, PaginatedResponse
from src.shared.database.pagination import PaginationParams, get_pagination
from src.modules.auth.domain.models.user import User
from src.modules.auth.domain.services.user_management_service import UserManagementService
from src.modules.auth.presentation.dependencies import (
    get_user_management_service,
    require_admin,
)
from src.modules.auth.presentation.dtos.requests import CreateUserRequest, UpdateUserRequest
from src.modules.auth.presentation.dtos.responses import UserResponse

router = APIRouter(dependencies=[Depends(require_admin)])


@router.get("")
async def get_users(
    pagination: PaginationParams = Depends(get_pagination),
    role: Optional[str] = Query(None, pattern="^(ADMIN|DOCTOR)$"),
    is_active: Optional[bool] = None,
    service: UserManagementService = Depends(get_user_management_service),
):
    users, total = await service.get_users(
        page=pagination.page, page_size=pagination.page_size, role=role, is_active=is_active
    )
    return PaginatedResponse.ok(
        value=[UserResponse.from_user(u) for u in users],
        page=pagination.page,
        total=total,
        page_size=pagination.page_size,
    )


@router.get("/{user_id}")
async def get_user(
    user_id: UUID,
    _admin: User = Depends(require_admin),
    service: UserManagementService = Depends(get_user_management_service),
):
    user = await service.get_user(user_id)
    return ApiResponse.ok(value=UserResponse.from_user(user))


@router.post("", status_code=201)
async def create_user(
    body: CreateUserRequest,
    _admin: User = Depends(require_admin),
    service: UserManagementService = Depends(get_user_management_service),
):
    user = await service.create_user(
        email=body.email,
        username=body.username,
        first_name=body.first_name,
        last_name=body.last_name,
        password=body.password,
        role=body.role,
    )
    return ApiResponse.ok(value=UserResponse.from_user(user), message="User created")


@router.patch("/{user_id}")
async def update_user(
    user_id: UUID,
    body: UpdateUserRequest,
    service: UserManagementService = Depends(get_user_management_service),
):
    user = await service.update_user(
        user_id=user_id,
        first_name=body.first_name,
        last_name=body.last_name,
        username=body.username,
        role=body.role,
        is_active=body.is_active,
    )
    return ApiResponse.ok(value=UserResponse.from_user(user), message="User updated")


@router.delete("/{user_id}", status_code=200)
async def delete_user(
    user_id: UUID,
    service: UserManagementService = Depends(get_user_management_service),
):
    await service.delete_user(user_id)
    return ApiResponse.ok(value=None, message="User deleted")