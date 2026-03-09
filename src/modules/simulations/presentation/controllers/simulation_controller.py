import asyncio
import json
import logging

from fastapi import APIRouter, Depends, UploadFile, File, Form, Query
from sse_starlette.sse import EventSourceResponse

from src.shared.responses import ApiResponse, PaginatedResponse, ErrorDetail
from src.shared.exceptions import BadRequestException
from src.modules.auth.presentation.dependencies import get_current_user
from src.modules.auth.domain.models.user import User
from src.modules.simulations.presentation.dependencies import (
    require_admin,
    get_simulation_service,
)
from src.modules.simulations.domain.services.simulation_service import SimulationService
from src.modules.simulations.presentation.dtos.responses import (
    SimulationResponse,
    SimulationStepResponse,
)
from src.modules.simulations.internal.simulation_runner import (
    run_simulation,
    parse_and_validate_csv,
)
from src.modules.simulations.internal.stream_manager import (
    stream_manager,
    STREAM_COMPLETE,
    STREAM_ERROR,
)

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(require_admin)])


@router.post("")
async def start_simulation(
    file: UploadFile = File(..., description="CSV file with patient records (min 100 rows)"),
    initial_epsilon: float = Form(default=0.3, ge=0.0, le=1.0),
    epsilon_decay: float = Form(default=0.997, ge=0.9, le=1.0),
    min_epsilon: float = Form(default=0.01, ge=0.0, le=1.0),
    random_seed: int = Form(default=42, ge=0, le=999999),
    service: SimulationService = Depends(get_simulation_service),
    current_user: User = Depends(get_current_user),
):
    """
    Start a new bandit simulation.

    Accepts a CSV file with patient records (minimum 100 rows, 16 clinical features).
    Optionally accepts epsilon schedule and seed as form fields.
    Returns the simulation immediately, then runs the loop in a background task
    with SSE streaming.
    """
    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise BadRequestException(
            message="Only CSV files are accepted",
            error_detail=ErrorDetail(
                title="Invalid File",
                code="INVALID_FILE_TYPE",
                status=400,
                details=["Upload a .csv file containing patient records"],
            ),
        )

    # Read and validate CSV content
    try:
        content = await file.read()
        csv_content = content.decode("utf-8")
    except UnicodeDecodeError:
        raise BadRequestException(
            message="CSV file must be UTF-8 encoded",
            error_detail=ErrorDetail(
                title="Invalid Encoding",
                code="INVALID_ENCODING",
                status=400,
                details=["Ensure the CSV file is saved with UTF-8 encoding"],
            ),
        )

    try:
        patients = parse_and_validate_csv(csv_content)
    except ValueError as e:
        raise BadRequestException(
            message="CSV validation failed",
            error_detail=ErrorDetail(
                title="Validation Error",
                code="CSV_VALIDATION_FAILED",
                status=400,
                details=str(e).split("\n"),
            ),
        )

    # Create simulation record
    simulation = await service.create_simulation(
        dataset_filename=file.filename,
        dataset_row_count=len(patients),
        initial_epsilon=initial_epsilon,
        epsilon_decay=epsilon_decay,
        min_epsilon=min_epsilon,
        random_seed=random_seed,
        reset_posterior=reset_posterior,
    )

    # Launch background task
    asyncio.create_task(
        run_simulation(
            simulation_id=simulation.id,
            patients=patients,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            min_epsilon=min_epsilon,
            random_seed=random_seed,
            reset_posterior=reset_posterior,
        )
    )

    return ApiResponse.ok(
        value=SimulationResponse.from_entity(simulation),
        message=f"Simulation started with {len(patients)} patients",
    )


@router.get("/{simulation_id}/stream")
async def stream_simulation(
    simulation_id: str,
    last_step: int = Query(default=0, ge=0, description="Resume from this step (for reconnection)"),
    service: SimulationService = Depends(get_simulation_service),
    current_user: User = Depends(get_current_user),
):
    """
    SSE endpoint — streams simulation steps in real-time.

    If last_step > 0, replays stored steps from DB first (reconnection support),
    then switches to live stream.
    """
    from uuid import UUID as UUIDType

    sim_id = UUIDType(simulation_id)
    simulation = await service.get_simulation(sim_id)

    async def event_generator():
        # Replay stored steps if reconnecting
        if last_step > 0:
            stored_steps = await service.get_steps_from(sim_id, from_step=1, limit=last_step)
            for step in stored_steps:
                step_data = SimulationStepResponse.from_entity(step)
                yield {
                    "event": "step",
                    "data": step_data.model_dump_json(by_alias=True),
                }

        # Subscribe to live stream
        queue = await stream_manager.subscribe(sim_id)

        if queue is None:
            # Simulation not actively streaming — send all stored steps
            if last_step == 0:
                all_steps = await service.get_simulation_steps(sim_id, skip=0, limit=50000)
                for step in all_steps:
                    step_data = SimulationStepResponse.from_entity(step)
                    yield {
                        "event": "step",
                        "data": step_data.model_dump_json(by_alias=True),
                    }

            yield {"event": "complete", "data": json.dumps({"status": simulation.status.value})}
            return

        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield {"event": "ping", "data": ""}
                    continue

                if event == STREAM_COMPLETE:
                    yield {"event": "complete", "data": json.dumps({"status": "COMPLETED"})}
                    break

                if isinstance(event, dict) and event.get("type") == STREAM_ERROR:
                    yield {
                        "event": "error",
                        "data": json.dumps({"error": event["error"]}),
                    }
                    continue

                # Regular step event — skip if already replayed
                if isinstance(event, dict) and event.get("step", 0) <= last_step:
                    continue

                yield {
                    "event": "step",
                    "data": json.dumps(event),
                }
        finally:
            await stream_manager.unsubscribe(sim_id, queue)

    return EventSourceResponse(event_generator())


@router.get("")
async def list_simulations(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100, alias="pageSize"),
    service: SimulationService = Depends(get_simulation_service),
    current_user: User = Depends(get_current_user),
):
    """List all simulations with pagination."""
    simulations, total = await service.get_simulations(page=page, page_size=page_size)

    return PaginatedResponse.ok(
        value=[SimulationResponse.from_entity(s) for s in simulations],
        page=page,
        total=total,
        page_size=page_size,
        message=f"Found {total} simulations",
    )


@router.get("/{simulation_id}")
async def get_simulation(
    simulation_id: str,
    service: SimulationService = Depends(get_simulation_service),
    current_user: User = Depends(get_current_user),
):
    """Get simulation details with final aggregates."""
    from uuid import UUID as UUIDType

    simulation = await service.get_simulation(UUIDType(simulation_id))

    return ApiResponse.ok(
        value=SimulationResponse.from_entity(simulation),
        message="Simulation retrieved",
    )


@router.get("/{simulation_id}/steps")
async def get_simulation_steps(
    simulation_id: str,
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=1000),
    service: SimulationService = Depends(get_simulation_service),
    current_user: User = Depends(get_current_user),
):
    """Get stored simulation steps (for replay or analysis)."""
    from uuid import UUID as UUIDType

    steps = await service.get_simulation_steps(
        UUIDType(simulation_id), skip=skip, limit=limit
    )

    return ApiResponse.ok(
        value=[SimulationStepResponse.from_entity(s) for s in steps],
        message=f"Retrieved {len(steps)} steps",
    )


@router.delete("/{simulation_id}")
async def delete_simulation(
    simulation_id: str,
    service: SimulationService = Depends(get_simulation_service),
    current_user: User = Depends(get_current_user),
):
    """Delete a simulation and all its steps. Cannot delete running simulations."""
    from uuid import UUID as UUIDType

    await service.delete_simulation(UUIDType(simulation_id))

    return ApiResponse.ok(
        value=None,
        message="Simulation deleted",
    )