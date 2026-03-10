import asyncio
import json
import logging

from fastapi import APIRouter, Depends, UploadFile, File, Form, Query
from sse_starlette.sse import EventSourceResponse

from src.shared.responses import ApiResponse, PaginatedResponse, ErrorDetail
from src.shared.exceptions import BadRequestException, ConflictException
from src.modules.auth.presentation.dependencies import get_current_user
from src.modules.auth.domain.models.user import User
from src.modules.simulations.presentation.dependencies import (
    require_admin,
    get_simulation_service,
)
from src.modules.simulations.domain.services.simulation_service import SimulationService
from src.modules.simulations.domain.models.enums import SimulationStatus
from src.modules.simulations.presentation.dtos.responses import (
    SimulationResponse,
    SimulationStepResponse,
    SSEStepResponse,
)
from src.modules.simulations.internal.simulation_runner import (
    run_simulation,
    parse_and_validate_csv,
    simulation_registry,
)

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(require_admin)])

# Chunk size for DB replay in SSE stream
REPLAY_CHUNK_SIZE = 500


@router.post("")
async def start_simulation(
    file: UploadFile = File(..., description="CSV file with patient records (min 100 rows)"),
    initial_epsilon: float = Form(default=0.3, ge=0.0, le=1.0),
    epsilon_decay: float = Form(default=0.997, ge=0.9, le=1.0),
    min_epsilon: float = Form(default=0.01, ge=0.0, le=1.0),
    random_seed: int = Form(default=42, ge=0, le=999999),
    reset_posterior: bool = Form(default=True),
    service: SimulationService = Depends(get_simulation_service),
    current_user: User = Depends(get_current_user),
):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise BadRequestException(
            message="Only CSV files are accepted",
            error_detail=ErrorDetail(
                title="Invalid File", code="INVALID_FILE_TYPE", status=400,
                details=["Upload a .csv file containing patient records"],
            ),
        )

    try:
        content = await file.read()
        csv_content = content.decode("utf-8")
    except UnicodeDecodeError:
        raise BadRequestException(
            message="CSV file must be UTF-8 encoded",
            error_detail=ErrorDetail(
                title="Invalid Encoding", code="INVALID_ENCODING", status=400,
                details=["Ensure the CSV file is saved with UTF-8 encoding"],
            ),
        )

    try:
        patients = parse_and_validate_csv(csv_content)
    except ValueError as e:
        raise BadRequestException(
            message="CSV validation failed",
            error_detail=ErrorDetail(
                title="Validation Error", code="CSV_VALIDATION_FAILED", status=400,
                details=str(e).split("\n"),
            ),
        )

    simulation = await service.create_simulation(
        dataset_filename=file.filename,
        dataset_row_count=len(patients),
        initial_epsilon=initial_epsilon,
        epsilon_decay=epsilon_decay,
        min_epsilon=min_epsilon,
        random_seed=random_seed,
        reset_posterior=reset_posterior,
    )

    # Launch simulation as background task — runs in this process
    task = asyncio.create_task(
        run_simulation(
            simulation_id=simulation.id,
            patients=patients,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            min_epsilon=min_epsilon,
            random_seed=random_seed,
            reset_posterior=reset_posterior,
        ),
        name=f"simulation-{simulation.id}",
    )

    # Store task reference in registry so cancel endpoint can reach it
    simulation_registry.set_task(simulation.id, task)

    def _on_done(t: asyncio.Task) -> None:
        if t.cancelled():
            logger.info("Simulation task cancelled: %s", simulation.id)
        elif t.exception():
            logger.error("Simulation task failed: %s — %s", simulation.id, t.exception())

    task.add_done_callback(_on_done)

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

    If the simulation is still running, subscribes to the in-memory queue.
    If the simulation is completed, replays from the database in chunks.
    Supports reconnection via last_step parameter.

    Both paths use SSEStepResponse DTO for consistent output shape.
    """
    from uuid import UUID as UUIDType

    sim_id = UUIDType(simulation_id)
    simulation = await service.get_simulation(sim_id)

    async def event_generator():
        # Snapshot registry state atomically to avoid race condition where
        # the simulation completes between is_registered and subscribe calls.
        is_live = simulation_registry.is_registered(sim_id) and not simulation_registry.is_completed(sim_id)
        queue = simulation_registry.subscribe(sim_id) if is_live else None

        if queue is not None:
            # ── Live path: subscribe to in-memory queue ──
            logger.info("SSE client subscribed to live simulation %s", simulation_id)

            try:
                while True:
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    except asyncio.TimeoutError:
                        yield {"event": "ping", "data": ""}
                        continue

                    if event["type"] == "step":
                        step_json = event["data"]
                        step_num = json.loads(step_json).get("step", 0)
                        if step_num <= last_step:
                            continue
                        yield {"event": "step", "data": step_json}
                        await asyncio.sleep(0)

                    elif event["type"] == "complete":
                        yield {"event": "complete", "data": json.dumps(event["data"])}
                        return

                    elif event["type"] == "error":
                        yield {"event": "error", "data": json.dumps(event["data"])}
                        return

            finally:
                simulation_registry.unsubscribe(sim_id, queue)
                logger.info("SSE client unsubscribed from simulation %s", simulation_id)

        else:
            # ── Replay path: stream from DB in chunks ──
            # If we raced (sim completed between check and subscribe), we
            # fall through here safely and replay from the database.
            logger.info("Replaying simulation %s from DB (last_step=%d)", simulation_id, last_step)

            skip = last_step
            while True:
                chunk = await service.get_simulation_steps(sim_id, skip=skip, limit=REPLAY_CHUNK_SIZE)
                if not chunk:
                    break

                for step in chunk:
                    if step.step_number <= last_step:
                        continue
                    sse_step = SSEStepResponse.from_entity(step, simulation.dataset_row_count)
                    yield {"event": "step", "data": sse_step.model_dump_json(by_alias=True)}
                    await asyncio.sleep(0)

                # If we got fewer than the chunk size, we've reached the end
                if len(chunk) < REPLAY_CHUNK_SIZE:
                    break

                skip += len(chunk)

            # Fetch final status
            sim = await service.get_simulation(sim_id)
            yield {"event": "complete", "data": json.dumps({"status": sim.status.value})}

    return EventSourceResponse(event_generator())


@router.post("/{simulation_id}/cancel")
async def cancel_simulation(
    simulation_id: str,
    service: SimulationService = Depends(get_simulation_service),
    current_user: User = Depends(get_current_user),
):
    """
    Cancel a running simulation.

    Sets the cancellation flag in the registry so the run loop exits gracefully.
    Also cancels the asyncio.Task as a fallback.
    """
    from uuid import UUID as UUIDType

    sim_id = UUIDType(simulation_id)
    simulation = await service.get_simulation(sim_id)

    if simulation.status != SimulationStatus.RUNNING:
        raise ConflictException(
            message="Only running simulations can be cancelled",
            error_detail=ErrorDetail(
                title="Conflict",
                code="SIMULATION_NOT_RUNNING",
                status=409,
                details=[f"Simulation status is '{simulation.status.value}'"],
            ),
        )

    cancelled = simulation_registry.cancel(sim_id)
    if not cancelled:
        # Not in registry (maybe already finished) — mark in DB directly
        await service.mark_cancelled(sim_id)

    # Re-fetch to return updated state
    simulation = await service.get_simulation(sim_id)
    return ApiResponse.ok(
        value=SimulationResponse.from_entity(simulation),
        message="Simulation cancellation requested",
    )


@router.get("")
async def list_simulations(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100, alias="pageSize"),
    service: SimulationService = Depends(get_simulation_service),
    current_user: User = Depends(get_current_user),
):
    simulations, total = await service.get_simulations(page=page, page_size=page_size)
    return PaginatedResponse.ok(
        value=[SimulationResponse.from_entity(s) for s in simulations],
        page=page, total=total, page_size=page_size,
        message=f"Found {total} simulations",
    )


@router.get("/{simulation_id}")
async def get_simulation(
    simulation_id: str,
    service: SimulationService = Depends(get_simulation_service),
    current_user: User = Depends(get_current_user),
):
    from uuid import UUID as UUIDType
    simulation = await service.get_simulation(UUIDType(simulation_id))
    return ApiResponse.ok(value=SimulationResponse.from_entity(simulation), message="Simulation retrieved")


@router.get("/{simulation_id}/steps")
async def get_simulation_steps(
    simulation_id: str,
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=1000),
    service: SimulationService = Depends(get_simulation_service),
    current_user: User = Depends(get_current_user),
):
    from uuid import UUID as UUIDType
    steps = await service.get_simulation_steps(UUIDType(simulation_id), skip=skip, limit=limit)
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
    from uuid import UUID as UUIDType
    await service.delete_simulation(UUIDType(simulation_id))
    return ApiResponse.ok(value=None, message="Simulation deleted")