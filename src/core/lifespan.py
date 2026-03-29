import os
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.shared.database.engine import engine
from src.shared.neo4j import connect, close as close_neo4j
from src.configs import neo4j as neo4j_config
from src.configs import logging as log_config


logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    log_dir = os.path.dirname(log_config.file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_level = getattr(logging, log_config.level.upper(), logging.INFO)

    logging.basicConfig(
        level=log_level,
        format=log_config.format,
        handlers=[
            logging.FileHandler(log_config.file_path),
            logging.StreamHandler(),
        ],
        force=True,
    )


def _build_engine(app: FastAPI) -> None:
    from src.shared.inference_bootstrap import build_inference_engine

    app.state.engine = build_inference_engine()


async def _recover_orphaned_simulations() -> None:
    """
    Mark any simulations left in RUNNING status as FAILED on startup.
    This handles the case where the server process restarted (crash, deploy,
    OOM kill) while simulations were running — their background tasks died
    but the DB records were never updated.
    """
    from sqlalchemy import update
    from src.modules.simulations.domain.models.simulation import Simulation
    from src.modules.simulations.domain.models.enums import SimulationStatus
    from src.shared.database import async_session

    async with async_session() as session:
        async with session.begin():
            stmt = (
                update(Simulation)
                .where(Simulation.status == SimulationStatus.RUNNING)
                .values(
                    status=SimulationStatus.FAILED,
                    error_message="Server restarted during simulation",
                )
            )
            result = await session.execute(stmt)
            count = result.rowcount
            if count:
                logger.warning("Recovered %d orphaned RUNNING simulation(s) → FAILED", count)


async def _periodic_registry_sweep() -> None:
    """Sweep stale registry entries every 10 minutes."""
    from src.modules.simulations.internal.simulation_runner import simulation_registry

    while True:
        try:
            await asyncio.sleep(600)  # 10 minutes
            swept = simulation_registry.sweep_stale()
            if swept:
                logger.info("Periodic sweep removed %d stale registry entries", swept)
        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("Error in periodic registry sweep")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──
    _setup_logging()
    logger.info("Starting up — logging and DB pool initialised")

    app.state.engine = None
    try:
        _build_engine(app)
        logger.info("InferenceEngine constructed (ready=%s)", app.state.engine.ready)
    except Exception as e:
        logger.error("InferenceEngine construction failed: %s", e)

    # Neo4j
    try:
        connect(neo4j_config.uri, neo4j_config.username, neo4j_config.password)
        logger.info("Neo4j connected")
    except Exception as e:
        logger.error("Neo4j connection failed: %s", e)

    from src.modules.auth.internal.admin_seeder import seed_admin
    try:
        await seed_admin()
    except Exception as e:
        logger.error("Admin seeding failed: %s", e)

    # Recover orphaned simulations from previous process lifecycle
    try:
        await _recover_orphaned_simulations()
    except Exception as e:
        logger.error("Orphaned simulation recovery failed: %s", e)

    from src.modules.auth.internal.token_cleanup import start_token_cleanup
    cleanup_task = asyncio.create_task(start_token_cleanup())
    sweep_task = asyncio.create_task(_periodic_registry_sweep())

    yield

    # ── Shutdown ──
    cleanup_task.cancel()
    sweep_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    try:
        await sweep_task
    except asyncio.CancelledError:
        pass

    close_neo4j()
    logger.info("Neo4j connection closed")

    await engine.dispose()
    logger.info("Shutting down — DB pool disposed")