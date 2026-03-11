import os
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.shared.database.engine import engine
from src.shared.neo4j import connect, close as close_neo4j
from src.configs import neo4j as neo4j_config
from src.configs import logging as log_config, model as model_config


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


def _load_models() -> None:
    from src.modules.models.internal.model_loader import registry

    registry.load(
        name="default",
        model_file=model_config.model_file,
        pipeline_file=model_config.pipeline_file,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──
    _setup_logging()
    logger.info("Starting up — logging and DB pool initialised")

    try:
        _load_models()
        logger.info("ML models loaded into memory")
    except Exception as e:
        logger.error("Failed to load ML models: %s", e)

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

    from src.modules.auth.internal.token_cleanup import start_token_cleanup
    cleanup_task = asyncio.create_task(start_token_cleanup())

    yield

    # ── Shutdown ──
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

    close_neo4j()
    logger.info("Neo4j connection closed")

    await engine.dispose()
    logger.info("Shutting down — DB pool disposed")