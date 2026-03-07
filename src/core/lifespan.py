import os
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.shared.database.engine import engine
from src.configs import logging as log_config


logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    """Configure logging from application.yaml settings."""
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    _setup_logging()
    logger.info("Starting up — logging and DB pool initialised")

    # Seed default admin
    from src.modules.auth.internal.admin_seeder import seed_admin
    try:
        await seed_admin()
    except Exception as e:
        logger.error("Admin seeding failed: %s", e)

    # Start background token cleanup
    from src.modules.auth.internal.token_cleanup import start_token_cleanup
    cleanup_task = asyncio.create_task(start_token_cleanup())

    yield

    # Cancel cleanup on shutdown
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    # Shutdown
    await engine.dispose()
    logger.info("Shutting down — DB pool disposed")