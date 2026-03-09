from fastapi import FastAPI
from src.configs import application
from src.core.lifespan import lifespan
from src.core.middleware import register_middleware
from src.shared.exceptions.error_handlers import register_error_handlers


def create_app() -> FastAPI:
    app = FastAPI(
        title=application.name,
        version=application.version,
        debug=application.debug,
        lifespan=lifespan,
    )

    register_middleware(app)
    register_error_handlers(app)
    _register_routers(app)

    return app


def _register_routers(app: FastAPI) -> None:
    from src.modules.auth import auth_router, user_router
    from src.modules.patients import patient_router
    from src.modules.models import inference_router
    from src.modules.predictions import prediction_router
    from src.modules.simulations import simulation_router

    app.include_router(auth_router, prefix="/api/v1/auth", tags=["Auth"])
    app.include_router(user_router, prefix="/api/v1/users", tags=["Users"])
    app.include_router(patient_router, prefix="/api/v1/patients", tags=["Patients"])
    app.include_router(inference_router, prefix="/api/v1/inference", tags=["Inference"])
    app.include_router(prediction_router, prefix="/api/v1/predictions", tags=["Predictions"])
    app.include_router(simulation_router, prefix="/api/v1/simulations", tags=["Simulations"])