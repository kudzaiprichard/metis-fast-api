from src.modules.patients.presentation.controllers.patient_controller import router as patient_router
from src.modules.patients.presentation.controllers.similar_patients_controller import router as similar_patients_router

__all__ = ["patient_router", "similar_patients_router"]