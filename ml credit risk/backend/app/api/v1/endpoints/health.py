from fastapi import APIRouter
from backend.app.core.config import settings
from backend.app.schemas.loan import HealthCheckResponse
from backend.app.services.model_service import model_service

router = APIRouter()

@router.get("/health", response_model=HealthCheckResponse, summary="System Health & Diagnostic Status")
def get_health() -> HealthCheckResponse:
    """Returns the operational status of the service, model availability, and metadata."""
    return HealthCheckResponse(
        status="Online",
        version=settings.VERSION,
        model_loaded=model_service.is_loaded,
        features_count=len(model_service.feature_names) if model_service.is_loaded else 0
    )
