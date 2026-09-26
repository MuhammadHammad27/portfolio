from fastapi import APIRouter
from backend.app.api.v1.endpoints.health import router as health_router
from backend.app.api.v1.endpoints.predict import router as predict_router

api_router = APIRouter()

api_router.include_router(health_router, tags=["System Health"])
api_router.include_router(predict_router, tags=["Credit Risk Prediction"])
