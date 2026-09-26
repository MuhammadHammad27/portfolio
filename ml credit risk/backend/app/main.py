import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.core.config import settings
from backend.app.api.router import api_router
from backend.app.api.v1.endpoints.health import router as health_router
from backend.app.api.v1.endpoints.predict import router as predict_router
from backend.app.services.model_service import model_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("credit_risk_api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager that handles startup resource initialization and shutdown."""
    logger.info("Starting Credit Risk Assessment Service...")
    loaded = model_service.load_model()
    if loaded:
        logger.info("Model loaded and validated successfully.")
    else:
        logger.warning(
            "Model artifact could not be pre-loaded at startup. "
            "Please check if 'model/loan_model.pkl' is present."
        )
    yield
    logger.info("Shutting down Credit Risk Assessment Service...")

# FastAPI Application Factory
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Production-grade REST API for calibrated loan default probability prediction and SHAP explainability.",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Process timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    response.headers["X-Process-Time"] = f"{process_time * 1000:.2f}ms"
    return response

# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled error processing request: {request.url.path} - {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred. Please check server logs."}
    )

# Root Endpoint
@app.get("/", tags=["Root"])
def root():
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "Online",
        "documentation": "/docs",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "v1_health": "/api/v1/health",
            "v1_predict": "/api/v1/predict"
        }
    }

# Include API Routers (Both Versioned /api/v1 and Root /predict, /health for backward compatibility)
app.include_router(api_router, prefix="/api")
app.include_router(health_router, tags=["Health"])
app.include_router(predict_router, tags=["Predict"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
