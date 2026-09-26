import logging
from fastapi import APIRouter, HTTPException, status
from backend.app.schemas.loan import LoanRequest, PredictionResponse
from backend.app.services.model_service import model_service

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict Loan Default Risk and Explain Decisions"
)
def predict_credit_risk(request: LoanRequest) -> PredictionResponse:
    """Evaluates applicant loan parameters and returns calibrated default probability,
    approval status, risk category, and SHAP explainability insights.
    """
    try:
        response = model_service.predict(request)
        return response
    except ValueError as ve:
        logger.warning(f"Validation error in prediction: {ve}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except RuntimeError as re:
        logger.error(f"Runtime model error: {re}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(re))
    except Exception as e:
        logger.exception(f"Unexpected prediction failure: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}"
        )
