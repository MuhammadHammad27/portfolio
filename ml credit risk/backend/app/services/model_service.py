import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import joblib
import numpy as np
import pandas as pd

from backend.app.core.config import settings
from backend.app.schemas.loan import LoanRequest, PredictionResponse, ShapExplanation

logger = logging.getLogger(__name__)

class ModelService:
    """Production service for credit risk model inference and SHAP explainability."""

    _instance: Optional["ModelService"] = None

    def __init__(self):
        self.artifact: Optional[Dict[str, Any]] = None
        self.preprocessor = None
        self.calibrated_model = None
        self.explainer = None
        self.feature_names: List[str] = []
        self.num_cols: List[str] = []
        self.cat_cols: List[str] = []
        self.is_loaded: bool = False

    @classmethod
    def get_instance(cls) -> "ModelService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def find_model_path(self, override_path: Optional[str] = None) -> Optional[Path]:
        """Locates the model artifact across multiple standard paths."""
        candidate_paths = []
        if override_path:
            candidate_paths.append(Path(override_path))
        if settings.MODEL_PATH:
            candidate_paths.append(Path(settings.MODEL_PATH))
            
        # Standard fallback paths
        root_dir = Path(__file__).resolve().parent.parent.parent.parent
        candidate_paths.extend([
            root_dir / "model" / "loan_model.pkl",
            Path("model/loan_model.pkl"),
            Path("../model/loan_model.pkl"),
            Path("../../model/loan_model.pkl"),
        ])

        for path in candidate_paths:
            if path.exists() and path.is_file():
                return path.resolve()
        return None

    def load_model(self, model_path: Optional[str] = None) -> bool:
        """Loads and initializes model artifacts from disk."""
        resolved_path = self.find_model_path(model_path)
        if not resolved_path:
            logger.error("Model file could not be found in any expected location.")
            return False

        logger.info(f"Loading credit risk model from: {resolved_path}")
        start_time = time.time()
        try:
            self.artifact = joblib.load(str(resolved_path))
            self.preprocessor = self.artifact.get("preprocessor")
            self.calibrated_model = self.artifact.get("calibrated_model")
            self.explainer = self.artifact.get("explainer")
            self.feature_names = self.artifact.get("feature_names", [])
            self.num_cols = self.artifact.get("num_cols", [])
            self.cat_cols = self.artifact.get("cat_cols", [])

            self.is_loaded = (
                self.preprocessor is not None and
                self.calibrated_model is not None and
                self.explainer is not None
            )
            elapsed = time.time() - start_time
            logger.info(f"Model successfully loaded in {elapsed:.2f}s with {len(self.feature_names)} features.")
            return self.is_loaded
        except Exception as e:
            logger.exception(f"Error loading model artifact: {e}")
            self.is_loaded = False
            return False

    def predict(self, request_data: LoanRequest) -> PredictionResponse:
        """Runs full inference and computes SHAP feature contributions."""
        if not self.is_loaded:
            loaded = self.load_model()
            if not loaded:
                raise RuntimeError("Model artifact is not available. Please ensure model/loan_model.pkl exists.")

        start_time = time.perf_counter()

        # Convert Pydantic request to DataFrame
        input_data = request_data.model_dump() if hasattr(request_data, "model_dump") else request_data.dict()
        input_df = pd.DataFrame([input_data])

        # 1. Pipeline Feature Transformation
        transformed_input = self.preprocessor.transform(input_df)

        # 2. Binary Prediction (0 = Non-Default/Approved, 1 = Default/Rejected)
        raw_pred = self.calibrated_model.predict(transformed_input)
        prediction = int(np.ravel(raw_pred)[0])

        # 3. Calibrated Probabilities
        probabilities = self.calibrated_model.predict_proba(transformed_input)
        prob_array = np.ravel(probabilities)
        default_prob = round(float(prob_array[1] if len(prob_array) > 1 else prob_array[0]), 4)

        # 4. SHAP Explainability
        shap_values = self.explainer.shap_values(transformed_input)
        if isinstance(shap_values, list):
            # Binary classification list: [class_0_shap, class_1_shap]
            raw_vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            raw_vals = shap_values

        vals = np.array(raw_vals).reshape(-1)

        explanations: List[ShapExplanation] = []
        for feat, v in zip(self.feature_names, vals):
            v_float = round(float(v), 4)
            impact_text = "Increases Default Risk" if v_float > 0 else "Decreases Default Risk"
            explanations.append(
                ShapExplanation(
                    feature=str(feat),
                    shap_value=v_float,
                    impact=impact_text
                )
            )

        # Sort explanations by absolute contribution
        explanations.sort(key=lambda x: abs(x.shap_value), reverse=True)

        # Decision & Risk Categorization
        is_approved = bool(prediction == 0)
        if default_prob < 0.20:
            risk_level = "Low Risk"
        elif default_prob < 0.45:
            risk_level = "Moderate Risk"
        else:
            risk_level = "High Risk"

        risk_score = int(round(default_prob * 1000))
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        decision_message = (
            "Loan Approved: Applicant profile meets risk thresholds."
            if is_approved
            else "Loan Rejected: High probability of default detected."
        )

        return PredictionResponse(
            approved=is_approved,
            calibrated_probability=default_prob,
            risk_level=risk_level,
            risk_score=risk_score,
            message=decision_message,
            inference_time_ms=elapsed_ms,
            shap_explanations=explanations
        )

model_service = ModelService.get_instance()
