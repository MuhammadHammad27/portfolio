"""Credit Risk Model Training Pipeline.
Trains a balanced Random Forest classifier, applies Platt Scaling calibration,
builds SHAP TreeExplainer, and saves the complete serialized pipeline.
"""
import os
import sys
import argparse
import logging
import joblib
import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score, classification_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ml_train")

def parse_args():
    parser = argparse.ArgumentParser(description="Train and serialize Credit Risk ML model")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/credit_risk_dataset.csv",
        help="Path to credit risk dataset CSV"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="model",
        help="Directory to save trained model artifact"
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees in RandomForest"
    )
    parser.add_argument(
        "--compress",
        type=int,
        default=3,
        help="Joblib compression level (0-9, recommended 3 to reduce file size)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split fraction"
    )
    return parser.parse_args()

def run_training(args):
    data_path = args.data_path
    if not os.path.exists(data_path):
        logger.error(f"Dataset file not found at: {data_path}")
        logger.info("Please provide the correct CSV path using --data-path <path_to_csv>")
        sys.exit(1)

    logger.info(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    initial_len = len(df)
    df = df.dropna()
    logger.info(f"Cleaned dataset: {len(df)} rows retained (dropped {initial_len - len(df)} rows with missing values).")

    target_col = "loan_status"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset columns: {list(df.columns)}")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    cat_cols = list(X.select_dtypes(include=['object', 'category']).columns)
    num_cols = list(X.select_dtypes(include=['int64', 'float64']).columns)

    logger.info(f"Numerical features ({len(num_cols)}): {num_cols}")
    logger.info(f"Categorical features ({len(cat_cols)}): {cat_cols}")

    # Build preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    logger.info("Fitting feature transformation pipeline...")
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # Base Classifier
    logger.info(f"Training base RandomForestClassifier with {args.n_estimators} estimators...")
    base_rf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    # Calibrated Classifier (Platt Scaling)
    logger.info("Calibrating classifier probabilities via CalibratedClassifierCV (Sigmoid)...")
    calibrated_clf = CalibratedClassifierCV(
        estimator=base_rf,
        method="sigmoid",
        cv=5
    )
    calibrated_clf.fit(X_train_trans, y_train)

    # Evaluation
    preds = calibrated_clf.predict(X_test_trans)
    probs = calibrated_clf.predict_proba(X_test_trans)[:, 1]

    acc = accuracy_score(y_test, preds)
    brier = brier_score_loss(y_test, probs)
    roc_auc = roc_auc_score(y_test, probs)

    logger.info("==========================================")
    logger.info(f"Test Accuracy:         {acc * 100:.2f}%")
    logger.info(f"Test ROC-AUC Score:    {roc_auc:.4f}")
    logger.info(f"Brier Calibration Loss:{brier:.4f}")
    logger.info("==========================================")
    print("\nClassification Report:\n", classification_report(y_test, preds))

    # Fit base model for SHAP explainer
    logger.info("Fitting standalone base estimator for SHAP TreeExplainer...")
    base_rf.fit(X_train_trans, y_train)
    explainer = shap.TreeExplainer(base_rf)

    # Extract all feature names after One-Hot Encoding
    if len(cat_cols) > 0:
        cat_feature_names = list(
            preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols)
        )
    else:
        cat_feature_names = []

    all_feature_names = num_cols + cat_feature_names

    # Save artifact
    os.makedirs(args.output_dir, exist_ok=True)
    model_output_path = os.path.join(args.output_dir, "loan_model.pkl")

    artifact = {
        "preprocessor": preprocessor,
        "calibrated_model": calibrated_clf,
        "explainer": explainer,
        "feature_names": all_feature_names,
        "num_cols": num_cols,
        "cat_cols": cat_cols
    }

    logger.info(f"Serializing artifact with compression level {args.compress} to: {model_output_path}")
    joblib.dump(artifact, model_output_path, compress=args.compress)

    file_size_mb = os.path.getsize(model_output_path) / (1024 * 1024)
    logger.info(f"Successfully saved artifact ({file_size_mb:.2f} MB) at {model_output_path}")

if __name__ == "__main__":
    args = parse_args()
    run_training(args)
