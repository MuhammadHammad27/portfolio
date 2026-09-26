"""Evaluation and diagnostic utility for trained Credit Risk models."""
import os
import sys
import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, brier_score_loss

def evaluate_model(model_path: str, test_csv_path: str):
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)
    if not os.path.exists(test_csv_path):
        print(f"Error: Test dataset not found at {test_csv_path}")
        sys.exit(1)

    print(f"Loading model artifact from {model_path}...")
    artifact = joblib.load(model_path)
    preprocessor = artifact["preprocessor"]
    calibrated_model = artifact["calibrated_model"]

    print(f"Loading test data from {test_csv_path}...")
    df = pd.read_csv(test_csv_path).dropna()
    X = df.drop(columns=["loan_status"])
    y = df["loan_status"]

    X_trans = preprocessor.transform(X)
    preds = calibrated_model.predict(X_trans)
    probs = calibrated_model.predict_proba(X_trans)[:, 1]

    print("\n" + "=" * 50)
    print("MODEL EVALUATION REPORT")
    print("=" * 50)
    print(f"Samples Evaluated: {len(y)}")
    print(f"ROC-AUC Score:     {roc_auc_score(y, probs):.4f}")
    print(f"Brier Loss Score:  {brier_score_loss(y, probs):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y, preds))
    print("\nClassification Report:")
    print(classification_report(y, preds))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Credit Risk Model")
    parser.add_argument("--model-path", default="model/loan_model.pkl", help="Path to model artifact")
    parser.add_argument("--test-data", required=True, help="Path to test CSV data")
    args = parser.parse_args()

    evaluate_model(args.model_path, args.test_data)
