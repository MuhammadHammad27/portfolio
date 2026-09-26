import os
import joblib
import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, brier_score_loss

# 1. Folder & File Path Setup
os.makedirs("model", exist_ok=True)
csv_path = r"C:\Users\New folder\credit_risk_dataset.csv"

# 2. Data Load & Clean
print(f"Loading data from {csv_path}...")
df = pd.read_csv(csv_path)
df = df.dropna()

print("Data loaded successfully!")
print("Columns in your dataset:", list(df.columns))

X = df.drop(columns=["loan_status"])  # <--- Change column name as per your CSV
y = df["loan_status"]

cat_cols = list(X.select_dtypes(include=['object', 'category']).columns)
num_cols = list(X.select_dtypes(include=['int64', 'float64']).columns)

print(f"Numerical Features: {num_cols}")
print(f"Categorical Features: {cat_cols}")

    # Preprocessing Pipeline
preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        ]
    )

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_trans = preprocessor.fit_transform(X_train)
X_test_trans = preprocessor.transform(X_test)

# Base Classifier
base_rf = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42
)

# Calibrated Classifier (Platt Scaling)
calibrated_clf = CalibratedClassifierCV(
    estimator=base_rf,
    method="sigmoid",
    cv=5
)

calibrated_clf.fit(X_train_trans, y_train)

# Evaluate Metrics
preds = calibrated_clf.predict(X_test_trans)
probs = calibrated_clf.predict_proba(X_test_trans)[:, 1]

acc = accuracy_score(y_test, preds)
brier = brier_score_loss(y_test, probs)

print("\nModel Training Complete!")
print(f"Accuracy: {acc * 100:.2f}% | Brier Score (Calibration): {brier:.4f}")

# TreeExplainer fit
base_rf.fit(X_train_trans, y_train)
explainer = shap.TreeExplainer(base_rf)

# Extract all feature names after One-Hot Encoding
if len(cat_cols) > 0:
    cat_feature_names = list(
        preprocessor.named_transformers_["cat"]
        .get_feature_names_out(cat_cols)
    )
else:
    cat_feature_names = []

all_feature_names = num_cols + cat_feature_names

# Save Artifacts
artifact = {
    "preprocessor": preprocessor,
    "calibrated_model": calibrated_clf,
    "explainer": explainer,
    "feature_names": all_feature_names,
    "num_cols": num_cols,
    "cat_cols": cat_cols
}

model_path = os.path.join("model", "loan_model.pkl")

joblib.dump(artifact, model_path)

print(f"Artifact saved at: {model_path}")
