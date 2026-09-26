# 🧠 Machine Learning Training & Evaluation Pipeline

Tools and pipelines for training, calibrating, and explaining credit risk classification models.

---

## 🔬 Pipeline Architecture

1. **Preprocessing**:
   - Numerical columns: Passthrough / identity transformation.
   - Categorical columns: `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`.
2. **Base Model**:
   - `RandomForestClassifier(class_weight="balanced", n_estimators=100)`.
3. **Probability Calibration**:
   - `CalibratedClassifierCV(method="sigmoid", cv=5)` (Platt Scaling) ensuring predicted probabilities match true default rates.
4. **Explainability**:
   - `shap.TreeExplainer` fitted on tree ensembles for real-time additive feature importance.
5. **Serialization**:
   - Serialized with `joblib` compression (`--compress 3`) to reduce artifact size.

---

## 📁 Directory Structure

```text
ml/
├── train.py          # Production training CLI with logging & metrics
├── evaluate.py       # Model evaluation & diagnostics tool
├── requirements.txt  # Training-specific Python dependencies
└── README.md         # Pipeline documentation
```

---

## ⚡ How to Train

Install training dependencies:
```bash
pip install -r ml/requirements.txt
```

Run training on your dataset:
```bash
python ml/train.py \
    --data-path path/to/credit_risk_dataset.csv \
    --output-dir model \
    --n-estimators 100 \
    --compress 3
```

### CLI Arguments
| Argument | Type | Default | Description |
|---|---|---|---|
| `--data-path` | `str` | `data/credit_risk_dataset.csv` | Path to raw credit CSV dataset |
| `--output-dir` | `str` | `model` | Directory where `loan_model.pkl` is saved |
| `--n-estimators` | `int` | `100` | Number of decision trees |
| `--compress` | `int` | `3` | Joblib compression level (0-9) |
| `--test-size` | `float` | `0.2` | Fraction of dataset reserved for testing |

---

## 📊 How to Evaluate

To evaluate a saved model against a test dataset:
```bash
python ml/evaluate.py \
    --model-path model/loan_model.pkl \
    --test-data path/to/test_credit_risk.csv
```
This generates:
- ROC-AUC Score
- Brier Calibration Loss
- Confusion Matrix
- Precision, Recall & F1-Score Report
