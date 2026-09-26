# Model Artifacts

This directory stores the serialized scikit-learn and SHAP model artifacts for the Credit Risk Assessment System.

## Primary Artifact: `loan_model.pkl`

The serialized dictionary contains:
- `preprocessor`: `ColumnTransformer` with numeric passthrough and categorical `OneHotEncoder(handle_unknown="ignore")`.
- `calibrated_model`: `CalibratedClassifierCV` (Platt Scaling / Sigmoid method) wrapping a balanced `RandomForestClassifier`.
- `explainer`: `shap.TreeExplainer` fitted on the base tree ensemble for real-time feature attributions.
- `feature_names`: Full list of feature names including one-hot encoded categories.
- `num_cols`: `['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']`.
- `cat_cols`: `['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']`.

## Production & Git LFS Notice

- Due to SHAP's `TreeExplainer` tree structures, uncompressed model files can exceed 100 MB.
- **GitHub Limit**: GitHub restricts files larger than 100 MB.
  - To push large models to GitHub, use **Git LFS**:
    ```bash
    git lfs install
    git lfs track "*.pkl"
    git add .gitattributes
    git add model/loan_model.pkl
    ```
- **Compression**: You can retrain with compression using `ml/train.py --compress 3`, which reduces file size significantly (often below 40MB).
- **Cloud Object Storage (S3 / Cloudflare R2 / Hugging Face)**: For serverless deployments (such as Vercel), download the model file dynamically at cold start or point to a signed CDN URL.
