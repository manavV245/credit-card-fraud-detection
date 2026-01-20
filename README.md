# Credit Card Fraud Detection (Imbalanced Classification)

Detect fraudulent credit card transactions on a highly imbalanced dataset (fraud is rare). This project focuses on practical evaluation (PR-AUC / precision–recall), leakage-safe resampling, and choosing an operating threshold that matches the problem.

## What's in this project:
- EDA (class imbalance, feature distributions/behaviour)
- Baseline Models (Logistic Regression + Tree Baseline)
- Imbalance handling:
  - cost-sensitive learning (class weighting)
  - SMOTE applied safely inside CV pipelines
- Threshold tuning using out-of-fold (OOF) probabilities (no test leakage)
- Model comparison with PR curves + confusion matrices

## Dataset
Kaggle: “Credit Card Fraud Detection” (MLG-ULB)  
Place `creditcard.csv` in `data/raw/` (it is in .gitignore so won't come with clone)

## Results 
- Best Model: **XGBoost**
- Test PR-AUC: **0.83**
- Test ROC-AUC: **0.98**
- Chosen Threshold: **0.91**
- Confusion Matrix (test): **[TN FP FN TP] = [56845 19 17 81]**

## Repo structure
    data/
      raw/            # creditcard.csv (from kaggle)
    notebooks/        # EDA + Modelling notebooks
    reports/
      figures/        # exported plots
      metrics/        # exported json/csv summaries

## Setup
    python -m venv .venv
    # Windows:
    .\.venv\Scripts\activate
    pip install -r requirements.txt

## Run
Open notebooks in order:
1. `notebooks/01_eda.ipynb`
2. `notebooks/03_thresholding_and_pr_curves.ipynb`
3. `notebooks/04_smote_and_xgboost.ipynb`

## Notes
- PR-AUC is emphasized because accuracy/ROC-AUC can look strong even when fraud recall is weak.
- Threshold selection is done from training OOF predictions to avoid peeking at the test set.
