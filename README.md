# Fraud Detection Project

## Overview
This project detects fraudulent credit card transactions using a modular machine learning workflow built from an existing, fully working Jupyter Notebook. The notebook remains the single source of truth, and the production code extracts reusable logic into a structured Python project.

The repository includes a Streamlit application for interactive inference and a modular pipeline for training and prediction. Existing artifacts are reused as-is.

## Dataset
The project uses the well-known credit card fraud dataset stored in the existing artifacts folder:

- `artifacts/Dataset/creditcard.csv`

The dataset contains anonymized transaction features, transaction time, transaction amount, and the target label `Class`.

## Problem Statement
The goal is to classify transactions as legitimate or fraudulent while handling severe class imbalance and preserving the exact behavior of the original notebook pipeline.

## Features
- Modular data ingestion, validation, preprocessing, training, and evaluation
- Existing artifact loading for deployment-safe inference
- Multiple model training branches preserved from the notebook
- Precision-recall based comparison support
- Streamlit UI for interactive fraud prediction
- Structured logging and exception handling

## Project Structure
- `app.py` - Streamlit application
- `main.py` - Streamlit launcher
- `setup.py` - Package metadata
- `requirements.txt` - Python dependencies
- `.gitignore` - Ignore rules
- `src/` - Modular source code
- `artifacts/` - Existing dataset, models, scaler, feature names, plots, and reports
- `notebooks/` - Original notebook source

## Installation
1. Create and activate a Python environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Ensure the existing `artifacts/` directory remains in place.

## Training
The training workflow is implemented in the modular pipeline and mirrors the notebook behavior.

- Training entry point: `src/pipeline/train_pipeline.py`
- The pipeline loads the existing dataset, performs the same cleaning, preprocessing, model training, and evaluation logic from the notebook.

## Prediction
Prediction uses only existing artifacts and never fits any transformer or model.

- Prediction entry point: `src/pipeline/prediction_pipeline.py`
- Required artifacts:
  - `artifacts/xgboost.pkl`
  - `artifacts/scaler.pkl`
  - `artifacts/feature_names.pkl`

## Streamlit
Run the application with:

- `streamlit run app.py`

The UI provides:
- Sidebar input widgets
- Input summary
- Prediction result
- Fraud probability when available

## Artifacts
The project reuses these existing artifacts without regeneration:

- `artifacts/Dataset/creditcard.csv`
- `artifacts/scaler.pkl`
- `artifacts/feature_names.pkl`
- `artifacts/baseline_lr.pkl`
- `artifacts/smote_lr.pkl`
- `artifacts/adasyn_lr.pkl`
- `artifacts/weighted_lr.pkl`
- `artifacts/weighted_rf.pkl`
- `artifacts/balanced_rf.pkl`
- `artifacts/xgboost.pkl`
- `artifacts/model_comparison.csv`
- `artifacts/pr_curve.png`

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- imbalanced-learn
- XGBoost
- Joblib
- Streamlit

## Results
The notebook already contains the completed training and evaluation workflow, along with saved models, comparison metrics, and visual artifacts. The modular project is designed to reuse those outputs directly and keep deployment behavior aligned with the notebook.
