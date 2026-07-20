from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
DATASET_DIR = ARTIFACTS_DIR / "Dataset"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"

CREDITCARD_CSV = DATASET_DIR / "creditcard.csv"
MODEL_COMPARISON_FILE = ARTIFACTS_DIR / "model_comparison.csv"
PR_CURVE_FILE = ARTIFACTS_DIR / "pr_curve.png"
SCALER_FILE = ARTIFACTS_DIR / "scaler.pkl"
FEATURE_NAMES_FILE = ARTIFACTS_DIR / "feature_names.pkl"
BASELINE_LR_FILE = ARTIFACTS_DIR / "baseline_lr.pkl"
SMOTE_LR_FILE = ARTIFACTS_DIR / "smote_lr.pkl"
ADASYN_LR_FILE = ARTIFACTS_DIR / "adasyn_lr.pkl"
WEIGHTED_LR_FILE = ARTIFACTS_DIR / "weighted_lr.pkl"
WEIGHTED_RF_FILE = ARTIFACTS_DIR / "weighted_rf.pkl"
BALANCED_RF_FILE = ARTIFACTS_DIR / "balanced_rf.pkl"
XGBOOST_FILE = ARTIFACTS_DIR / "xgboost.pkl"
