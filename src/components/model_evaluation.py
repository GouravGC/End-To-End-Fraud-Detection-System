from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from src.constants.paths import MODEL_COMPARISON_FILE, PR_CURVE_FILE
from src.exception import FraudDetectionException
from src.logger import logger


@dataclass
class ModelEvaluationConfig:
    comparison_file: Path = MODEL_COMPARISON_FILE
    pr_curve_file: Path = PR_CURVE_FILE


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig | None = None):
        self.config = config or ModelEvaluationConfig()

    def evaluate_predictions(self, y_true: Any, y_pred: Any) -> dict:
        try:
            logger.info("Evaluating predictions")
            metrics = {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred),
                "recall": recall_score(y_true, y_pred),
                "f1_score": f1_score(y_true, y_pred),
                "mcc": matthews_corrcoef(y_true, y_pred),
                "confusion_matrix": confusion_matrix(y_true, y_pred),
                "classification_report": classification_report(y_true, y_pred),
            }
            return metrics
        except Exception as error:
            logger.exception("Failed to evaluate predictions")
            raise FraudDetectionException("Error during model evaluation", error) from error

    def evaluate_probabilities(self, y_true: Any, y_prob: Any) -> dict:
        try:
            logger.info("Evaluating prediction probabilities")
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
            return {
                "average_precision": average_precision_score(y_true, y_prob),
                "precision_curve": precision_curve,
                "recall_curve": recall_curve,
            }
        except Exception as error:
            logger.exception("Failed to evaluate prediction probabilities")
            raise FraudDetectionException("Error during probability evaluation", error) from error

    def load_existing_model_comparison(self) -> pd.DataFrame:
        try:
            logger.info("Loading existing model comparison from %s", self.config.comparison_file)
            return pd.read_csv(self.config.comparison_file)
        except Exception as error:
            logger.exception("Failed to load model comparison CSV")
            raise FraudDetectionException("Error loading model comparison artifact", error) from error

    def existing_pr_curve_path(self) -> Path:
        return self.config.pr_curve_file
