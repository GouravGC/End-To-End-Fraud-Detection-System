from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.constants.paths import FEATURE_NAMES_FILE, SCALER_FILE, XGBOOST_FILE
from src.exception import FraudDetectionException
from src.logger import logger
from src.utils import load_object


@dataclass
class PredictionPipelineConfig:
    model_path: Path = XGBOOST_FILE
    scaler_path: Path = SCALER_FILE
    feature_names_path: Path = FEATURE_NAMES_FILE


class PredictionPipeline:
    def __init__(self, config: PredictionPipelineConfig | None = None):
        self.config = config or PredictionPipelineConfig()
        self.model = load_object(self.config.model_path)
        self.scaler = load_object(self.config.scaler_path)
        self.feature_names = load_object(self.config.feature_names_path)

    def preprocess_input(self, input_data: pd.DataFrame) -> np.ndarray:
        try:
            logger.info("Preprocessing inference input")
            aligned_input = input_data[self.feature_names].copy()
            return self.scaler.transform(aligned_input)
        except Exception as error:
            logger.exception("Failed to preprocess inference input")
            raise FraudDetectionException("Error during prediction preprocessing", error) from error

    def predict(self, input_data: pd.DataFrame) -> dict:
        try:
            processed_input = self.preprocess_input(input_data)
            prediction = self.model.predict(processed_input)
            probability = None
            if hasattr(self.model, "predict_proba"):
                probability = self.model.predict_proba(processed_input)[:, 1]
            logger.info("Inference completed")
            return {
                "prediction": prediction,
                "probability": probability,
            }
        except Exception as error:
            logger.exception("Prediction failed")
            raise FraudDetectionException("Error during prediction", error) from error
