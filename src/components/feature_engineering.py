from dataclasses import dataclass

import pandas as pd

from src.exception import FraudDetectionException
from src.logger import logger


@dataclass
class FeatureEngineeringConfig:
    target_column: str = "Class"


class FeatureEngineering:
    def __init__(self, config: FeatureEngineeringConfig | None = None):
        self.config = config or FeatureEngineeringConfig()

    def split_features_and_target(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        try:
            logger.info("Splitting features and target")
            x = df.drop(self.config.target_column, axis=1)
            y = df[self.config.target_column]
            return x, y
        except Exception as error:
            logger.exception("Failed during feature engineering")
            raise FraudDetectionException("Error during feature engineering", error) from error
