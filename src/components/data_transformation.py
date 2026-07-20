from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.exception import FraudDetectionException
from src.logger import logger


@dataclass
class DataTransformationConfig:
    test_size: float = 0.2
    random_state: int = 42


class DataTransformation:
    def __init__(self, config: DataTransformationConfig | None = None):
        self.config = config or DataTransformationConfig()
        self.scaler = StandardScaler()

    def split_train_test(
        self,
        x: pd.DataFrame,
        y: pd.Series,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        try:
            logger.info("Creating train and test splits")
            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                test_size=self.config.test_size,
                stratify=y,
                random_state=self.config.random_state,
            )
            return x_train, x_test, y_train, y_test
        except Exception as error:
            logger.exception("Failed during train/test split")
            raise FraudDetectionException("Error during data transformation", error) from error

    def scale_features(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        try:
            logger.info("Scaling feature matrix")
            x_train_scaled = self.scaler.fit_transform(x_train)
            x_test_scaled = self.scaler.transform(x_test)
            return x_train_scaled, x_test_scaled
        except Exception as error:
            logger.exception("Failed during feature scaling")
            raise FraudDetectionException("Error during feature scaling", error) from error
