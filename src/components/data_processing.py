from dataclasses import dataclass

import pandas as pd

from src.exception import FraudDetectionException
from src.logger import logger


@dataclass
class DataProcessingConfig:
    target_column: str = "Class"


class DataProcessing:
    def __init__(self, config: DataProcessingConfig | None = None):
        self.config = config or DataProcessingConfig()

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info("Starting data cleaning")
            cleaned_df = df.copy()
            cleaned_df.dropna(inplace=True)
            cleaned_df.drop_duplicates(inplace=True)
            if self.config.target_column in cleaned_df.columns:
                cleaned_df[self.config.target_column] = cleaned_df[self.config.target_column].astype(int)
            logger.info("Data cleaning completed with shape: %s", cleaned_df.shape)
            return cleaned_df
        except Exception as error:
            logger.exception("Failed during data cleaning")
            raise FraudDetectionException("Error during data processing", error) from error
