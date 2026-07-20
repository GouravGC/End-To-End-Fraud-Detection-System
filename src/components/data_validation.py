from dataclasses import dataclass

import pandas as pd

from src.exception import FraudDetectionException
from src.logger import logger


@dataclass
class DataValidationConfig:
    target_column: str = "Class"


class DataValidation:
    def __init__(self, config: DataValidationConfig | None = None):
        self.config = config or DataValidationConfig()

    def validate_dataset(self, df: pd.DataFrame) -> dict:
        try:
            logger.info("Starting dataset validation")
            validation_report = {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "missing_values": df.isnull().sum().to_dict(),
                "duplicate_rows": int(df.duplicated().sum()),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "target_distribution": df[self.config.target_column].value_counts(normalize=True).to_dict()
                if self.config.target_column in df.columns
                else {},
            }
            logger.info("Dataset validation completed")
            return validation_report
        except Exception as error:
            logger.exception("Failed during dataset validation")
            raise FraudDetectionException("Error during data validation", error) from error
