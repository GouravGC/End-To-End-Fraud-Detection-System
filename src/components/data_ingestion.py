from dataclasses import dataclass

import pandas as pd

from src.constants.paths import CREDITCARD_CSV
from src.exception import FraudDetectionException
from src.logger import logger


@dataclass
class DataIngestionConfig:
    dataset_path: str = str(CREDITCARD_CSV)


class DataIngestion:
    def __init__(self, config: DataIngestionConfig | None = None):
        self.config = config or DataIngestionConfig()

    def initiate_data_ingestion(self) -> pd.DataFrame:
        try:
            logger.info("Loading dataset from %s", self.config.dataset_path)
            df = pd.read_csv(self.config.dataset_path)
            logger.info("Dataset loaded successfully with shape: %s", df.shape)
            return df
        except Exception as error:
            logger.exception("Failed to load dataset")
            raise FraudDetectionException("Error during data ingestion", error) from error
