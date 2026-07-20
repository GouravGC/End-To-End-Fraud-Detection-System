from pathlib import Path
from typing import Any

import joblib

from src.exception import FraudDetectionException
from src.logger import logger


def save_object(file_path: Path, obj: Any) -> None:
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, file_path)
        logger.info("Saved object to %s", file_path)
    except Exception as error:
        logger.exception("Failed to save object to %s", file_path)
        raise FraudDetectionException(f"Error saving object to {file_path}", error) from error


def load_object(file_path: Path) -> Any:
    try:
        return joblib.load(file_path)
    except Exception as error:
        logger.exception("Failed to load object from %s", file_path)
        raise FraudDetectionException(f"Error loading object from {file_path}", error) from error
