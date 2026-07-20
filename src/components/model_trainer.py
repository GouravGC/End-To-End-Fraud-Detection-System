from dataclasses import dataclass
from typing import Any

import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import ADASYN, SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.exception import FraudDetectionException
from src.logger import logger


@dataclass
class ModelTrainerConfig:
    random_state: int = 42
    max_iter: int = 1000
    rf_n_estimators: int = 200
    xgb_eval_metric: str = "logloss"


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig | None = None):
        self.config = config or ModelTrainerConfig()

    def train_baseline_lr(self, x_train: Any, y_train: pd.Series) -> LogisticRegression:
        try:
            logger.info("Training baseline logistic regression")
            model = LogisticRegression(
                random_state=self.config.random_state,
                max_iter=self.config.max_iter,
            )
            model.fit(x_train, y_train)
            return model
        except Exception as error:
            logger.exception("Failed to train baseline logistic regression")
            raise FraudDetectionException("Error training baseline logistic regression", error) from error

    def train_smote_lr(self, x_train: Any, y_train: pd.Series) -> tuple[LogisticRegression, Any, Any]:
        try:
            logger.info("Training SMOTE logistic regression")
            smote = SMOTE(random_state=self.config.random_state)
            x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)
            model = LogisticRegression(
                random_state=self.config.random_state,
                max_iter=self.config.max_iter,
            )
            model.fit(x_train_smote, y_train_smote)
            return model, x_train_smote, y_train_smote
        except Exception as error:
            logger.exception("Failed to train SMOTE logistic regression")
            raise FraudDetectionException("Error training SMOTE logistic regression", error) from error

    def train_adasyn_lr(self, x_train: Any, y_train: pd.Series) -> tuple[LogisticRegression, Any, Any]:
        try:
            logger.info("Training ADASYN logistic regression")
            adasyn = ADASYN(random_state=self.config.random_state)
            x_train_adasyn, y_train_adasyn = adasyn.fit_resample(x_train, y_train)
            model = LogisticRegression(
                random_state=self.config.random_state,
                max_iter=self.config.max_iter,
            )
            model.fit(x_train_adasyn, y_train_adasyn)
            return model, x_train_adasyn, y_train_adasyn
        except Exception as error:
            logger.exception("Failed to train ADASYN logistic regression")
            raise FraudDetectionException("Error training ADASYN logistic regression", error) from error

    def train_weighted_lr(self, x_train: Any, y_train: pd.Series) -> LogisticRegression:
        try:
            logger.info("Training weighted logistic regression")
            model = LogisticRegression(
                class_weight="balanced",
                random_state=self.config.random_state,
                max_iter=self.config.max_iter,
            )
            model.fit(x_train, y_train)
            return model
        except Exception as error:
            logger.exception("Failed to train weighted logistic regression")
            raise FraudDetectionException("Error training weighted logistic regression", error) from error

    def train_weighted_rf(self, x_train: Any, y_train: pd.Series) -> RandomForestClassifier:
        try:
            logger.info("Training weighted random forest")
            model = RandomForestClassifier(
                n_estimators=self.config.rf_n_estimators,
                class_weight="balanced",
                random_state=self.config.random_state,
                n_jobs=-1,
            )
            model.fit(x_train, y_train)
            return model
        except Exception as error:
            logger.exception("Failed to train weighted random forest")
            raise FraudDetectionException("Error training weighted random forest", error) from error

    def train_balanced_rf(self, x_train: Any, y_train: pd.Series) -> BalancedRandomForestClassifier:
        try:
            logger.info("Training balanced random forest")
            model = BalancedRandomForestClassifier(
                n_estimators=self.config.rf_n_estimators,
                random_state=self.config.random_state,
                n_jobs=-1,
            )
            model.fit(x_train, y_train)
            return model
        except Exception as error:
            logger.exception("Failed to train balanced random forest")
            raise FraudDetectionException("Error training balanced random forest", error) from error

    def train_xgboost(self, x_train: Any, y_train: pd.Series, scale_pos_weight: float) -> XGBClassifier:
        try:
            logger.info("Training XGBoost model")
            model = XGBClassifier(
                objective="binary:logistic",
                scale_pos_weight=scale_pos_weight,
                random_state=self.config.random_state,
                eval_metric=self.config.xgb_eval_metric,
            )
            model.fit(x_train, y_train)
            return model
        except Exception as error:
            logger.exception("Failed to train XGBoost model")
            raise FraudDetectionException("Error training XGBoost model", error) from error
