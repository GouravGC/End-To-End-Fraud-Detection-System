from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_processing import DataProcessing
from src.components.data_transformation import DataTransformation
from src.components.feature_engineering import FeatureEngineering
from src.components.model_evaluation import ModelEvaluation
from src.components.model_trainer import ModelTrainer
from src.constants.paths import FEATURE_NAMES_FILE, SCALER_FILE
from src.exception import FraudDetectionException
from src.logger import logger
from src.utils import save_object


@dataclass
class TrainPipelineArtifacts:
    x_train: Any
    x_test: Any
    y_train: Any
    y_test: Any
    scaler: Any
    feature_names: list[str]
    models: dict[str, Any]
    evaluation: dict[str, dict]


class TrainPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_processing = DataProcessing()
        self.feature_engineering = FeatureEngineering()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()
        self.model_evaluation = ModelEvaluation()

    def run(self) -> TrainPipelineArtifacts:
        try:
            logger.info("Starting training pipeline")
            df = self.data_ingestion.initiate_data_ingestion()
            cleaned_df = self.data_processing.clean_data(df)
            x, y = self.feature_engineering.split_features_and_target(cleaned_df)
            x_train, x_test, y_train, y_test = self.data_transformation.split_train_test(x, y)
            x_train_scaled, x_test_scaled = self.data_transformation.scale_features(x_train, x_test)

            negative_class = y_train.value_counts()[0]
            positive_class = y_train.value_counts()[1]
            scale_pos_weight = negative_class / positive_class

            baseline_model = self.model_trainer.train_baseline_lr(x_train_scaled, y_train)
            smote_model, x_train_smote, y_train_smote = self.model_trainer.train_smote_lr(x_train_scaled, y_train)
            adasyn_model, x_train_adasyn, y_train_adasyn = self.model_trainer.train_adasyn_lr(x_train_scaled, y_train)
            weighted_lr = self.model_trainer.train_weighted_lr(x_train_scaled, y_train)
            weighted_rf = self.model_trainer.train_weighted_rf(x_train_scaled, y_train)
            balanced_rf = self.model_trainer.train_balanced_rf(x_train_scaled, y_train)
            xgb_model = self.model_trainer.train_xgboost(x_train_scaled, y_train, scale_pos_weight)

            y_pred_baseline = baseline_model.predict(x_test_scaled)
            y_prob_baseline = baseline_model.predict_proba(x_test_scaled)[:, 1]
            y_pred_smote = smote_model.predict(x_test_scaled)
            y_prob_smote = smote_model.predict_proba(x_test_scaled)[:, 1]
            y_pred_adasyn = adasyn_model.predict(x_test_scaled)
            y_prob_adasyn = adasyn_model.predict_proba(x_test_scaled)[:, 1]
            y_pred_weighted_lr = weighted_lr.predict(x_test_scaled)
            y_prob_weighted_lr = weighted_lr.predict_proba(x_test_scaled)[:, 1]
            y_pred_weighted_rf = weighted_rf.predict(x_test_scaled)
            y_prob_weighted_rf = weighted_rf.predict_proba(x_test_scaled)[:, 1]
            y_pred_balanced_rf = balanced_rf.predict(x_test_scaled)
            y_prob_balanced_rf = balanced_rf.predict_proba(x_test_scaled)[:, 1]
            y_pred_xgb = xgb_model.predict(x_test_scaled)
            y_prob_xgb = xgb_model.predict_proba(x_test_scaled)[:, 1]

            evaluation = {
                "baseline": self.model_evaluation.evaluate_predictions(y_test, y_pred_baseline),
                "smote": self.model_evaluation.evaluate_predictions(y_test, y_pred_smote),
                "adasyn": self.model_evaluation.evaluate_predictions(y_test, y_pred_adasyn),
                "weighted_lr": self.model_evaluation.evaluate_predictions(y_test, y_pred_weighted_lr),
                "weighted_rf": self.model_evaluation.evaluate_predictions(y_test, y_pred_weighted_rf),
                "balanced_rf": self.model_evaluation.evaluate_predictions(y_test, y_pred_balanced_rf),
                "xgb": self.model_evaluation.evaluate_predictions(y_test, y_pred_xgb),
            }

            feature_names = x_train.columns.tolist()
            save_object(SCALER_FILE, self.data_transformation.scaler)
            save_object(FEATURE_NAMES_FILE, feature_names)

            logger.info("Training pipeline completed")
            return TrainPipelineArtifacts(
                x_train=x_train,
                x_test=x_test,
                y_train=y_train,
                y_test=y_test,
                scaler=self.data_transformation.scaler,
                feature_names=feature_names,
                models={
                    "baseline": baseline_model,
                    "smote": smote_model,
                    "adasyn": adasyn_model,
                    "weighted_lr": weighted_lr,
                    "weighted_rf": weighted_rf,
                    "balanced_rf": balanced_rf,
                    "xgb": xgb_model,
                },
                evaluation=evaluation,
            )
        except Exception as error:
            logger.exception("Training pipeline failed")
            raise FraudDetectionException("Error during training pipeline", error) from error
