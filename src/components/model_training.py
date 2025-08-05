import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, TimeSeriesSplit
from sklearn.base import BaseEstimator
from typing import Dict, Any
from pathlib import Path
import joblib

from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

from prettytable import PrettyTable

from src.utils.logger import get_logger
from src.utils.data_loader import load_csv
from src.utils.config import (
    X_TRAIN_PROCESSED_FILE, Y_TRAIN_TRANSFORMED_FILE,
    X_VAL_PROCESSED_FILE, Y_VAL_TRANSFORMED_FILE
    )

logger = get_logger("training_pipeline")

class ModelTrainer:
    def __init__(self, model_configs: Dict[str, Dict[str, Any]],
                 search_type: str = "random", n_iter: int = 10, n_splits: int = 3,
                 is_time_series: bool = True, log_to_mlflow: bool = True):
        self.model_configs = model_configs
        self.search_type = search_type
        self.n_iter = n_iter
        self.n_splits = n_splits
        self.is_time_series = is_time_series
        self.log_to_mlflow = log_to_mlflow

    @staticmethod
    def _calculate_metrics(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return mse, rmse, mae, r2

    @staticmethod
    def _adjusted_r2(r2, n, p):
        return 1 - (1 - r2) * ((n - 1) / (n - p - 1)) if n > p + 1 else r2

    def _log_model(self, model, model_name, X_sample):
        try:
            if isinstance(model, BaseEstimator):
                mlflow.sklearn.log_model(model, name=model_name, input_example=X_sample)
            elif isinstance(model, (XGBRegressor, XGBClassifier)):
                mlflow.xgboost.log_model(model.get_booster(), name=model_name)
            elif isinstance(model, (LGBMRegressor, LGBMClassifier)):
                mlflow.lightgbm.log_model(model.booster_, name=model_name)
            else:
                logger.warning(f"Unsupported model type for logging: {type(model)}")
        except Exception as e:
            logger.error(f"Model logging failed: {e}")

    def evaluate_model(self, model, X_train, y_train, X_val, y_val, model_name, run_name):
        y_train_pred_log = model.predict(X_train)
        y_val_pred_log = model.predict(X_val)

        y_train_orig = np.expm1(y_train)
        y_val_orig = np.expm1(y_val)
        y_train_pred_orig = np.expm1(y_train_pred_log)
        y_val_pred_orig = np.expm1(y_val_pred_log)

        train_mse = mean_squared_error(y_train_orig, y_train_pred_orig)
        val_mse = mean_squared_error(y_val_orig, y_val_pred_orig)

        train_rmse = np.sqrt(train_mse)
        val_rmse = np.sqrt(val_mse)

        train_mae = mean_absolute_error(y_train_orig, y_train_pred_orig)
        val_mae = mean_absolute_error(y_val_orig, y_val_pred_orig)

        train_r2 = r2_score(y_train_orig, y_train_pred_orig)
        val_r2 = r2_score(y_val_orig, y_val_pred_orig)

        train_adj_r2 = self._adjusted_r2(train_r2, len(X_train), X_train.shape[1])
        val_adj_r2 = self._adjusted_r2(val_r2, len(X_val), X_val.shape[1])

        if self.log_to_mlflow and run_name:
            with mlflow.start_run(run_name=run_name):
                mlflow.log_metrics({
                    "Train_RMSE": train_rmse,
                    "Val_RMSE": val_rmse,
                    "Train_MAE": train_mae,
                    "Val_MAE": val_mae,
                    "Train_R2": train_r2,
                    "Val_R2": val_r2
                })
                if hasattr(model, "get_params"):
                    mlflow.log_params(model.get_params())
                self._log_model(model, model_name, X_train.head(2).astype(np.float64))

        table = PrettyTable()
        table.field_names = ["Metric", "Train", "Validation"]
        table.add_row(["R2 Score", round(train_r2, 4), round(val_r2, 4)])
        table.add_row(["Adjusted R2", round(train_adj_r2, 4), round(val_adj_r2, 4)])
        table.add_row(["MSE", round(train_mse, 4), round(val_mse, 4)])
        table.add_row(["RMSE", round(train_rmse, 4), round(val_rmse, 4)])
        table.add_row(["MAE", round(train_mae, 4), round(val_mae, 4)])

        return table, val_r2

    def train(self, X_train, y_train, X_val, y_val):
        best_model, best_score = None, float('-inf')
        best_model_name, best_params = None, {}

        cv_strategy = TimeSeriesSplit(n_splits=self.n_splits) if self.is_time_series else self.n_splits

        for model_name, config in self.model_configs.items():
            try:
                logger.info(f"Training model: {model_name}")
                base_model = config["model"]
                param_grid = config.get("params", {})
                run_name = config.get("run_name", f"{model_name}_tuning")

                search = (RandomizedSearchCV(base_model, param_distributions=param_grid, n_iter=self.n_iter,
                                             cv=cv_strategy, n_jobs=-1, verbose=1)
                          if self.search_type == "random"
                          else GridSearchCV(base_model, param_grid=param_grid, cv=cv_strategy, n_jobs=-1, verbose=1))

                search.fit(X_train, y_train)
                model = search.best_estimator_

                table, val_r2 = self.evaluate_model(model, X_train, y_train, X_val, y_val, model_name, run_name)
                print(table)

                if val_r2 > best_score:
                    best_model = model
                    best_model_name = model_name
                    best_score = val_r2
                    best_params = search.best_params_

            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")

        logger.info(f"Best model: {best_model_name} with R2: {best_score}")
        return best_model, best_model_name, best_params

    def train_final_model(self, model, model_name,
                          X_train: pd.DataFrame, y_train: pd.Series,
                          X_val: pd.DataFrame, y_val: pd.Series):
        
        X_combined = pd.concat([X_train, X_val])
        y_combined = pd.concat([y_train, y_val])

        model.fit(X_combined, y_combined)
        logger.info("Retrained model on combined train + validation set.")

        model_dir = Path("artifacts") / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"{model_name}_final_model.pkl"
        joblib.dump(model, model_path)
        logger.info(f"Saved model to {model_path}")

        with mlflow.start_run(run_name=f"{model_name}_FinalModel"):
            mlflow.log_artifact(str(model_path), artifact_path="final_model")
            mlflow.log_params(model.get_params())
            self._log_model(model, model_name, X_combined.head(2).astype(np.float64))

        return model, model_path


def main():
    X_train = load_csv(X_TRAIN_PROCESSED_FILE)
    y_train = load_csv(Y_TRAIN_TRANSFORMED_FILE).squeeze()
    X_val = load_csv(X_VAL_PROCESSED_FILE)
    y_val = load_csv(Y_VAL_TRANSFORMED_FILE).squeeze()

    model_configs = {
        "XGBoost": {
            "model": XGBRegressor(),
            # "params": {
            #     'n_estimators': [100, 300, 500],
            #     'alpha': [0.1, 10, 100],
            #     'learning_rate': [0.1, 1.0]
            # },
            "run_name": "XGBoost_Tuning"
        },
        "LightGBM": {
            "model": LGBMRegressor(verbose=-1),
            # "params": {
            #     'alpha': [0.1, 10, 100],
            #     'num_leaves': [20, 31, 40],
            #     'learning_rate': [0.1, 1.0],
            #     'n_estimators': [100, 300, 500]
            # },
            "run_name": "LightGBM_Tuning"
        }
    }

    trainer = ModelTrainer(model_configs=model_configs, search_type="random", n_iter=5,
                           is_time_series=False, n_splits=2, log_to_mlflow=True)

    best_model, best_model_name, best_params = trainer.train(X_train, y_train, X_val, y_val)
    final_model, model_path = trainer.train_final_model(
        best_model, best_model_name,
        X_train, y_train, X_val, y_val
        )

    return final_model, model_path, best_model_name


if __name__ == "__main__":
    final_model, model_path, best_model_name = main()
