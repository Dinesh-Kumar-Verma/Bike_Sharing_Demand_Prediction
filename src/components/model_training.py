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

from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

from prettytable import PrettyTable

from src.utils.logger import get_logger
from src.utils.data_loader import load_csv
from src.components.data_splitter import split_data
from src.utils.config import (X_TRAIN_FILE, Y_TRAIN_FILE, X_VAL_FILE, Y_VAL_FILE,)




logger = get_logger("training_pipeline")


# ----------------- Metrics Utils -------------------

def _calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, r2

def _adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * ((n - 1) / (n - p - 1)) if n > p + 1 else r2


# ----------------- MLflow Logging -------------------

def _log_model_with_mlflow(model, model_name, X_sample):
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


def evaluate_and_log_model(model, X_train, y_train, X_val, y_val, model_name, run_name, log=True):
    # Get predictions in log scale
    y_train_pred_log = model.predict(X_train)
    y_val_pred_log = model.predict(X_val)

    # Inverse transform to original scale
    y_train_orig = np.expm1(y_train)
    y_val_orig = np.expm1(y_val)
    y_train_pred_orig = np.expm1(y_train_pred_log)
    y_val_pred_orig = np.expm1(y_val_pred_log)

    # Compute metrics in original scale
    train_mse = mean_squared_error(y_train_orig, y_train_pred_orig)
    val_mse = mean_squared_error(y_val_orig, y_val_pred_orig)

    train_rmse = np.sqrt(train_mse)
    val_rmse = np.sqrt(val_mse)

    train_mae = mean_absolute_error(y_train_orig, y_train_pred_orig)
    val_mae = mean_absolute_error(y_val_orig, y_val_pred_orig)

    train_r2 = r2_score(y_train_orig, y_train_pred_orig)
    val_r2 = r2_score(y_val_orig, y_val_pred_orig)

    train_adj_r2 = _adjusted_r2(train_r2, len(X_train), X_train.shape[1])
    val_adj_r2 = _adjusted_r2(val_r2, len(X_val), X_val.shape[1])

    # MLflow logging
    if log and run_name:
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
                params = model.get_params()
                mlflow.log_params(params)

            _log_model_with_mlflow(model, model_name,  X_train.head(2).astype(np.float64))

    # PrettyTable Summary
    table = PrettyTable()
    table.field_names = ["Metric", "Train", "Validation"]
    table.add_row(["R2 Score", round(train_r2, 4), round(val_r2, 4)])
    table.add_row(["Adjusted R2", round(train_adj_r2, 4), round(val_adj_r2, 4)])
    table.add_row(["MSE", round(train_mse, 4), round(val_mse, 4)])
    table.add_row(["RMSE", round(train_rmse, 4), round(val_rmse, 4)])
    table.add_row(["MAE", round(train_mae, 4), round(val_mae, 4)])

    return table




# ----------------- Training Function -------------------

def train_models_with_search(
    X_train, y_train, X_val, y_val,
    model_configs: Dict[str, Dict[str, Any]],
    search_type: str = "random",
    n_iter=10,
    is_time_series=True,
    n_splits=3,
    log_to_mlflow=True
):
    logger.info("Starting model training...")

    best_model, best_score = None, float('-inf')
    best_model_name = None
    best_params = {}

    cv_strategy = TimeSeriesSplit(n_splits=n_splits) if is_time_series else n_splits

    for model_name, config in model_configs.items():
        try:
            logger.info(f"Training model: {model_name}")
            base_model = config["model"]
            param_grid = config.get("params", {})
            run_name = config.get("run_name", f"{model_name}_tuning")

            if search_type == "random":
                search = RandomizedSearchCV(base_model, param_distributions=param_grid, n_iter=n_iter,
                                            cv=cv_strategy, n_jobs=-1, verbose=1)
            else:
                search = GridSearchCV(base_model, param_grid=param_grid,
                                      cv=cv_strategy, n_jobs=-1, verbose=1)

            search.fit(X_train, y_train)
            model = search.best_estimator_

            table = evaluate_and_log_model(
                model, X_train, y_train, X_val, y_val,
                model_name, run_name, log=log_to_mlflow
            )
            print(table)

            # Track best model by validation R2
            y_val_pred = model.predict(X_val)
            val_r2 = r2_score(np.expm1(y_val), np.expm1(y_val_pred))

            if val_r2 > best_score:
                best_model = model
                best_model_name = model_name
                best_score = val_r2
                best_params = search.best_params_

        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")

    logger.info(f"Best model: {best_model_name} with R2: {best_score}")
    return best_model, best_model_name, best_params

def run_experiment():
    X_train = load_csv(X_TRAIN_FILE)
    y_train = load_csv(Y_TRAIN_FILE).squeeze()
    X_val = load_csv(X_VAL_FILE)
    y_val = load_csv(Y_VAL_FILE).squeeze()

    model_configs = {
        "XGBoost": {
            "model": XGBRegressor(),
            "params": {
                'n_estimators': [100, 300, 500],
                'alpha': [0.1, 10, 100],
                'learning_rate': [0.1, 1.0]
            },
            "run_name": "XGBoost_Tuning"
        },
        "LightGBM": {
            "model": LGBMRegressor(verbose=-1),
            "params": {
                'alpha': [0.1, 10, 100],
                'num_leaves': [20, 31, 40],
                'learning_rate': [0.1, 1.0],
                'n_estimators': [100, 300, 500]
            },
            "run_name": "LightGBM_Tuning"
        }
    }

    best_model, best_model_name, best_params = train_models_with_search(
        X_train, y_train, X_val, y_val,
        model_configs=model_configs,
        search_type="random",
        n_iter=20,
        is_time_series=True,
        n_splits=5,
        log_to_mlflow=True
    )

    return best_model, best_model_name, best_params



def train_final_model(best_model):
    X_train = load_csv(X_TRAIN_FILE)
    y_train = load_csv(Y_TRAIN_FILE).squeeze()
    X_val = load_csv(X_VAL_FILE)
    y_val = load_csv(Y_VAL_FILE).squeeze()

    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])

    best_model.fit(X_train_val, y_train_val)
    logger.info("Final model retrained on train + validation set.")

    return best_model


# ----------------- Main Execution -------------------

def main():
    best_model, best_model_name, best_params = run_experiment()
    final_model = train_final_model(best_model)

if __name__ == "__main__":
    main()

