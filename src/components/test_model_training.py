# second code - 232




import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from prettytable import PrettyTable
from sklearn.tree import DecisionTreeRegressor
from src.utils.logger import get_logger
from src.utils.config import (
    INTERIM_DATA_DIR, TIMESERIES_FOLDS_DIR,
    X_TRAIN_FILE, Y_TRAIN_FILE, X_VAL_FILE, Y_VAL_FILE
)
import mlflow
import mlflow.sklearn

import warnings
warnings.filterwarnings('ignore')  # Not recommended globally

# Better approach — suppress specific warning types:
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

logger = get_logger(name='train_model', log_file='train_model.log')


def load_data(path_X, path_y):
    X = pd.read_csv(path_X)
    y = pd.read_csv(path_y).squeeze()  # Flatten y if needed
    return X, y


def evaluate_and_log_model(model, X_train, y_train, X_val, y_val, model_name, run_name=None, is_logged=True):
    """
    Evaluates the model, logs metrics, and returns a formatted PrettyTable.
    Assumes y_train and y_val are log1p-transformed and need expm1 for real scale.
    """
    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    # Reverse transformation
    y_train_org = np.expm1(y_train)
    y_val_org = np.expm1(y_val)
    y_train_pred_org = np.expm1(y_train_pred)
    y_val_pred_org = np.expm1(y_val_pred)

    # Train Metrics
    train_MSE = mean_squared_error(y_train_org, y_train_pred_org)
    train_RMSE = np.sqrt(train_MSE)
    train_MAE = mean_absolute_error(y_train_org, y_train_pred_org)
    train_r2 = r2_score(y_train_org, y_train_pred_org)
    train_adj_r2 = 1 - (1 - train_r2) * ((X_train.shape[0] - 1) / (X_train.shape[0] - X_train.shape[1] - 1))

    # Val/Test Metrics
    val_MSE = mean_squared_error(y_val_org, y_val_pred_org)
    val_RMSE = np.sqrt(val_MSE)
    val_MAE = mean_absolute_error(y_val_org, y_val_pred_org)
    val_r2 = r2_score(y_val_org, y_val_pred_org)
    val_adj_r2 = 1 - (1 - val_r2) * ((X_val.shape[0] - 1) / (X_val.shape[0] - X_val.shape[1] - 1))

    # MLflow Logging (optional)
    if is_logged and run_name:
        with mlflow.start_run(run_name=run_name):
            mlflow.log_metric("Train_RMSE", train_RMSE)
            mlflow.log_metric("Train_R2", train_r2)
            mlflow.log_metric("Train_MAE", train_MAE)
            mlflow.log_metric("Val_RMSE", val_RMSE)
            mlflow.log_metric("Val_R2", val_r2)
            mlflow.log_metric("Val_MAE", val_MAE)
            mlflow.sklearn.log_model(model, model_name, input_example=X_train.head(2))

    # Logging table
    score_chart = PrettyTable()
    score_chart.field_names = ["Metrics", "Train Data", "Validation Data"]
    score_chart.add_row(["R2 Score", round(train_r2, 4), round(val_r2, 4)])
    score_chart.add_row(["Adjusted R2", round(train_adj_r2, 4), round(val_adj_r2, 4)])
    score_chart.add_row(["MSE", round(train_MSE, 4), round(val_MSE, 4)])
    score_chart.add_row(["RMSE", round(train_RMSE, 4), round(val_RMSE, 4)])
    score_chart.add_row(["MAE", round(train_MAE, 4), round(val_MAE, 4)])

    return {
        "train_r2": train_r2, "val_r2": val_r2,
        "train_rmse": train_RMSE, "val_rmse": val_RMSE,
        "train_mae": train_MAE, "val_mae": val_MAE,
        "table": score_chart
    }


def train_with_timeseries_folds(models, param_grids, n_iter=10):
    """
    Train and tune models using TimeSeriesSplit folds.
    """
    logger.info("Detected TimeSeriesSplit folds. Starting fold-wise training...")
    results = []

    for fold_dir in sorted(TIMESERIES_FOLDS_DIR.glob("fold_*")):
        logger.info("Training on %s", fold_dir.name)
        X_train, y_train = load_data(fold_dir / "X_train.csv", fold_dir / "y_train.csv")
        X_val, y_val = load_data(fold_dir / "X_val.csv", fold_dir / "y_val.csv")

        for name, model, param_grid in zip(models.keys(), models.values(), param_grids.values()):
            logger.info("Hyperparameter tuning for model: %s", name)
            tscv = TimeSeriesSplit(n_splits=3) 

            search = RandomizedSearchCV(
                model, param_distributions=param_grid,
                n_iter=n_iter, scoring='neg_root_mean_squared_error', cv=tscv, verbose=1
            )
            search.fit(X_train, y_train)

            logger.info("Best Params for %s in %s: %s", name, fold_dir.name, search.best_params_)

            metrics = evaluate_and_log_model(
                search.best_estimator_,
                X_train, y_train, X_val, y_val,
                model_name=f"{name}_{fold_dir.name}",  # or name+"_train_val"
                run_name=f"{name}_{fold_dir.name}",
                is_logged=True
                )


            logger.info("\n%s", metrics['table'])




    return results


def train_with_train_val(models, param_grids, n_iter=10):
    """
    Train and tune models using the standard Train/Validation split.
    """
    logger.info("Using Train/Validation split (fallback).")
    X_train, y_train = load_data(X_TRAIN_FILE, Y_TRAIN_FILE)
    X_val, y_val = load_data(X_VAL_FILE, Y_VAL_FILE)

    results = []

    for name, model, param_grid in zip(models.keys(), models.values(), param_grids.values()):
        logger.info("Hyperparameter tuning for model: %s", name)

        search = RandomizedSearchCV(
            model, param_distributions=param_grid,
            n_iter=n_iter, scoring='neg_root_mean_squared_error', verbose=1
        )
        search.fit(X_train, y_train)

        logger.info("Best Params for %s: %s", name, search.best_params_)

        #rmse, r2, mae = evaluate_model(search.best_estimator_, X_val, y_val)

        # Log to MLflow
        with mlflow.start_run(run_name=f"{name}_train_val"):
            mlflow.log_params(search.best_params_)
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("R2_Score", r2)
            mlflow.log_metric("MAE", mae)
            mlflow.sklearn.log_model(search.best_estimator_, f"{name}_train_val")

        results.append({
            "model": name,
            "params": search.best_params_,
            "rmse": rmse,
            "r2_score": r2,
            "mae": mae
        })

    return results


def main():
    logger.info("-" * 80)
    logger.info("Starting Model Training Pipeline...")

    # Define models and parameter grids
    models = {
        "RandomForest": RandomForestRegressor(),
        # "Ridge": Ridge(),
        # "Lasso": Lasso(), 
        "DecisionTree": DecisionTreeRegressor()
    }
    param_grids = {
        "RandomForest": {
            "n_estimators": [200, 250, 300],
            "max_depth": [20, 30, 40]
        },
        # "Ridge": {"alpha": [0.01, 0.1, 1.0]},
        # "Lasso": {"alpha": [0.01, 0.1, 1.0]},
        "DecisionTree": {
            "max_depth": [10, 20, None],
            "min_samples_leaf": [2, 5, 10]
            }
    }

    try:
        if TIMESERIES_FOLDS_DIR.exists() and any(TIMESERIES_FOLDS_DIR.iterdir()):
            logger.info("TimeSeriesSplit directory found: %s", TIMESERIES_FOLDS_DIR)
            results = train_with_timeseries_folds(models, param_grids)
        else:
            logger.warning("TimeSeriesSplit directory NOT found. Falling back to Train/Val split.")
            results = train_with_train_val(models, param_grids)

        logger.info("Model Training Pipeline completed successfully.")
        summary_df = pd.DataFrame(results)
        logger.info("\nModel Performance Summary:\n%s", summary_df.to_string(index=False))

        # Save summary to CSV
        summary_csv = INTERIM_DATA_DIR / "model_performance_summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        logger.info("Saved results summary to %s", summary_csv)

    except Exception as e:
        logger.exception("Model Training Pipeline failed: %s", e)

    logger.info("-" * 80)


if __name__ == "__main__":
    main()








#************************************************************************************************************************

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from prettytable import PrettyTable
from sklearn.tree import DecisionTreeRegressor
from src.utils.logger import get_logger
from src.utils.config import (
    INTERIM_DATA_DIR, TIMESERIES_FOLDS_DIR,
    X_TRAIN_FILE, Y_TRAIN_FILE, X_VAL_FILE, Y_VAL_FILE
)
import mlflow
import mlflow.sklearn

import warnings
warnings.filterwarnings('ignore')  # Not recommended globally

# Better approach — suppress specific warning types:
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

logger = get_logger(name='train_model', log_file='train_model.log')


def load_data(path_X, path_y):
    X = pd.read_csv(path_X)
    y = pd.read_csv(path_y).squeeze()  # Flatten y if needed
    return X, y


def evaluate_and_log_model(model, X_train, y_train, X_val, y_val, model_name, run_name=None, is_logged=True):
    """
    Evaluates the model, logs metrics, and returns a formatted PrettyTable.
    Assumes y_train and y_val are log1p-transformed and need expm1 for real scale.
    """
    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    # Reverse transformation
    y_train_org = np.expm1(y_train)
    y_val_org = np.expm1(y_val)
    y_train_pred_org = np.expm1(y_train_pred)
    y_val_pred_org = np.expm1(y_val_pred)

    # Train Metrics
    train_MSE = mean_squared_error(y_train_org, y_train_pred_org)
    train_RMSE = np.sqrt(train_MSE)
    train_MAE = mean_absolute_error(y_train_org, y_train_pred_org)
    train_r2 = r2_score(y_train_org, y_train_pred_org)
    train_adj_r2 = 1 - (1 - train_r2) * ((X_train.shape[0] - 1) / (X_train.shape[0] - X_train.shape[1] - 1))

    # Val/Test Metrics
    val_MSE = mean_squared_error(y_val_org, y_val_pred_org)
    val_RMSE = np.sqrt(val_MSE)
    val_MAE = mean_absolute_error(y_val_org, y_val_pred_org)
    val_r2 = r2_score(y_val_org, y_val_pred_org)
    val_adj_r2 = 1 - (1 - val_r2) * ((X_val.shape[0] - 1) / (X_val.shape[0] - X_val.shape[1] - 1))

    # MLflow Logging
    if is_logged and run_name:
        with mlflow.start_run(run_name=run_name):
            mlflow.log_metric("Train_RMSE", train_RMSE)
            mlflow.log_metric("Train_R2", train_r2)
            mlflow.log_metric("Train_MAE", train_MAE)
            mlflow.log_metric("Val_RMSE", val_RMSE)
            mlflow.log_metric("Val_R2", val_r2)
            mlflow.log_metric("Val_MAE", val_MAE)
            mlflow.sklearn.log_model(model= model, name= model_name, input_example=X_train.head(2))

    # Logging table
    score_chart = PrettyTable()
    score_chart.field_names = ["Metrics", "Train Data", "Validation Data"]
    score_chart.add_row(["R2 Score", round(train_r2, 4), round(val_r2, 4)])
    score_chart.add_row(["Adjusted R2", round(train_adj_r2, 4), round(val_adj_r2, 4)])
    score_chart.add_row(["MSE", round(train_MSE, 4), round(val_MSE, 4)])
    score_chart.add_row(["RMSE", round(train_RMSE, 4), round(val_RMSE, 4)])
    score_chart.add_row(["MAE", round(train_MAE, 4), round(val_MAE, 4)])

    return {
        "train_r2": train_r2, "val_r2": val_r2,
        "train_rmse": train_RMSE, "val_rmse": val_RMSE,
        "train_mae": train_MAE, "val_mae": val_MAE,
        "table": score_chart
    }


def train_with_train_val(models, param_grids, n_iter=10):
    """
    Train and tune models using the standard Train/Validation split.
    Returns performance metrics and logs all artifacts using MLflow.
    """
    logger.info("Using Train/Validation split (fallback mode).")
    X_train, y_train = load_data(X_TRAIN_FILE, Y_TRAIN_FILE)
    X_val, y_val = load_data(X_VAL_FILE, Y_VAL_FILE)

    results = []

    for (name, model), param_grid in zip(models.items(), param_grids.values()):
        logger.info("Hyperparameter tuning for model: %s", name)

        search = RandomizedSearchCV(
            model,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring='neg_root_mean_squared_error',
            verbose=1,
            n_jobs=-1,
            cv=3,
            random_state=42
        )
        search.fit(X_train, y_train)
        logger.info("Best Params for %s: %s", name, search.best_params_)
        logger.info("Best Cross-Validated Score (Neg RMSE): %.4f", search.best_score_)

        # Evaluate and log with MLflow
        eval_result = evaluate_and_log_model(
            model=search.best_estimator_,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            model_name=f"{name}_train_val",
            run_name=f"{name}_train_val",
            is_logged=True
        )

        results.append({
            "model": name,
            "params": search.best_params_,
            "train_rmse": eval_result["train_rmse"],
            "val_rmse": eval_result["val_rmse"],
            "train_r2": eval_result["train_r2"],
            "val_r2": eval_result["val_r2"],
            "mae": eval_result["val_mae"]
        })

        logger.info("\nEvaluation Metrics for %s:\n%s", name, eval_result["table"])

    return results


def main():
    logger.info("-" * 80)
    logger.info("Starting Model Training Pipeline...")

    # Define models and parameter grids
    models = {
        "RandomForest": RandomForestRegressor(),
        # "Ridge": Ridge(),
        # "Lasso": Lasso(), 
        "DecisionTree": DecisionTreeRegressor()
    }
    param_grids = {
        "RandomForest": {
            "n_estimators": [250, 300],
            "max_depth": [30, 40]
        },
        # "Ridge": {"alpha": [0.01, 0.1, 1.0]},
        # "Lasso": {"alpha": [0.01, 0.1, 1.0]},
        "DecisionTree": {
            "max_depth": [20, None],
            "min_samples_leaf": [5, 10]
            }
    }

    try:
        results = train_with_train_val(models, param_grids)

        logger.info("Model Training Pipeline completed successfully.")
        summary_df = pd.DataFrame(results)
        logger.info("\nModel Performance Summary:\n%s", summary_df.to_string(index=False))

        # Save summary to CSV
        summary_csv = INTERIM_DATA_DIR / "model_performance_summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        logger.info("Saved results summary to %s", summary_csv)

    except Exception as e:
        logger.exception("Model Training Pipeline failed: %s", e)

    logger.info("-" * 80)


if __name__ == "__main__":
    main()
