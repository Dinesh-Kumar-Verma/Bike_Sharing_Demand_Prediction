import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from src.utils.logger import get_logger
from src.utils.config import (
    INTERIM_DATA_DIR, TIMESERIES_FOLDS_DIR,
    X_TRAIN_FILE, Y_TRAIN_FILE, X_VAL_FILE, Y_VAL_FILE
)
import mlflow
import mlflow.sklearn

logger = get_logger(name='train_model', log_file='train_model.log')


def load_data(path_X, path_y):
    X = pd.read_csv(path_X)
    y = pd.read_csv(path_y).squeeze()  # Flatten y if needed
    return X, y


def evaluate_model(model, X_val, y_val):
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    r2 = r2_score(y_val, preds)
    mae = mean_absolute_error(y_val, preds)

    logger.info("Validation RMSE: %.4f", rmse)
    logger.info("Validation R2 Score: %.4f", r2)
    logger.info("Validation MAE: %.4f", mae)

    return rmse, r2, mae


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

            search = RandomizedSearchCV(
                model, param_distributions=param_grid,
                n_iter=n_iter, scoring='neg_root_mean_squared_error', cv=3, verbose=1
            )
            search.fit(X_train, y_train)

            logger.info("Best Params for %s in %s: %s", name, fold_dir.name, search.best_params_)

            rmse, r2, mae = evaluate_model(search.best_estimator_, X_val, y_val)

            # Log to MLflow
            with mlflow.start_run(run_name=f"{name}_{fold_dir.name}"):
                mlflow.log_params(search.best_params_)
                mlflow.log_metric("RMSE", rmse)
                mlflow.log_metric("R2_Score", r2)
                mlflow.log_metric("MAE", mae)
                mlflow.sklearn.log_model(
                    search.best_estimator_,
                    name=f"{name}_{fold_dir.name}",
                    input_example=X_train.head(4),
                )

            results.append({
                "fold": fold_dir.name,
                "model": name,
                "params": search.best_params_,
                "rmse": rmse,
                "r2_score": r2,
                "mae": mae
            })

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
            n_iter=n_iter, scoring='neg_root_mean_squared_error', cv=3, verbose=1
        )
        search.fit(X_train, y_train)

        logger.info("Best Params for %s: %s", name, search.best_params_)

        rmse, r2, mae = evaluate_model(search.best_estimator_, X_val, y_val)

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
        "Ridge": Ridge(),
        "Lasso": Lasso()
    }
    param_grids = {
        "RandomForest": {
            "n_estimators": [200, 250, 300],
            "max_depth": [20, 30, 40]
        },
        "Ridge": {"alpha": [0.01, 0.1, 1.0]},
        "Lasso": {"alpha": [0.01, 0.1, 1.0]}
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
