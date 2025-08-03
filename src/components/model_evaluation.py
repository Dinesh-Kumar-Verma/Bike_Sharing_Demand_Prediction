import numpy as np
import joblib
import mlflow
from pathlib import Path
from prettytable import PrettyTable
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from src.utils.logger import get_logger
from src.utils.data_loader import load_csv
from src.utils.config import X_TEST_FILE, Y_TEST_FILE

logger = get_logger(name="model_evaluation", log_file="model_evaluation.log")

def load_final_model(model_dir: str = "artifacts/models", model_suffix: str = "final_model.pkl"):
    """
    Load the final trained model from a specified directory.
    
    Args:
        model_dir (str): Directory where model is saved.
        model_suffix (str): Suffix of the model filename.

    Returns:
        model: Trained machine learning model.
    """
    model_dir_path = Path(model_dir)
    final_model_path = next(model_dir_path.glob(f"*_{model_suffix}"))
    logger.info(f"Loading model from {final_model_path}")
    return joblib.load(final_model_path)

def evaluate_model(model, X_test, y_test, is_log_transformed: bool = False):
    """
    Evaluate a trained model and log metrics using MLflow.

    Args:
        model: Trained model object.
        X_test: Test features.
        y_test: True target values.
        is_log_transformed (bool): Whether target variable was log1p-transformed during training.

    Returns:
        dict: Evaluation metrics (R2, Adjusted R2, MSE, RMSE, MAE).
    """
    logger.info("Starting evaluation of the model...")

    y_pred = model.predict(X_test)

    if is_log_transformed:
        y_pred = np.expm1(y_pred)
        y_test = np.expm1(y_test)
        logger.info("Inverse transforming predictions and targets using expm1.")

    r2 = r2_score(y_test, y_pred)
    adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    metrics = {
        "R2 Score": round(r2, 4),
        "Adjusted R2": round(adj_r2, 4),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
    }

    table = PrettyTable()
    table.field_names = ["Metric", "Test (Actual Scale)"]
    for k, v in metrics.items():
        table.add_row([k, v])

    logger.info(f"Test set evaluation metrics:\n{table}")

    with mlflow.start_run(run_name="Test_Evaluation"):
        for k, v in metrics.items():
            mlflow.log_metric(f"test_{k.replace(' ', '_').lower()}", v)

    return metrics

def run_evaluation(model_path: str = None, x_path: str = None, y_path: str = None, is_log_transformed: bool = True):
    """
    High-level API for running model evaluation.

    Args:
        model_path (str): Path to the saved model. If None, loads from default.
        x_path (str): Path to X_test data. If None, loads from default.
        y_path (str): Path to y_test data. If None, loads from default.
        is_log_transformed (bool): Whether target variable was log1p-transformed.

    Returns:
        dict: Evaluation metrics.
    """
    X_test = load_csv(X_TEST_FILE)
    y_test = load_csv(Y_TEST_FILE).squeeze()
    model = joblib.load(model_path) if model_path else load_final_model()

    return evaluate_model(model, X_test, y_test, is_log_transformed)

def main():
    _ = run_evaluation(is_log_transformed=True)

if __name__ == "__main__":
    main()
