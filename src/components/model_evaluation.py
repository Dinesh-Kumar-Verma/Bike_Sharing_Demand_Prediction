import numpy as np
from prettytable import PrettyTable
from src.utils.logger import get_logger
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
from src.utils.data_loader import load_csv
from src.utils.config import X_TEST_FILE, Y_TEST_FILE
from pathlib import Path
import joblib

logger = get_logger(name="model_evaluation", log_file="model_evaluation.log")

def evaluate_model(model, X_test, y_test, is_log_transformed=False):
    """
    Evaluate the model and return metrics in both transformed and actual scales.

    Args:
        model: Trained ML model
        X_test: Test features
        y_test: Test target values
        is_log_transformed (bool): Whether target was log1p-transformed during training.

    Returns:
        dict: Dictionary of evaluation metrics.
    """
    y_pred = model.predict(X_test)

    if is_log_transformed:
        # Inverse transform the predictions and targets
        y_pred_actual = np.expm1(y_pred)
        y_test_actual = np.expm1(y_test)

        logger.info("Inverse transforming predictions and targets using expm1.")
    else:
        y_pred_actual = y_pred
        y_test_actual = y_test

    # Metrics in actual scale
    r2 = r2_score(y_test_actual, y_pred_actual)
    adj_r2 = 1 - (1 - r2) * (len(y_test_actual) - 1) / (len(y_test_actual) - X_test.shape[1] - 1)
    mse = mean_squared_error(y_test_actual, y_pred_actual)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_actual, y_pred_actual)

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

    # Log to MLflow
    with mlflow.start_run(run_name="Test_Evaluation"):
        for k, v in metrics.items():
            mlflow.log_metric(f"test_{k.replace(' ', '_').lower()}", v)

    return metrics

def load_final_model(model_dir="artifacts/models", model_suffix="final_model.pkl"):
    model_dir_path = Path(model_dir)
    final_model_path = next(model_dir_path.glob(f"*_{model_suffix}"))
    logger.info(f"Loading model from {final_model_path}")
    return joblib.load(final_model_path)

def main():
    X_test = load_csv(X_TEST_FILE)
    y_test = load_csv(Y_TEST_FILE).squeeze()

    final_model = load_final_model()

    test_metrics = evaluate_model(
        model=final_model,
        X_test=X_test,
        y_test=y_test,
        is_log_transformed=True  # Adjust if your training used log1p transformation
    )

if __name__ == "__main__":
    main()
