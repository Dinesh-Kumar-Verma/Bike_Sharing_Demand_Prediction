import numpy as np
import joblib
import mlflow
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from prettytable import PrettyTable
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlflow.models.signature import infer_signature

from src.utils.logger import get_logger
from src.utils.data_loader import load_csv
from src.utils.config import X_TEST_PROCESSED_FILE, Y_TEST_TRANSFORMED_FILE, EVALUATION_DIR

logger = get_logger(name="model_evaluation", log_file="model_evaluation.log")


def load_final_model(model_dir: str = "artifacts/models", model_suffix: str = "final_model.pkl"):
    model_dir_path = Path(model_dir)
    final_model_path = next(model_dir_path.glob(f"*_{model_suffix}"))
    logger.info(f"Loading model from {final_model_path}")
    return joblib.load(final_model_path)


def log_evaluation_plots(y_true, y_pred):
    residuals = y_true - y_pred

    # Residuals Plot
    plt.figure(figsize=(8, 4))
    sns.histplot(residuals, bins=50, kde=True)
    plt.title("Residuals Distribution")
    plt.xlabel("Residuals")
    plt.savefig(EVALUATION_DIR / "residuals_plot.png")
    mlflow.log_artifact("residuals_plot.png")
    plt.close()

    # Actual vs Predicted
    plt.figure(figsize=(8, 4))
    sns.scatterplot(x=y_true, y=y_pred)
    plt.title("Actual vs Predicted")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.savefig(EVALUATION_DIR / "actual_vs_predicted.png")
    mlflow.log_artifact("actual_vs_predicted.png")
    plt.close()


def evaluate_model(model, X_test, y_test, is_log_transformed: bool = False):
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

    # Save PrettyTable summary to text file
    with open(EVALUATION_DIR / "evaluation_table.txt", "w") as f:
        f.write(str(table))

    # MLflow logging
    with mlflow.start_run(run_name="Test_Evaluation"):
        # Log metrics
        for k, v in metrics.items():
            mlflow.log_metric(f"test_{k.replace(' ', '_').lower()}", v)

        # Log model
        mlflow.sklearn.log_model(model, "model_evaluation_artifact",
                                 signature=infer_signature(X_test, model.predict(X_test)),
                                 input_example=X_test.head().astype(np.float64))

        # Log commit hash
        commit_hash = subprocess.getoutput("git rev-parse HEAD")
        mlflow.set_tag("git_commit", commit_hash)

        # Set additional tags
        mlflow.set_tags({
            "stage": "evaluation",
            "model_type": type(model).__name__,
            "framework": "scikit-learn"
        })

        # Log visualizations and table
        mlflow.log_artifact(EVALUATION_DIR / "evaluation_table.txt")
        log_evaluation_plots(y_test, y_pred)

    return metrics


def run_evaluation(model_path: str = None, x_path: str = None, y_path: str = None, is_log_transformed: bool = True):
    X_test = load_csv(x_path) if x_path else load_csv(X_TEST_PROCESSED_FILE)
    y_test = load_csv(y_path) if y_path else load_csv(Y_TEST_TRANSFORMED_FILE).squeeze()
    model = joblib.load(model_path) if model_path else load_final_model()

    return evaluate_model(model, X_test, y_test, is_log_transformed)


def main():
    _ = run_evaluation(is_log_transformed=True)


if __name__ == "__main__":
    main()
