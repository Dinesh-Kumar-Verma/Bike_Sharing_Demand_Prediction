import pandas as pd
import joblib
from pathlib import Path
from src.utils.logger import get_logger
from src.utils.config import BASE_DIR
from src.utils.config import (
    INTERIM_DATA_DIR,
    X_TRAIN_FILE, Y_TRAIN_FILE,
    X_VAL_FILE, Y_VAL_FILE,
    X_TEST_FILE, Y_TEST_FILE
)


logger = get_logger('data_saver')

def save_csv(dataframe: pd.DataFrame, filepath, index: bool = False) -> None:
    """
    Save a pandas DataFrame to a CSV file.

    Args:
        dataframe (pd.DataFrame): The DataFrame to save.
        filepath (str or Path): The target CSV file path.
        index (bool): Whether to write row names (index). Default is False.
    """
    try:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(str(path), index=index)  #  str(path)
        rel_path = Path(path).relative_to(BASE_DIR)
        logger.info("DataFrame saved to: %s", rel_path)
        # logger.info("DataFrame saved successfully to: %s", str(path))
    except Exception as e:
        logger.exception("Failed to save DataFrame to: %s. Error: %s", str(path), e)
        raise

def save_model(model, filepath) -> None:
    """
    Save a machine learning model to a file using joblib.

    Args:
        model: The model object to save.
        filepath (str or Path): The target file path for the model.
    """
    try:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, str(path))  #  str(path)
        rel_path = Path(path).relative_to(BASE_DIR)
        logger.info("Model saved successfully to: %s", rel_path)
    except Exception as e:
        logger.exception("Failed to save model to: %s. Error: %s", str(path), e)
        raise

def save_text(data: str, filepath) -> None:
    """
    Save a string (text) to a text file.

    Args:
        data (str): The text data to save.
        filepath (str or Path): The target text file path.
    """
    try:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as file:  # 👈 pathlib’s open works fine
            file.write(data)
        logger.info("Text data saved successfully to: %s", str(path))
    except Exception as e:
        logger.exception("Failed to save text to: %s. Error: %s", str(path), e)
        raise

def save_split_data(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Save split datasets to CSV files in interim directory.
    """
    try:
        INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
        X_train.to_csv(X_TRAIN_FILE, index=False)
        y_train.to_csv(Y_TRAIN_FILE, index=False)
        X_val.to_csv(X_VAL_FILE, index=False)
        y_val.to_csv(Y_VAL_FILE, index=False)
        X_test.to_csv(X_TEST_FILE, index=False)
        y_test.to_csv(Y_TEST_FILE, index=False)

        logger.info("Saved Train/Val/Test splits to interim directory.")

    except Exception as e:
        logger.exception("Failed to save split data: %s", e)
        raise