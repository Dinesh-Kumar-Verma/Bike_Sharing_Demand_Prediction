import pandas as pd
import joblib
from pathlib import Path
from src.utils.logger import get_logger

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
        logger.info("DataFrame saved successfully to: %s", str(path))
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
        logger.info("Model saved successfully to: %s", str(path))
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
