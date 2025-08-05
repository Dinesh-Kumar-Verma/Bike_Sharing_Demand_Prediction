import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from src.components.data_preprocessing import DataPreprocessor
from src.utils.config import FINAL_MODEL_FILE

def predict_bike_demand(input_dict: dict) -> np.ndarray:
    """
    Predicts bike demand based on input features.

    Parameters:
    ----------
    input_dict : dict
        Dictionary containing feature names and corresponding values.

    Returns:
    -------
    np.ndarray
        Predicted bike demand in original scale.
    """
    try:
        # Load preprocessing pipeline
        processor = DataPreprocessor()
        processor.load_pipeline()

        # Convert input to DataFrame
        X_input = pd.DataFrame([input_dict])

        # Apply preprocessing transformations
        X_transformed = processor.transform(X_input)

        # Load the final trained model
        model = joblib.load(FINAL_MODEL_FILE)

        # Make prediction and inverse transform
        y_pred = model.predict(X_transformed, predict_disable_shape_check=True)
        y_pred_original = np.expm1(y_pred)

        return y_pred_original

    except Exception as e:
        raise RuntimeError(f"Error during prediction pipeline: {e}")
