import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from src.components.data_preprocessing import DataPreprocessor
from src.utils.config import MODELS_DIR
from src.utils.params import load_params

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

        # Load the model name from params.yaml
        params = load_params()
        model_name = params['prediction']['model_name']
        model_path = MODELS_DIR / model_name

        # Load the final trained model
        model = joblib.load(model_path)

        # Make prediction and inverse transform
        y_pred = model.predict(X_transformed, predict_disable_shape_check=True)
        y_pred_original = np.expm1(y_pred)

        return y_pred_original

    except Exception as e:
        raise RuntimeError(f"Error during prediction pipeline: {e}")
