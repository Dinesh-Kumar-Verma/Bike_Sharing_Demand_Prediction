from src.utils.logger import get_logger
from src.components.data_preprocessing import DataPreprocessor
import pandas as pd
import numpy as np
import joblib
from src.utils.config import FINAL_MODEL_FILE

logger = get_logger('model_prediction')

def model_prediction(Input_dict:dict):
    logger.info('Starting model_prediction...')
    processor = DataPreprocessor()
    processor.load_pipeline()

    # Preprocess incoming data (e.g., from API or Streamlit form)
    X_input = pd.DataFrame([Input_dict])  # input_dict = your input features
    X_transformed = processor.transform(X_input)

    # Predict using model
    model = joblib.load(FINAL_MODEL_FILE)  # your trained model
    y_pred = model.predict(X_transformed, predict_disable_shape_check=True)
    y_pred_original = np.expm1(y_pred)
    logger.info(f'y_pred: {y_pred_original}')
    logger.info('Completed model_prediction.')

if __name__ == '__main__':
    Input_dict = {
        'Date': str('30/04/2018'),
        'Hour': 20,
        'Temperature(°C)': 20.0,
        'Humidity(%)': 69,
        'Wind speed (m/s)': 2.4,
        'Visibility (10m)': 533,
        'Dew point temperature(°C)': 14.1,
        'Solar Radiation (MJ/m2)': 0.6,
        'Rainfall(mm)': 0.5,
        'Snowfall (cm)': 0.3,
        'Seasons': 'Spring',
        'Holiday': 'No Holiday',
        'Functioning Day': 'Yes'
    }
    
    model_prediction(Input_dict)
