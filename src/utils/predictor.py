import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from src.components.feature_cleaning import FeatureCleaner
from src.components.feature_engineering import FeatureEngineer
from src.components.feature_scaling import FeatureScaler
from src.utils.config import SCALER_PIPELINE_FILE, FINAL_MODEL_FILE, TEST_PURPOSE_FILE
from sklearn.pipeline import Pipeline
import joblib
import streamlit as st
# def predict_bike_demand(input_dict):
#     df = pd.DataFrame([input_dict])

#     # Apply preprocessing steps
#     raw_FeatureCleaner = FeatureCleaner()
#     df_cleaned = raw_FeatureCleaner.transform(df) 
#     fe = FeatureEngineer(run_vif=False)
#     df_engineered = fe.transform(df_cleaned)

#     # Load and apply scaler
#     scaler = FeatureScaler()
#     scaler.load_pipeline()
#     df_scaled = scaler.transform(df_engineered)

#     # Load model and predict
#     model = joblib.load(FINAL_MODEL_FILE)
#     y_pred_log = model.predict(df_scaled, pred_disable_shape_check=True)

#     # Inverse of log1p transformation
#     y_pred = np.expm1(y_pred_log)[0]
#     return y_pred

def predict_bike_demand(input_dict):
    st.write(" Raw input:")
    st.write(input_dict)

    df = pd.DataFrame([input_dict])
    st.write(" DataFrame created from input:")
    st.write(df)
    st.write("Shape:", df.shape)

    # Step 1: Feature Cleaning
    try:
        raw_FeatureCleaner = FeatureCleaner()
        df_cleaned = raw_FeatureCleaner.transform(df)
        st.write("After Feature Cleaning:")
        st.write(df_cleaned.head())
        st.write("Shape:", df_cleaned.shape)
    except Exception as e:
        st.error(f"Feature Cleaning Error: {e}")
        return

    # Step 2: Feature Engineering
    try:
        fe = FeatureEngineer(run_vif=False)
        df_engineered = fe.transform(df_cleaned, save_to=TEST_PURPOSE_FILE)
        st.write(" After Feature Engineering:")
        st.write(df_engineered.head())
        st.write("Shape:", df_engineered.shape)
    except Exception as e:
        st.error(f"Feature Engineering Error: {e}")
        return

    # Step 3: Feature Scaling
    try:
        scaler = FeatureScaler()
        scaler.load_pipeline()
        df_scaled = scaler.transform(df_engineered)
        st.write("After Feature Scaling:")
        st.write(df_scaled)
        st.write("Shape:", df_scaled.shape)
    except Exception as e:
        st.error(f" Feature Scaling Error: {e}")
        return

    # Step 4: Load Model & Predict
    try:
        model = joblib.load(FINAL_MODEL_FILE)
        y_pred_log = model.predict(df_scaled,  predict_disable_shape_check=True)
        st.write("Raw prediction (log-transformed):", y_pred_log)
    except Exception as e:
        st.error(f"Model Prediction Error: {e}")
        return

    try:
        y_pred = np.expm1(y_pred_log)[0]
        st.write("Final Prediction (original scale):", y_pred)
        return y_pred
    except Exception as e:
        st.error(f"Error converting log prediction to original scale: {e}")
        return
