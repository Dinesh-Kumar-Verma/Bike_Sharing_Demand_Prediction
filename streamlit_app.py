import streamlit as st
from app.utils import load_model, load_pipeline, preprocess_input
import numpy as np

st.set_page_config(page_title="Bike Demand Predictor", layout="centered")

st.title("🚲 Bike Sharing Demand Prediction")
st.markdown("Predict the expected number of bike rentals based on input conditions.")

# Load model and pipeline
model = load_model()
pipeline = load_pipeline()

# --- UI Input ---
season = st.selectbox("Season", [1, 2, 3, 4])
holiday = st.selectbox("Is Holiday?", [0, 1])
workingday = st.selectbox("Is Working Day?", [0, 1])
weather = st.selectbox("Weather", [1, 2, 3, 4])
temp = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, value=22.0)
atemp = st.number_input("Feels Like Temperature (°C)", min_value=-10.0, max_value=50.0, value=24.0)
humidity = st.slider("Humidity (%)", 0, 100, 60)
windspeed = st.slider("Windspeed", 0.0, 100.0, 10.0)
hour = st.slider("Hour of Day", 0, 23, 9)

if st.button("Predict Demand"):
    user_input = {
        "season": season,
        "holiday": holiday,
        "workingday": workingday,
        "weather": weather,
        "temp": temp,
        "atemp": atemp,
        "humidity": humidity,
        "windspeed": windspeed,
        "hour": hour
    }
    
    X = preprocess_input(user_input, pipeline)
    pred = model.predict(X)
    bike_count = int(np.expm1(pred[0]))  # reverse log1p

    st.success(f"🔮 Estimated bike demand: **{bike_count} bikes**")


import joblib
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_PATH = Path("artifacts/models/LightGBM_final_model.pkl")
PIPELINE_PATH = Path("artifacts/transformers/preprocessing_pipeline.pkl")

def load_model():
    return joblib.load(MODEL_PATH)

def load_pipeline():
    return joblib.load(PIPELINE_PATH)

def preprocess_input(user_input: dict, pipeline):
    df = pd.DataFrame([user_input])
    return pipeline.transform(df)
