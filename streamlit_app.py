import streamlit as st
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Bike Demand Predictor", layout="centered")

st.title("🚲 Bike Sharing Demand Prediction")
st.markdown("Bike Sharing Demand Prediction")

# # Load model and pipeline
# model = load_model()
# pipeline = load_pipeline()

with st.form("input_form"):
    date = st.date_input("Date", value=datetime.today())
    hour = st.slider("Hour of Day", 0, 23)
    temperature = st.number_input("Temperature (°C)", value=20.0)
    humidity = st.number_input("Humidity (%)", value=60.0)
    wind_speed = st.number_input("Wind Speed (m/s)", value=2.0)
    visibility = st.number_input("Visibility (10m)", value=2000.0)
    dew_point_temp = st.number_input("Dew Point Temperature (°C)", value=10.0)
    solar_radiation = st.number_input("Solar Radiation (MJ/m2)", value=0.0)
    rainfall = st.number_input("Rainfall (mm)", value=0.0)
    snowfall = st.number_input("Snowfall (cm)", value=0.0)
    seasons = st.selectbox("Season", ["Spring", "Summer", "Autumn", "Winter"])
    holiday = st.selectbox("Holiday", ["No Holiday", "Holiday"])
    functioning_day = st.selectbox("Functioning Day", ["Yes", "No"])
    
    submitted = st.form_submit_button("Predict")

if submitted:
    input_dict = {
        'Date': str(date),
        'Hour': hour,
        'Temperature(°C)': temperature,
        'Humidity(%)': humidity,
        'Wind speed (m/s)': wind_speed,
        'Visibility (10m)': visibility,
        'Dew point temperature(°C)': dew_point_temp,
        'Solar Radiation (MJ/m2)': solar_radiation,
        'Rainfall(mm)': rainfall,
        'Snowfall (cm)': snowfall,
        'Seasons': seasons,
        'Holiday': holiday,
        'Functioning Day': functioning_day
    }

    # prediction = predict_bike_demand(input_dict)
    st.success(f"🔮 Predicted Rented Bike Count: **{int(prediction)}**")
    

 


