import streamlit as st
import numpy as np
from datetime import datetime
from src.utils.predictor import predict_bike_demand

def main():
    # ----------------------------- Page Config --------------------------------
    st.set_page_config(
        page_title="🚲 Bike Demand Predictor",
        layout="centered",
        page_icon="🚴"
    )

    # ----------------------------- Custom CSS Styling -------------------------
    st.markdown(
        """
        <style>
            .main {
                background-color: #f4f6f8;
                padding: 2rem;
                border-radius: 10px;
            }
            .stButton>button {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                height: 3em;
                width: 100%;
                border-radius: 8px;
            }
            .stTitle {
                color: #0072B5;
            }
            .footer {
                margin-top: 3rem;
                text-align: center;
                color: gray;
                font-size: 0.9rem;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # ----------------------------- Title & Intro ------------------------------
    st.title("🚲 Bike Sharing Demand Prediction")
    st.markdown("### Predict how many bikes will be rented based on the weather and day conditions.")
    st.info("This model uses machine learning to forecast bike rental demand. Fill in the inputs and hit **Predict** to get started!", icon="ℹ️")

    # ----------------------------- Input Form ---------------------------------
    with st.form("input_form"):
        st.subheader("🧾 Input Features")

        col1, col2 = st.columns(2)

        with col1:
            date = st.date_input("📅 Date", value=datetime.today())
            hour = st.slider("⏰ Hour of Day", 0, 23)
            temperature = st.number_input("🌡️ Temperature (°C)", value=20.0)
            humidity = st.number_input("💧 Humidity (%)", value=60.0)
            wind_speed = st.number_input("🌬️ Wind Speed (m/s)", value=2.0)
            visibility = st.number_input("👀 Visibility (10m)", value=2000.0)

        with col2:
            dew_point_temp = st.number_input("🌫️ Dew Point Temperature (°C)", value=10.0)
            solar_radiation = st.number_input("☀️ Solar Radiation (MJ/m²)", value=0.0)
            rainfall = st.number_input("🌧️ Rainfall (mm)", value=0.0)
            snowfall = st.number_input("❄️ Snowfall (cm)", value=0.0)
            seasons = st.selectbox("🗓️ Season", ["Spring", "Summer", "Autumn", "Winter"])
            holiday = st.selectbox("🏖️ Holiday", ["No Holiday", "Holiday"])
            functioning_day = st.selectbox("🏢 Functioning Day", ["Yes", "No"])

        submitted = st.form_submit_button("📊 Predict Bike Demand")

    # ----------------------------- Prediction Logic ---------------------------
    if submitted:
        input_dict = {
            'Date': date.strftime("%d/%m/%Y"),
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

        try:
            prediction = predict_bike_demand(input_dict)

            st.markdown("---")
            st.subheader("📈 Prediction Result")
            if prediction is None:
                st.error("Prediction failed. Model returned None.")
            else:
                st.markdown(
                    f"""
                    <div style='text-align: center; padding: 20px;'>
                        <h1 style='font-size: 60px; color: #0072B5; font-weight: bold;'>
                            {int(prediction[0]):,} 🚴
                        </h1>
                        <h3>Predicted Rented Bike Count</h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        except Exception as e:
            st.error(f"🚨 Error during prediction: {e}")

    # ----------------------------- Footer -------------------------------------
    st.markdown("---")
    st.markdown('<div class="footer">Built with ❤️ by Dinesh | Powered by Machine Learning</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
