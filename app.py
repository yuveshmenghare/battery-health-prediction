import streamlit as st
import pandas as pd
import joblib

# Load trained model and scaler
model = joblib.load("battery_rf_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Battery Health Predictor", layout="centered")

st.title("🔋 Smartphone Battery Health Predictor")

# -------------------------------
# USER INPUTS
# -------------------------------

device_age = st.slider("Device Age (months)", 0, 60, 24)
battery_capacity = st.number_input("Battery Capacity (mAh)", 1000, 6000, 4000)

screen_time = st.slider("Screen On Hours/Day", 0.0, 12.0, 5.0)
charging_cycles = st.slider("Charging Cycles/Week", 0, 30, 10)
temp = st.slider("Battery Temperature (°C)", 20, 50, 30)

fast_charge = st.slider("Fast Charging Usage (%)", 0, 100, 50)
overnight = st.slider("Overnight Charging (per week)", 0, 7, 3)

gaming = st.slider("Gaming Hours/Week", 0, 40, 10)
video = st.slider("Streaming Hours/Week", 0, 40, 10)

charging_score = st.slider("Charging Habit Score", 0.0, 1.0, 0.5)
usage_score = st.slider("Usage Intensity Score", 0.0, 1.0, 0.5)
thermal_index = st.slider("Thermal Stress Index", 0.0, 1.0, 0.5)

# categorical inputs
bg_usage = st.selectbox("Background App Usage", ["low", "medium", "high"])
signal = st.selectbox("Signal Strength", ["low", "medium", "high"])

# encoding mapping (must match training)
mapping = {"low": 0, "medium": 1, "high": 2}

# -------------------------------
# PREDICTION
# -------------------------------

if st.button("Predict"):

    input_data = pd.DataFrame([{
        'device_age_months': device_age,
        'battery_capacity_mah': battery_capacity,
        'avg_screen_on_hours_per_day': screen_time,
        'avg_charging_cycles_per_week': charging_cycles,
        'avg_battery_temp_celsius': temp,
        'fast_charging_usage_percent': fast_charge,
        'overnight_charging_freq_per_week': overnight,
        'gaming_hours_per_week': gaming,
        'video_streaming_hours_per_week': video,
        'background_app_usage_level': mapping[bg_usage],
        'signal_strength_avg': mapping[signal],
        'charging_habit_score': charging_score,
        'usage_intensity_score': usage_score,
        'thermal_stress_index': thermal_index
    }])

    # Ensure correct column order
    input_data = input_data[[
        'device_age_months',
        'battery_capacity_mah',
        'avg_screen_on_hours_per_day',
        'avg_charging_cycles_per_week',
        'avg_battery_temp_celsius',
        'fast_charging_usage_percent',
        'overnight_charging_freq_per_week',
        'gaming_hours_per_week',
        'video_streaming_hours_per_week',
        'background_app_usage_level',
        'signal_strength_avg',
        'charging_habit_score',
        'usage_intensity_score',
        'thermal_stress_index'
    ]]

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    pred_value = model.predict(input_scaled)[0]

    # Clamp output (0–100)
    # pred_value = max(0, min(100, pred))

    # -------------------------------
    # DISPLAY RESULTS
    # -------------------------------

    st.success(f"🔋 Predicted Battery Health: {round(pred_value, 2)}%")

    # Recommended action
    if pred_value >= 75:
        action = "✅ Keep Using"
    elif pred_value >= 50:
        action = "⚠️ Replace Battery"
    else:
        action = "❌ Change Phone"

    st.info(f"📌 Recommended Action: {action}")

    