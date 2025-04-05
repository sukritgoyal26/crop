import streamlit as st
import numpy as np
import joblib
from keras.models import load_model
from tfkan.layers import DenseKAN

# Load model and preprocessing tools
model = load_model("crop_recommender_model.keras", custom_objects={'DenseKAN': DenseKAN})
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# App title
st.title("🌾 Crop Recommendation System")
st.markdown("Enter soil and weather conditions to get the best crop recommendation.")

# Input form
with st.form("input_form"):
    N = st.number_input("Nitrogen (N)", min_value=0.0)
    P = st.number_input("Phosphorus (P)", min_value=0.0)
    K = st.number_input("Potassium (K)", min_value=0.0)
    temperature = st.number_input("Temperature (°C)")
    humidity = st.number_input("Humidity (%)")
    ph = st.number_input("Soil pH")
    rainfall = st.number_input("Rainfall (mm)")

    submitted = st.form_submit_button("Recommend Crop")

# Predict and display result
if submitted:
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    input_scaled = scaler.transform(input_data)
    prediction_prob = model.predict(input_scaled)
    predicted_idx = np.argmax(prediction_prob, axis=1)[0]
    recommended_crop = label_encoder.inverse_transform([predicted_idx])[0]
    
    st.success(f"🌱 **Recommended Crop:** {recommended_crop}")
