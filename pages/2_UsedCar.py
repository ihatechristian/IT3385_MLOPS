import pandas as pd
import streamlit as st
from pycaret.regression import load_model, predict_model

st.set_page_config(page_title="Used Car Price", page_icon="🚗", layout="wide")
st.title("🚗 Used Car Price Predictor")

# Load model
model = load_model("models/usedcar_price_model")

# --- Feature schema ---
numeric_features = [
    "Year",
    "Age",
    "Kilometers_Driven",
    "Log_Km",
    "Mileage",
    "Engine",
    "Power",
    "Seats"
]

categorical_features = [
    "Fuel_Type",
    "Transmission",
    "Owner_Type",
    "Location",
    "Brand_Model",
    "Brand"
]

target = "Price"

# All features in order for DataFrame
FEATURES = numeric_features + categorical_features + [target]

# --- Defaults for single prediction ---
DEFAULTS = {
    "Year": 2015,
    "Age": 5,
    "Kilometers_Driven": 50000,
    "Log_Km": 10.82,
    "Mileage": 15.0,
    "Engine": 1200,
    "Power": 90,
    "Seats": 5,
    "Fuel_Type": "Petrol",
    "Transmission": "Manual",
    "Owner_Type": "First",
    "Location": "Mumbai",
    "Brand_Model": "Maruti Alto",
    "Brand": "Maruti",
    "Price": 5.0
}

mode = st.radio("Choose input mode:", ["Single Prediction", "Batch Upload"])

# --- Single Prediction Input ---
def make_inputs():
    vals = {}
    errs = []
    cols = st.columns(2)
    for i, name in enumerate(FEATURES[:-1]):  # exclude target
        with cols[i % 2]:
            s = st.text_input(name, value=str(DEFAULTS.get(name, "")))
        try:
            if name in numeric_features:
                vals[name] = float(s.strip())
            else:
                vals[name] = s.strip()
        except Exception:
            errs.append(f"{name} must be numeric")
    if errs:
        st.error(" | ".join(errs))
        return None
    return vals

# --- Single Prediction Mode ---
if mode == "Single Prediction":
    vals = make_inputs()
    if st.button("Predict", type="primary"):
        if vals:
            try:
                df = pd.DataFrame([vals], columns=FEATURES[:-1])
                preds = predict_model(model, data=df)
                st.success("Prediction complete!")
                st.dataframe(preds)
            except Exception as e:
                st.error(f"Prediction error: {e}")

# --- Batch Upload Mode ---
else:
    uploaded = st.file_uploader("Upload a CSV file with all required columns", type=["csv"])
    if uploaded:
        try:
            data = pd.read_csv(uploaded)
            missing = [c for c in FEATURES[:-1] if c not in data.columns]
            if missing:
                st.error(f"Missing columns in CSV: {missing}")
            else:
                data = data[FEATURES[:-1]]  # enforce column order
                preds = predict_model(model, data=data)
                st.success("Batch prediction complete!")
                st.dataframe(preds)

                # Download predictions
                csv = preds.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=csv,
                    file_name="usedcar_predictions.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Batch prediction error: {e}")
