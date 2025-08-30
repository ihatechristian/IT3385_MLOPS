import pandas as pd
import streamlit as st
from pycaret.regression import load_model, predict_model

st.set_page_config(page_title="Melbourne Housing", page_icon="🏠", layout="wide")
st.title("🏠 Melbourne Housing Price Estimator")

# Load model
model = load_model("models/melbourne_housing_model")

# Selected columns for input (exclude target Price_Dollars)
FEATURES = [
    'Rooms', 'Property_Type', 'Sale_Method', 'Distance_to_CBD_km', 'Parking_Spaces',
    'Land_Size', 'Building_Size', 'Governing_Council', 'Latitude', 'Longitude',
    'Region', 'Property_Age'
]

# Defaults for input fields
DEFAULTS = {
    "Rooms": 3,
    "Property_Type": "t",                 # maps to "Type" column in your dataset
    "Sale_Method": "PI",                  # maps to "Method"
    "Distance_to_CBD_km": 13.5,
    "Parking_Spaces": 1,                  # maps to "Car"
    "Land_Size": 303,                      # Landsize
    "Building_Size": 225,                  # BuildingArea
    "Governing_Council": "Moonee Valley",  # CouncilArea
    "Latitude": -37.718,                   # Lattitude
    "Longitude": 144.878,                  # Longtitude
    "Region": "Western Metropolitan",
    "Property_Age": 2023 - 2016            # current year minus YearBuilt
}

# Define numeric and categorical features
NUMERIC_FEATURES = [
    'Rooms', 'Distance_to_CBD_km', 'Parking_Spaces', 'Land_Size',
    'Building_Size', 'Latitude', 'Longitude', 'Property_Age'
]
CATEGORICAL_FEATURES = [f for f in FEATURES if f not in NUMERIC_FEATURES]

mode = st.radio("Choose input mode:", ["Single Prediction", "Batch Upload"])

def make_inputs():
    vals = {}
    errs = []
    cols = st.columns(2)
    for i, name in enumerate(FEATURES):
        with cols[i % 2]:
            s = st.text_input(name, value=str(DEFAULTS.get(name, "")))
        try:
            if name in NUMERIC_FEATURES:
                vals[name] = float(s.strip())
            else:
                vals[name] = s.strip()
        except Exception:
            errs.append(f"{name} must be numeric")
    if errs:
        st.error(" | ".join(errs))
        return None
    return vals

# Single Prediction Mode
if mode == "Single Prediction":
    vals = make_inputs()
    if st.button("Predict", type="primary"):
        if vals:
            try:
                df = pd.DataFrame([vals], columns=FEATURES)
                # Convert categorical columns to category type
                for col in CATEGORICAL_FEATURES:
                    df[col] = df[col].astype("category")

                preds = predict_model(model, data=df)
                st.success("Prediction complete!")
                st.dataframe(preds)
            except Exception as e:
                st.error(f"Prediction error: {e}")

# Batch Upload Mode
else:
    uploaded = st.file_uploader("Upload a CSV file with the required columns", type=["csv"])
    if uploaded:
        try:
            data = pd.read_csv(uploaded)
            # Keep only selected columns
            data = data[FEATURES]
            # Convert categorical columns to category type
            for col in CATEGORICAL_FEATURES:
                data[col] = data[col].astype("category")
            preds = predict_model(model, data=data)
            st.success("Batch prediction complete!")
            st.dataframe(preds)

            # Download predictions
            csv = preds.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv,
                file_name="melbourne_predictions.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Batch prediction error: {e}")