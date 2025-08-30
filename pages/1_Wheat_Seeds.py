import pandas as pd
import streamlit as st
from pycaret.classification import load_model, predict_model

st.set_page_config(page_title="Wheat Seeds", page_icon="🌾", layout="wide")
st.title("🌾 Wheat Seeds Variety Classifier")

# Load the PyCaret model
model = load_model("models/wheat_seeds")

# Features used by the model
FEATURES = ["Area", "Compactness", "Length", "AsymmetryCoeff", "Groove"]

# Defaults based on your sample row
DEFAULTS = {
    "Area": 15.26,
    "Compactness": 0.871,
    "Length": 5.763,
    "AsymmetryCoeff": 2.221,
    "Groove": 5.22
}

mode = st.radio("Choose input mode:", ["Single Prediction", "Batch Upload"])

# Function to handle single prediction inputs
def make_inputs():
    vals = {}
    cols = st.columns(2)
    for i, name in enumerate(FEATURES):
        with cols[i % 2]:
            s = st.text_input(name, value=f"{DEFAULTS[name]}")
        try:
            vals[name] = float(s.strip())
        except Exception:
            st.error(f"{name} must be numeric")
            return None
    return vals

# --- Single Prediction Mode ---
if mode == "Single Prediction":
    vals = make_inputs()
    if st.button("Predict", type="primary"):
        if vals:
            try:
                df = pd.DataFrame([vals], columns=FEATURES)
                preds = predict_model(model, data=df)
                st.success("Prediction complete!")
                st.dataframe(preds)
            except Exception as e:
                st.error(f"Prediction error: {e}")

# --- Batch Upload Mode ---
else:
    uploaded = st.file_uploader("Upload a CSV file with the required columns", type=["csv"])
    if uploaded:
        try:
            data = pd.read_csv(uploaded)
            # Keep only the columns the model expects
            data = data[FEATURES]
            preds = predict_model(model, data=data)
            st.success("Batch prediction complete!")
            st.dataframe(preds)

            # Allow download of predictions
            csv = preds.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv,
                file_name="wheat_predictions.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Batch prediction error: {e}")
