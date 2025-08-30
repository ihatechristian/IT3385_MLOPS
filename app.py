import streamlit as st

st.set_page_config(page_title="ML Predictor Hub", page_icon="🤖", layout="wide")

st.title("Welcome to the ML Predictor Hub 🤖")
st.write("""
This app lets you try out 3 different machine learning models:

1. 🌾 **Wheat Seeds Variety Classifier**  
2. 🚗 **Used Car Price Predictor**  
3. 🏠 **Melbourne Housing Price Estimator**

👉 Use the sidebar to navigate between them.
""")

st.info("Select a model from the left sidebar to get started.")
