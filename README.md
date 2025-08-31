Overview

This project implements an end-to-end MLOps workflow for 3 different machine learning models. The project follows best practices for data science and MLOps.


Team Members

Christian
Keagan
Nicolas


Folder Structure

├── data/                 # Raw and processed datasets

│   ├── 01_Melbourne_Residential.csv

│   ├── 02_Used_Car_Prices.xlsx

│   └── 03_Wheat_Seeds.csv

├── eda/                  # Raw and processed datasets

├── pipelines/            # Source code for the project

├── models/               # Pickle Files

│   ├── melbourne_housing_model.pkl

│   ├── usedcar_price_model.pkl

│   └── wheat_seeds.pkl

├── mlruns/               # Files to view models in mlflow or u can run the pipeline files and this would be created

├── pages/                # Streamlit pages 

│   ├── 01_Wheat_Seeds.py

│   ├── 02_UsedCar.py

│   └── 03_MelbourneHousing.py

├── pipelines/            # Notebook files that create the model files (Run this to update the model to new data)

├── app.py # Run this to start the app

├── README.md             # This file

└── requirements.txt      # Project dependencies (Created by Poetry)


Deployment Guide

Requires Python 3.10

pip install the requirements.txt
run the 3 ipynb files in the pipelines folder (Optional)
run the app.py with 
python -m streamlit run app.py 
or 
streamlit app.py

