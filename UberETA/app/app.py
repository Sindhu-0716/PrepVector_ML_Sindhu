import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import datetime
import joblib

# Define paths for model and encoders
model_path = "models/random_forest_model_compressed.pkl"
encoder_path = "models/label_encoders.pkl"
image_path = "delivery_driver.jpg"

# Load the trained best model
with open(model_path, "rb") as model_file:
    model = joblib.load(model_file)

# Load the label encoders
with open(encoder_path, "rb") as encoders_file:
    label_encoders = pickle.load(encoders_file)

# Retrieve feature names from the trained model
model_features = model.feature_names_in_

# Streamlit UI Styling
st.set_page_config(page_title="Food Delivery Time Estimator", layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
        .main {background-color: #f8f9fa;}
        .title {text-align: center; font-size: 36px; color: #ff6600; font-weight: bold;}
        .footer {position: fixed; bottom: 0; width: 100%; text-align: center; padding: 10px; background: #222; color: white; font-size: 14px;}
        .stButton>button {background-color: #ff6600; color: white; font-weight: bold; border-radius: 8px; padding: 10px 20px;}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='title'>Food Delivery Time Estimator</div>", unsafe_allow_html=True)
st.write("### Enter the required details to estimate the delivery time in minutes.")

# Display an image of a delivery driver
if os.path.isfile(image_path):
    st.write(f"Image found at: {image_path}")
    image = Image.open(image_path)
    st.image(image, caption="Delivery driver picking up food", use_column_width=True)
else:
    st.warning(f"Image not found at {image_path}. Please check the filename and ensure it is in the 'app/' folder.")

# User input fields - Wide Layout
col1, col2 = st.columns(2)
user_input = {}

# Categorical inputs with dropdowns
categorical_cols = ["City_type", "Festival", "Type_of_order", "Type_of_vehicle", "Road_traffic_density", "Weatherconditions"]
options = {
    "City_type": ["Metropolitan", "Urban", "Semi-Urban"],
    "Festival": ["Yes", "No"],
    "Type_of_order": ["Drinks", "Snack", "Meal", "Buffet"],
    "Type_of_vehicle": ["Bike", "Scooter", "Electric Bike", "Car"],
    "Road_traffic_density": ["Low", "Medium", "High", "Jam"],
    "Weatherconditions": ["Sunny", "Stormy", "Sandstorm", "Windy", "Cloudy", "Fog"]
}

with col1:
    for col in categorical_cols[:3]:
        user_input[col] = st.selectbox(f"{col}", options[col])

with col2:
    for col in categorical_cols[3:]:
        user_input[col] = st.selectbox(f"{col}", options[col])

# Numeric inputs
with col1:
    user_input["Multiple_deliveries"] = st.number_input("Number of multiple deliveries", min_value=0, max_value=5, value=1)
    user_input["Vehicle_condition"] = st.slider("Vehicle Condition (1-5)", 1, 5, 3)
    user_input["Delivery_person_Age"] = st.number_input("Delivery Person Age", min_value=18, max_value=60, value=25)

with col2:
    user_input["distance_km"] = st.number_input("Distance (km)", min_value=0.1, max_value=50.0, value=5.0)
    user_input["Delivery_person_Ratings"] = st.slider("Delivery Person Ratings (1-5)", 1.0, 5.0, 4.5)

# Generate missing features dynamically
current_date = datetime.datetime.now()
user_input["is_weekend"] = 1 if current_date.weekday() >= 5 else 0
user_input["month_intervals"] = "start_month" if current_date.day <= 10 else ("middle_month" if current_date.day <= 20 else "end_month")
user_input["year_quarter"] = (current_date.month - 1) // 3 + 1
user_input["order_prepare_time"] = np.random.uniform(5, 30)  # Estimate order prep time randomly within a reasonable range

# Convert user input to dataframe
input_df = pd.DataFrame([user_input])

# Apply label encoding to categorical variables
for col in categorical_cols + ["month_intervals"]:
    if col in label_encoders:
        input_df[col] = input_df[col].map(lambda x: label_encoders[col].transform([x])[0] if x in label_encoders[col].classes_ else label_encoders[col].transform([label_encoders[col].classes_[0]])[0])

# Ensure correct column order and fill missing columns
for col in model_features:
    if col not in input_df.columns:
        input_df[col] = 0  # Assign default values for missing columns
input_df = input_df[model_features]  # Reorder columns to match training set

# Convert all data to float to avoid string errors
input_df = input_df.apply(pd.to_numeric, errors='coerce')

# Make prediction
if st.button("Estimate Delivery Time"):
    prediction = model.predict(input_df)
    st.success(f"Estimated Delivery Time: {prediction[0]:.2f} minutes")

# Footer
st.markdown("<div class='footer'>Powered by Food Delivery Analytics</div>", unsafe_allow_html=True)
