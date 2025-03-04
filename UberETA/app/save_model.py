import pickle
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression  # Import this!
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score

# Ensure the app directory exists
os.makedirs("app", exist_ok=True)

# Load dataset (Update path if necessary)
uber_data = pd.read_csv("../data/train.csv")

# Define target variable
target_column = "Time_taken(min)"

# Separate features (X) and target (Y)
X = uber_data.drop(columns=[target_column])
Y = uber_data[target_column].fillna(uber_data[target_column].median())

# Separate numerical and categorical columns
num_cols = X.select_dtypes(include=["number"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

# Handle missing values
num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")

X[num_cols] = num_imputer.fit_transform(X[num_cols])
X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])

# Apply Label Encoding
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize best model tracking
best_model_obj = None
best_model_name = None
best_r2 = float('-inf')

# Train multiple models and pick the best one
models = {
    "Linear Regression": LinearRegression(),  # FIXED: Now properly imported
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

for name, model in models.items():
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(Y_test, y_pred)
    
    print(f"{name} R² Score: {r2:.4f}")
    
    if r2 > best_r2:
        best_r2 = r2
        best_model_obj = model
        best_model_name = name

print(f"✅ Best Model: {best_model_name} with R² Score: {best_r2:.4f}")

# Save the best model
model_path = "app/best_model.pkl"
with open(model_path, "wb") as model_file:
    pickle.dump(best_model_obj, model_file)

# Save the label encoders
encoder_path = "app/label_encoders.pkl"
with open(encoder_path, "wb") as encoders_file:
    pickle.dump(label_encoders, encoders_file)

print(f"🎉 Best Model saved as: {model_path}")
print(f"🎉 Label Encoders saved as: {encoder_path}")
