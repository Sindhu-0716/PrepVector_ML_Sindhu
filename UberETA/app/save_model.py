import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import os

# Load dataset (Update path if necessary)
df = pd.read_csv("../data/train.csv")

# Define target variable
target_column = "Time_taken(min)"

# Separate features (X) and target (Y)
X = df.drop(columns=[target_column])
Y = df[target_column].fillna(df[target_column].median())

# Handle missing values
num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")

# Identify categorical and numerical columns
num_cols = X.select_dtypes(include=["number"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

# Apply imputers
X[num_cols] = num_imputer.fit_transform(X[num_cols])
X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])

# Encode categorical columns
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
rf_model.fit(X_train, Y_train)

# Create directory if not exists
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

# Save the compressed model
model_path = os.path.join(model_dir, "random_forest_model_compressed.pkl")
joblib.dump(rf_model, model_path, compress=3)  # Compression level 3

# Save label encoders
encoder_path = os.path.join(model_dir, "label_encoders.pkl")
os.makedirs(os.path.dirname(encoder_path), exist_ok=True)
with open(encoder_path, "wb") as enc_file:
    joblib.dump(label_encoders, enc_file)

print(f"✅ Compressed model saved at {model_path}")
print(f"✅ Label encoders saved at {encoder_path}")