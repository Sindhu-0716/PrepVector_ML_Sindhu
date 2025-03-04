import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

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

# Save the model
model_path = "models/random_forest_model.pkl"
with open(model_path, "wb") as model_file:
    pickle.dump(rf_model, model_file)

# Save label encoders
encoder_path = "models/label_encoders.pkl"
with open(encoder_path, "wb") as enc_file:
    pickle.dump(label_encoders, enc_file)

print(f"✅ Model saved at {model_path}")
print(f"✅ Label encoders saved at {encoder_path}")
