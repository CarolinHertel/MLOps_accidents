import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import os

# Set MLflow tracking URI to your local directory
mlflow.set_tracking_uri("file:///Users/martahoisak/MLOps_accidents/mlruns")
mlflow.set_experiment("Accident_Prediction")

# Load data
X_train = pd.read_csv("data/preprocessed/X_train_clean.csv")
X_test = pd.read_csv("data/preprocessed/X_test_clean.csv")
y_train = pd.read_csv("data/preprocessed/y_train.csv")
y_test = pd.read_csv("data/preprocessed/y_test.csv")

# Flatten labels
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

# One-hot encode and align columns
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Convert int to float64 to avoid schema issues
X_train = X_train.astype({col: 'float64' for col in X_train.select_dtypes(include='int').columns})
X_test = X_test.astype({col: 'float64' for col in X_test.select_dtypes(include='int').columns})

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy:.4f}")

# Save model manually
model_filename = "src/models/trained_model.joblib"
joblib.dump(model, model_filename)

# Log parameters + metrics only (no artifacts/models to avoid filesystem error)
with mlflow.start_run():
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("accuracy", accuracy)

print("✅ Model saved and metrics logged successfully.")
