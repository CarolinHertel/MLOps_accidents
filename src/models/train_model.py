import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score
import bentoml
import os
from pathlib import Path

print(f"Joblib version: {joblib.__version__}")

# Lade die Daten
X_train = pd.read_csv('../../data/preprocessed/X_train.csv')
X_test = pd.read_csv('../../data/preprocessed/X_test.csv')
y_train = pd.read_csv('../../data/preprocessed/y_train.csv')
y_test = pd.read_csv('../../data/preprocessed/y_test.csv')
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

# MLflow Tracking URI auf einen Pfad mit Schreibberechtigung setzen
mlflow_tracking_path = Path("./mlruns")
mlflow_tracking_path.mkdir(parents=True, exist_ok=True)
mlflow.set_tracking_uri(f"file://{mlflow_tracking_path.absolute()}")

# MLflow Experiment setzen
mlflow.set_experiment("Accident_Prediction")

# Training und MLflow Tracking
with mlflow.start_run(): 
    print("Starting MLflow run...")
    
    # Random Forest Classifier initialisieren und trainieren
    rf_classifier = ensemble.RandomForestClassifier(
        n_jobs=-1, 
        n_estimators=120, 
        max_depth=None, 
        random_state=45
    )
    rf_classifier.fit(X_train, y_train)
    print("Model training completed.")

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy:.4f}")

# Save model manually
model_filename = "src/models/trained_model.joblib"
joblib.dump(model, model_filename)

    # Parameter zu MLflow loggen
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
 93f2a6a (Train model with MLflow tracking and joblib saving)
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)

    # Metriken zu MLflow login 
 8a713b86c898e1119cc3122bd5e773f09fc1c5c9
    mlflow.log_metric("accuracy", accuracy)

    # Modell zu MLflow loggen (korrekte Syntax ohne artifact_path)
    try:
        mlflow.sklearn.log_model(
            sk_model=rf_classifier,
            artifact_path="model",  # Dieser Parameter ist korrekt für den Pfad im MLflow Run
            input_example=X_train.iloc[:5],  # Eingabebeispiel hinzufügen
            signature=mlflow.models.infer_signature(X_train, y_train)  # Signatur hinzufügen
        )
        print("✅ Model successfully logged to MLflow")
    except Exception as e:
        print(f"❌ MLflow logging failed: {e}")
        print("Continuing with local model saving...")

    # Lokale Speicherung als joblib-Datei
    model_filename = 'trained_model.joblib'
    joblib.dump(rf_classifier, model_filename)
    print(f"✅ Model saved locally as: {model_filename}")
    
    # Artefakt zu MLflow loggen (falls MLflow funktioniert)
    try:
        mlflow.log_artifact(model_filename, artifact_path="model_artifacts")
        print("✅ Model artifact logged to MLflow")
    except Exception as e:
        print(f"⚠️ MLflow artifact logging failed: {e}")

# BentoML Model speichern
try:
    #bentoml.sklearn.save_model("predict_model", rf_classifier)
    print("✅ Model saved to BentoML successfully.")
except Exception as e:
    print(f"❌ BentoML saving failed: {e}")

print(f"\n🎉 Training completed!")
print(f"📊 Final accuracy: {accuracy:.4f}")
print(f"💾 Model saved locally and to BentoML")

    mlflow.log_metric("accuracy", accuracy)

print("✅ Model saved and metrics logged successfully.")
 93f2a6a (Train model with MLflow tracking and joblib saving)
