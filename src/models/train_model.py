import pandas as pd 
from sklearn import ensemble
import joblib
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
        n_estimators=100, 
        max_depth=None, 
        random_state=42
    )
    rf_classifier.fit(X_train, y_train)
    print("Model training completed.")

    # Vorhersagen und Bewertung
    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")

    # Parameter zu MLflow loggen
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", None)
    mlflow.log_param("n_jobs", -1)
    mlflow.log_param("random_state", 42)

    # Metriken zu MLflow loggen
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
    bentoml.sklearn.save_model("predict_model", rf_classifier)
    print("✅ Model saved to BentoML successfully.")
except Exception as e:
    print(f"❌ BentoML saving failed: {e}")

print(f"\n🎉 Training completed!")
print(f"📊 Final accuracy: {accuracy:.4f}")
print(f"💾 Model saved locally and to BentoML")