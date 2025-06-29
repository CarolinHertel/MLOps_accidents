import pandas as pd 
from sklearn import ensemble
import joblib
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn import ensemble
from sklearn.metrics import accuracy_score
import bentoml

print(joblib.__version__)

X_train = pd.read_csv('../../data/preprocessed/X_train.csv')
X_test = pd.read_csv('../../data/preprocessed/X_test.csv')
y_train = pd.read_csv('../../data/preprocessed/y_train.csv')
y_test = pd.read_csv('../../data/preprocessed/y_test.csv')
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

mlflow.set_experiment("Accident_Prediction")
#rf_classifier = ensemble.RandomForestClassifier(n_jobs = -1)

#--Train the model and start tracking run with MLflow
with mlflow.start_run(): 

#--Initialize and train the Random Forest Classifier
    rf_classifier = ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=100, max_depth=None, random_state=42)
    rf_classifier.fit(X_train, y_train)

# --Predict and evaluate the model
    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

#--Log model parameters and metrics to MLflow
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", None)
    mlflow.log_param("n_jobs", -1)
    mlflow.log_param("random_state", 42)

#--Log the accuracy
    mlflow.log_metric("accuracy", accuracy)

#--Log the trained model to MLflow
    mlflow.sklearn.log_model(rf_classifier, artifact_path="model")
    
# Also save locally as joblib file
    model_filename = 'trained_model.joblib'
    joblib.dump(rf_classifier, model_filename)
    mlflow.log_artifact(model_filename, artifact_path="model_artifacts")

rf_classifier.fit(X_train, y_train)

#--Save the trained model to a file
model_filename = 'trained_model.joblib'
joblib.dump(rf_classifier, model_filename)
print("Model trained and saved successfully.")

print(f"✅ Model trained and logged with accuracy: {accuracy:.4f}")
print("Model logged with MLflow.")

#-- Save Model to bentoML
bentoml.sklearn.save_model("predict_model", rf_classifier)
print("Model saved to BentoML successfully.")