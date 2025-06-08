# src/api/app.py
from fastapi import FastAPI
from src.models.predict_model import predict

app = FastAPI()

@app.post("/predict")
def get_prediction(input_data: dict):
    return predict(input_data)
