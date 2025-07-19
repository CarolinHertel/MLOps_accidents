import bentoml
from bentoml.io import JSON
import joblib
from pydantic import BaseModel
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
import os
from dotenv import load_dotenv
import numpy as np
import time
import pandas as pd  # <-- ADD THIS IMPORT

load_dotenv()
SECRET_KEY = os.getenv("JWT_SECRET", "I_know_that_this _is_unsecure_but_I_don't_care")
ALGORITHM = "HS256"
USERNAME = "admin"
PASSWORD = "4dm1N"

# Load column order for one-hot encoded features
COLUMNS_PATH = "trained_model_columns.npy"
if os.path.exists(COLUMNS_PATH):
    TRAIN_COLS = np.load(COLUMNS_PATH, allow_pickle=True)
else:
    TRAIN_COLS = None

class AdmissionRequest(BaseModel):
    place: int   
    catu: int
    sexe: int
    secu1: float
    year_acc: int
    victim_age: int
    catv: int
    obsm: int
    motor: int
    catr: int
    circ: int
    surf: int
    situ: int
    vma: int
    jour: int
    mois: int
    lum: int   
    dep: int
    com: int
    agg_: int
    int: int
    atm: int
    col: int
    lat: float
    long: float
    hour: int
    nb_victim: int
    nb_vehicules: int

class HTTPBearer401(HTTPBearer):
    async def __call__(self, request: Request):
        try:
            return await super().__call__(request)
        except HTTPException as exc:
            if exc.status_code == 403:
                raise HTTPException(status_code=401, detail="Not authenticated")
            raise

def load_model():
    try:
        if os.path.exists("trained_model.joblib"):
            return joblib.load("trained_model.joblib")
        else:
            raise FileNotFoundError("trained_model.joblib nicht gefunden!")
    except Exception as e:
        print(f"Fehler beim Laden des Modells: {e}")
        return None

model = load_model()
svc = bentoml.Service("accident_prediction")

def create_jwt_token(username: str) -> str:
    payload = {
        "sub": username,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=30)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer401())):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        return payload["sub"]
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def admission_request_to_dataframe(admission_request: AdmissionRequest) -> pd.DataFrame:
    data = pd.DataFrame([admission_request.dict()])
    data = pd.get_dummies(data)
    if TRAIN_COLS is not None:
        data = data.reindex(columns=TRAIN_COLS, fill_value=0)
    return data

app = FastAPI(title="Admission Prediction API", version="1.0.0")

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/login")
def login(request: dict):
    username = request.get("username")
    password = request.get("password")
    if username == USERNAME and password == PASSWORD:
        token = create_jwt_token(username)
        return {"token": token}
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")

@app.post("/predict", dependencies=[Depends(verify_token)])
async def predict_with_auth(input_data: AdmissionRequest):
    from prometheus_client import Counter, Gauge
    if not hasattr(predict_with_auth, "prediction_count"):
        predict_with_auth.prediction_count = Counter(
            'custom_predictions_total', 'Total predictions made by /predict endpoint'
        )
    if not hasattr(predict_with_auth, "prediction_score_gauge"):
        predict_with_auth.prediction_score_gauge = Gauge(
            'custom_last_prediction_score', 'Last prediction score by /predict endpoint'
        )
    predict_with_auth.prediction_count.inc()
    prediction_start = time.time()
    input_df = admission_request_to_dataframe(input_data)    # <-- CHANGED
    prediction = model.predict(input_df)                     # <-- CHANGED
    score = round(float(prediction[0]), 3)
    predict_with_auth.prediction_score_gauge.set(score)
    return {
        "chance_of_admit": score,
        "processing_time": round(time.time() - prediction_start, 3)
    }

@app.post("/predict-simple")
async def predict_simple(input_data: AdmissionRequest):
    from prometheus_client import Counter
    if not hasattr(predict_simple, "prediction_simple_count"):
        predict_simple.prediction_simple_count = Counter(
            'custom_predictions_simple_total', 'Total predictions made by /predict-simple endpoint'
        )
    predict_simple.prediction_simple_count.inc()
    prediction_start = time.time()
    input_df = admission_request_to_dataframe(input_data)    # <-- CHANGED
    prediction = model.predict(input_df)                     # <-- CHANGED
    score = round(float(prediction[0]), 3)
    return {
        "chance_of_admit": score,
        "processing_time": round(time.time() - prediction_start, 3)
    }

# Mount FastAPI app to BentoML service
svc.mount_asgi_app(app)

