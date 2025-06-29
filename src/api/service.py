# Erweiterte Version deiner API mit Prometheus Monitoring
import bentoml
from bentoml.io import JSON
from pydantic import BaseModel
from datetime import datetime, timedelta
from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
import os
from dotenv import load_dotenv
import numpy as np
import time

# Prometheus imports
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# Load secret key from .env or hardcode for local testing
load_dotenv()
SECRET_KEY = os.getenv("JWT_SECRET", "I_know_that_this _is_unsecure_but_I_don't_care")
ALGORITHM = "HS256"

# Dummy credentials
USERNAME = "admin"
PASSWORD = "4dm1N"

# Prometheus metrics
request_count = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
request_duration = Histogram('api_request_duration_seconds', 'Request duration', ['method', 'endpoint'])
prediction_count = Counter('predictions_total', 'Total predictions made')
prediction_score_gauge = Gauge('last_prediction_score', 'Last prediction score')
active_connections = Gauge('active_connections', 'Currently active connections')
model_load_time = Gauge('model_load_time_seconds', 'Time taken to load model')

# Request schema
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
    agg_:int
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

# Load model and create runner with timing
start_time = time.time()
model_ref = bentoml.sklearn.get("predict_model:latest")
model_runner = model_ref.to_runner()
load_time = time.time() - start_time
model_load_time.set(load_time)

# BentoML Service
svc = bentoml.Service("AdmissionPredictionService")

# Auth token generator
def create_jwt_token(username: str) -> str:
    payload = {
        "sub": username,
        "exp": datetime.now(datetime.timezone.utc) + timedelta(minutes=30)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

# Token validation dependency
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer401())):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        return payload["sub"]
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def admission_request_to_numpy(admission_request: AdmissionRequest) -> np.ndarray:
    features = [
        admission_request.place,
        admission_request.catu,
        admission_request.sexe,
        admission_request.secu1,
        admission_request.year_acc,
        admission_request.victim_age,
        admission_request.catv,
        admission_request.obsm,
        admission_request.motor,
        admission_request.catr,
        admission_request.circ, 
        admission_request.surf,
        admission_request.situ, 
        admission_request.vma,
        admission_request.jour,
        admission_request.mois, 
        admission_request.lum,
        admission_request.dep,
        admission_request.com,
        admission_request.agg_,
        admission_request.int,
        admission_request.atm,
        admission_request.col,
        admission_request.lat,
        admission_request.long,
        admission_request.hour,
        admission_request.nb_victim,
        admission_request.nb_vehicules
    ]
    return np.array([features], dtype=np.float32)

# FastAPI App with middleware for monitoring
app = FastAPI(title="Admission Prediction API", version="1.0.0")

@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    # Increment active connections
    active_connections.inc()
    
    start_time = time.time()
    method = request.method
    path = request.url.path
    
    try:
        response = await call_next(request)
        status_code = response.status_code
        
        # Record metrics
        request_count.labels(method=method, endpoint=path, status=status_code).inc()
        request_duration.labels(method=method, endpoint=path).observe(time.time() - start_time)
        
        return response
    except Exception as e:
        # Record failed requests
        request_count.labels(method=method, endpoint=path, status="500").inc()
        request_duration.labels(method=method, endpoint=path).observe(time.time() - start_time)
        raise e
    finally:
        # Decrement active connections
        active_connections.dec()

# Prometheus metrics endpoint
@app.get("/metrics")
def get_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Health check endpoint
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
    prediction_start = time.time()
    
    input_array = admission_request_to_numpy(input_data)
    prediction = await model_runner.async_run(input_array)
    
    # Update prediction metrics
    prediction_count.inc()
    score = round(float(prediction[0]), 3)
    prediction_score_gauge.set(score)
    
    return {"chance_of_admit": score, "processing_time": round(time.time() - prediction_start, 3)}

@app.post("/predict-simple")
async def predict_simple(input_data: AdmissionRequest):
    prediction_start = time.time()
    
    input_array = admission_request_to_numpy(input_data)
    prediction = await model_runner.async_run(input_array)
    
    # Update prediction metrics
    prediction_count.inc()
    score = round(float(prediction[0]), 3)
    prediction_score_gauge.set(score)
    
    return {"chance_of_admit": score, "processing_time": round(time.time() - prediction_start, 3)}

# Statistics endpoint
@app.get("/stats")
def get_stats():
    return {
        "model_load_time": model_load_time._value._value,
        "total_predictions": prediction_count._value._value,
        "last_prediction_score": prediction_score_gauge._value._value,
        "active_connections": active_connections._value._value
    }

# Mount FastAPI app to BentoML service
svc.mount_asgi_app(app)