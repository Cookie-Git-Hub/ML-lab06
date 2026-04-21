from fastapi import FastAPI
from pydantic import BaseModel, Field
import numpy as np
from sklearn.linear_model import LogisticRegression
from fastapi import FastAPI, HTTPException

import os

app = FastAPI(
    title="ML Prediction API",
    description="API do serwowania prostego modelu ML",
    version="1.0.0",
)

# Dane treningowe
X = np.array([
    [1, 40],
    [2, 50],
    [3, 60],
    [4, 70],
    [5, 80],
    [6, 90],
], dtype=float)

y = np.array([0, 0, 0, 1, 1, 1])

model = LogisticRegression()
model.fit(X, y)

FEATURE_NAMES = ["hours_studied", "attendance_percent"]


class PredictRequest(BaseModel):
    hours_studied: float = Field(..., ge=0)
    attendance_percent: float = Field(..., ge=0, le=100)


@app.get("/")
def root():
    api_name = os.getenv("API_NAME", "ML API")
    return {"message": api_name}


@app.post("/predict")
def predict(data: PredictRequest):
    # Logika obsługi sytuacji, w której dane wejściowe są nieprawidłowe
    if data.hours_studied == 0 and data.attendance_percent == 0:
        raise HTTPException(
            status_code=422,
            detail="Oczekiwano co najmniej jednej niezerowej wartości wejściowej."
        )
    features = np.array([[data.hours_studied, data.attendance_percent]])
    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][1])

    return {
        "prediction": prediction,
        "probability": round(probability, 4)
    }


@app.get("/info")
def info():
    return {
        "model_type": "LogisticRegression",
        "number_of_features": len(FEATURE_NAMES),
        "features": FEATURE_NAMES
    }


@app.get("/health")
def health():
    return {"status": "ok"}
