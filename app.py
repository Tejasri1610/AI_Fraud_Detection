from fastapi import FastAPI
from pydantic import BaseModel
from inference import predict_fraud

app = FastAPI(title="Fraud Detection API")

class Transaction(BaseModel):
    features: list  # Expects a list of numerical features

@app.post("/predict")
def get_prediction(data: Transaction):
    prediction = predict_fraud(data.features)
    return {"is_fraud": bool(prediction), "raw_prediction": int(prediction)}
