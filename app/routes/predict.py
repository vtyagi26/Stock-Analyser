# app/routes/predict.py
from fastapi import APIRouter, HTTPException
from app.schemas.prediction import PredictionResponse
from app.services.predict_service import predict_next_day

router = APIRouter(prefix="", tags=["Prediction"])

@router.get("/predict/{symbol}", response_model=PredictionResponse)
def predict(symbol: str):
    try:
        return predict_next_day(symbol.upper())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))