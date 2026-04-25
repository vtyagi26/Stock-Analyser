# app/schemas/prediction.py
from pydantic import BaseModel

class PredictionResponse(BaseModel):
    symbol: str
    predicted_close: float
    predicted_return: float
    predicted_direction: str
    confidence: float
    timestamp: str