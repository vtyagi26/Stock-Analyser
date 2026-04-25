# app/main.py
from fastapi import FastAPI
from app.routes.predict import router as predict_router
from app.services.model_service import load_models

app = FastAPI(
    title="Stock Predictor Service",
    version="1.0.0"
)

@app.on_event("startup")
def startup_event():
    load_models()

@app.get("/health")
def health():
    return {"status": "ok"}

app.include_router(predict_router)