from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.predict import router as predict_router
from app.services.model_service import load_models

app = FastAPI(title="Stock Prediction API")


@app.on_event("startup")
def startup_event():
    load_models()
    print("Models loaded successfully.")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router)


@app.get("/")
def root():
    return {"message": "Stock Prediction API is running"}