# app/services/model_service.py
import joblib
from xgboost import XGBRegressor
from app.core.config import MODEL_DIR

reg_model = None
clf_model = None

def load_models():
    global reg_model, clf_model

    reg = XGBRegressor()
    reg.load_model(f"{MODEL_DIR}/xgb_model.json")

    clf = joblib.load(f"{MODEL_DIR}/rf_model.pkl")

    reg_model = reg
    clf_model = clf

def get_models():
    return reg_model, clf_model