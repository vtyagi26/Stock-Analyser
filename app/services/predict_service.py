# app/services/predict_service.py
from datetime import datetime
from app.core.config import DEFAULT_START_DATE
from app.services.data_service import fetch_bars
from app.services.feature_service import build_features, FEATURE_COLUMNS
from app.services.model_service import get_models

def predict_next_day(symbol: str):
    end_date = datetime.utcnow().strftime("%Y-%m-%d")

    df = fetch_bars(symbol, DEFAULT_START_DATE, end_date)
    spy = fetch_bars("SPY", DEFAULT_START_DATE, end_date)
    qqq = fetch_bars("QQQ", DEFAULT_START_DATE, end_date)

    df = build_features(df, spy, qqq)

    latest_row = df.iloc[-1:]
    X_latest = latest_row[FEATURE_COLUMNS]
    latest_close = latest_row["close"].iloc[0]

    reg_model, clf_model = get_models()

    pred_return = float(reg_model.predict(X_latest)[0])
    pred_close = float(latest_close * (1 + pred_return))

    pred_dir = int(clf_model.predict(X_latest)[0])
    pred_prob = float(clf_model.predict_proba(X_latest)[0].max())

    return {
        "symbol": symbol,
        "predicted_close": round(pred_close, 2),
        "predicted_return": round(pred_return, 6),
        "predicted_direction": "UP" if pred_dir == 1 else "DOWN",
        "confidence": round(pred_prob, 4),
        "timestamp": datetime.utcnow().isoformat()
    }