# train.py
import os
import joblib
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor

from app.core.config import DEFAULT_START_DATE, MODEL_DIR
from app.services.data_service import fetch_bars
from app.services.feature_service import build_features, FEATURE_COLUMNS

SYMBOL = "AAPL"
END_DATE = "2026-04-04"

os.makedirs(MODEL_DIR, exist_ok=True)

df = fetch_bars(SYMBOL, DEFAULT_START_DATE, END_DATE)
spy = fetch_bars("SPY", DEFAULT_START_DATE, END_DATE)
qqq = fetch_bars("QQQ", DEFAULT_START_DATE, END_DATE)

df = build_features(df, spy, qqq)

df["target_return"] = df["close"].shift(-1) / df["close"] - 1
df["target_direction"] = (df["close"].shift(-1) > df["close"]).astype(int)
df = df.dropna().reset_index(drop=True)

X = df[FEATURE_COLUMNS]
y_reg = df["target_return"]
y_clf = df["target_direction"]

tscv = TimeSeriesSplit(n_splits=5)

reg_preds, reg_actuals = [], []
clf_preds, clf_actuals = [], []

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train_reg, y_test_reg = y_reg.iloc[train_idx], y_reg.iloc[test_idx]
    y_train_clf, y_test_clf = y_clf.iloc[train_idx], y_clf.iloc[test_idx]

    reg_model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    reg_model.fit(X_train, y_train_reg)

    pred_returns = reg_model.predict(X_test)
    pred_prices = X_test["close"].values * (1 + pred_returns)
    actual_prices = X_test["close"].values * (1 + y_test_reg.values)

    reg_preds.extend(pred_prices)
    reg_actuals.extend(actual_prices)

    clf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=42
    )
    clf_model.fit(X_train, y_train_clf)

    pred_dir = clf_model.predict(X_test)
    clf_preds.extend(pred_dir)
    clf_actuals.extend(y_test_clf)

mae = mean_absolute_error(reg_actuals, reg_preds)
acc = accuracy_score(clf_actuals, clf_preds)

print(f"Price MAE: ${mae:.2f}")
print(f"Direction Accuracy: {acc * 100:.2f}%")

reg_model.fit(X, y_reg)
clf_model.fit(X, y_clf)

reg_model.save_model(f"{MODEL_DIR}/xgb_model.json")
joblib.dump(clf_model, f"{MODEL_DIR}/rf_model.pkl")

print("Models saved successfully.")