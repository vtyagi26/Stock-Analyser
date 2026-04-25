import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dotenv import load_dotenv

from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBRegressor

from alpaca.data.timeframe import TimeFrame
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest


# =========================
# 1. LOAD ENV + API
# =========================
load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")

SYMBOL = "AAPL"
START_DATE = "2022-04-04"
END_DATE = "2026-04-04"

client = StockHistoricalDataClient(API_KEY, API_SECRET)


# =========================
# 2. FETCH MARKET DATA
# =========================
def fetch_bars(symbol, start, end):
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start,
        end=end
    )
    bars = client.get_stock_bars(request)
    df = bars.df.reset_index()

    # Alpaca may return symbol column or not for single ticker
    if "symbol" in df.columns:
        df = df[df["symbol"] == symbol]

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# Main stock
df = fetch_bars(SYMBOL, START_DATE, END_DATE)

# Market context
spy = fetch_bars("SPY", START_DATE, END_DATE)[["timestamp", "close"]].rename(columns={"close": "spy_close"})
qqq = fetch_bars("QQQ", START_DATE, END_DATE)[["timestamp", "close"]].rename(columns={"close": "qqq_close"})


# =========================
# 3. FEATURE ENGINEERING
# =========================
# Merge market context
df = df.merge(spy, on="timestamp", how="left")
df = df.merge(qqq, on="timestamp", how="left")

# Base returns
df["return"] = df["close"].pct_change()
df["spy_return"] = df["spy_close"].pct_change()
df["qqq_return"] = df["qqq_close"].pct_change()

# Volatility
df["volatility"] = df["return"].rolling(5).std()

# Moving averages
df["ma_5"] = df["close"].rolling(5).mean()
df["ma_10"] = df["close"].rolling(10).mean()
df["ma_20"] = df["close"].rolling(20).mean()

# Lag features
df["close_lag_1"] = df["close"].shift(1)
df["close_lag_2"] = df["close"].shift(2)
df["close_lag_3"] = df["close"].shift(3)

df["return_lag_1"] = df["return"].shift(1)
df["return_lag_2"] = df["return"].shift(2)
df["return_lag_3"] = df["return"].shift(3)

# Intraday structure
df["high_low_range"] = df["high"] - df["low"]
df["open_close_range"] = df["close"] - df["open"]
df["price_range_pct"] = (df["high"] - df["low"]) / df["close"]

# Momentum
df["momentum_3"] = df["close"] - df["close"].shift(3)
df["momentum_5"] = df["close"] - df["close"].shift(5)

# RSI
delta = df["close"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()
rs = gain / loss
df["rsi"] = 100 - (100 / (1 + rs))

# Volume intelligence
df["vol_ma_5"] = df["volume"].rolling(5).mean()
df["vol_ratio"] = df["volume"] / df["vol_ma_5"]

# =========================
# 4. TARGETS
# =========================
# Regression target: next-day return
df["target_return"] = df["close"].shift(-1) / df["close"] - 1

# Classification target: next-day direction
df["target_direction"] = (df["close"].shift(-1) > df["close"]).astype(int)

df = df.dropna().reset_index(drop=True)


# =========================
# 5. FEATURES
# =========================
features = [
    "close",
    "volume",
    "volatility",
    "ma_5",
    "ma_10",
    "ma_20",
    "close_lag_1",
    "close_lag_2",
    "close_lag_3",
    "return_lag_1",
    "return_lag_2",
    "return_lag_3",
    "high_low_range",
    "open_close_range",
    "price_range_pct",
    "momentum_3",
    "momentum_5",
    "rsi",
    "vol_ma_5",
    "vol_ratio",
    "spy_return",
    "qqq_return",
]

X = df[features]
y_reg = df["target_return"]
y_clf = df["target_direction"]


# =========================
# 6. WALK-FORWARD VALIDATION
# =========================
tscv = TimeSeriesSplit(n_splits=5)

reg_preds = []
reg_actuals = []

clf_preds = []
clf_actuals = []

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train_reg, y_test_reg = y_reg.iloc[train_idx], y_reg.iloc[test_idx]
    y_train_clf, y_test_clf = y_clf.iloc[train_idx], y_clf.iloc[test_idx]

    # =========================
    # 7. REGRESSION MODEL (XGBOOST)
    # =========================
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

    # Convert predicted return -> predicted price
    pred_prices = X_test["close"].values * (1 + pred_returns)
    actual_prices = X_test["close"].values * (1 + y_test_reg.values)

    reg_preds.extend(pred_prices)
    reg_actuals.extend(actual_prices)

    # =========================
    # 8. CLASSIFICATION MODEL (DIRECTION)
    # =========================
    clf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=42
    )
    clf_model.fit(X_train, y_train_clf)

    pred_dir = clf_model.predict(X_test)

    clf_preds.extend(pred_dir)
    clf_actuals.extend(y_test_clf)


# =========================
# 9. METRICS
# =========================
mae = mean_absolute_error(reg_actuals, reg_preds)
direction_acc = accuracy_score(clf_actuals, clf_preds)

print(f"\n{SYMBOL} Results")
print("=" * 40)
print(f"Price MAE           : ${mae:.2f}")
print(f"Direction Accuracy  : {direction_acc * 100:.2f}%")


# =========================
# 10. PLOT
# =========================
plt.figure(figsize=(12, 6))
plt.plot(reg_actuals, label="Actual Price")
plt.plot(reg_preds, label="Predicted Price")
plt.title(f"{SYMBOL} Price Prediction (XGBoost + Walk Forward)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()


# =========================
# 11. NEXT DAY FORECAST
# =========================
latest_data = X.iloc[-1:]
latest_close = df["close"].iloc[-1]

next_return_pred = reg_model.predict(latest_data)[0]
next_price_pred = latest_close * (1 + next_return_pred)

next_direction_pred = clf_model.predict(latest_data)[0]
direction_label = "UP" if next_direction_pred == 1 else "DOWN"

print(f"Next Day Predicted Close : ${next_price_pred:.2f}")
print(f"Next Day Direction       : {direction_label}")