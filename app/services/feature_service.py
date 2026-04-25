# app/services/feature_service.py
import pandas as pd

FEATURE_COLUMNS = [
    "close", "volume", "volatility", "ma_5", "ma_10", "ma_20",
    "close_lag_1", "close_lag_2", "close_lag_3",
    "return_lag_1", "return_lag_2", "return_lag_3",
    "high_low_range", "open_close_range", "price_range_pct",
    "momentum_3", "momentum_5", "rsi",
    "vol_ma_5", "vol_ratio", "spy_return", "qqq_return"
]

def build_features(df: pd.DataFrame, spy: pd.DataFrame, qqq: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(spy[["timestamp", "close"]].rename(columns={"close": "spy_close"}), on="timestamp", how="left")
    df = df.merge(qqq[["timestamp", "close"]].rename(columns={"close": "qqq_close"}), on="timestamp", how="left")

    df["return"] = df["close"].pct_change()
    df["spy_return"] = df["spy_close"].pct_change()
    df["qqq_return"] = df["qqq_close"].pct_change()

    df["volatility"] = df["return"].rolling(5).std()

    df["ma_5"] = df["close"].rolling(5).mean()
    df["ma_10"] = df["close"].rolling(10).mean()
    df["ma_20"] = df["close"].rolling(20).mean()

    df["close_lag_1"] = df["close"].shift(1)
    df["close_lag_2"] = df["close"].shift(2)
    df["close_lag_3"] = df["close"].shift(3)

    df["return_lag_1"] = df["return"].shift(1)
    df["return_lag_2"] = df["return"].shift(2)
    df["return_lag_3"] = df["return"].shift(3)

    df["high_low_range"] = df["high"] - df["low"]
    df["open_close_range"] = df["close"] - df["open"]
    df["price_range_pct"] = (df["high"] - df["low"]) / df["close"]

    df["momentum_3"] = df["close"] - df["close"].shift(3)
    df["momentum_5"] = df["close"] - df["close"].shift(5)

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    df["vol_ma_5"] = df["volume"].rolling(5).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma_5"]

    return df.dropna().reset_index(drop=True)