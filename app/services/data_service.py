# app/services/data_service.py
import pandas as pd
from alpaca.data.timeframe import TimeFrame
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from app.core.config import ALPACA_API_KEY, ALPACA_API_SECRET

client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)

def fetch_bars(symbol: str, start: str, end: str) -> pd.DataFrame:
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start,
        end=end
    )
    bars = client.get_stock_bars(request)
    df = bars.df.reset_index()

    if "symbol" in df.columns:
        df = df[df["symbol"] == symbol]

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df