import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import math
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="Stock Analysis & Forecasting", layout="wide")

# -----------------------
# Load Data
# -----------------------
@st.cache_data
def load_data():
    stocks = {}
    for name, file in [('Google', 'google.csv'), 
                       ('Microsoft', 'msft.csv'), 
                       ('Amazon', 'amzn.csv'), 
                       ('IBM', 'ibm.csv')]:
        df = pd.read_csv(file, parse_dates=['Date'], index_col='Date')
        df.dropna(inplace=True)
        stocks[name] = df
    return stocks

stocks = load_data()

# -----------------------
# Sidebar for User Selection
# -----------------------
st.sidebar.title("Settings")
stock_choice = st.sidebar.selectbox("Select Stock:", list(stocks.keys()))
column_choice = st.sidebar.selectbox("Select Column to Analyze:", ['Open','High','Low','Close','Volume'])

st.header(f"📊 {stock_choice} Stock Analysis")

df = stocks[stock_choice]

# -----------------------
# Show raw data
# -----------------------
if st.checkbox(f"Show {stock_choice} Data"):
    st.write(df.head())

# -----------------------
# Distribution Plot
# -----------------------
fig = px.histogram(df, x=column_choice, marginal='box', nbins=50,
                   title=f'{stock_choice} {column_choice} Distribution')
st.plotly_chart(fig, use_container_width=True)

# -----------------------
# Correlation Matrix
# -----------------------
if st.checkbox("Show Correlation Matrix"):
    numeric_cols = df.select_dtypes(include='number')
    corr = numeric_cols.corr()
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='Viridis', title=f'{stock_choice} Correlation Matrix')
    st.plotly_chart(fig_corr, use_container_width=True)

# -----------------------
# Time Series Visualization
# -----------------------
st.subheader(f"{stock_choice} Time Series Plot")
fig_ts = px.line(df, x=df.index, y=column_choice, title=f'{stock_choice} {column_choice} Over Time')
st.plotly_chart(fig_ts, use_container_width=True)

# -----------------------
# Comparison of High Prices
# -----------------------
if st.checkbox("Compare Normalized High Prices Across Stocks"):
    normalized = {}
    for name, df_s in stocks.items():
        normalized[name] = df_s['High'] / df_s['High'].iloc[0] * 100
    df_norm = pd.DataFrame(normalized)
    fig_norm = px.line(df_norm, title="Normalized High Prices Comparison")
    st.plotly_chart(fig_norm, use_container_width=True)

# -----------------------
# Trend and Seasonality
# -----------------------
if st.checkbox("Show Trend and Seasonality (High)"):
    decomposition = seasonal_decompose(df['High'], period=360)
    st.subheader("Trend")
    fig_trend = px.line(decomposition.trend, title=f"{stock_choice} Trend")
    st.plotly_chart(fig_trend, use_container_width=True)
    
    st.subheader("Seasonality")
    fig_season = px.line(decomposition.seasonal, title=f"{stock_choice} Seasonality")
    st.plotly_chart(fig_season, use_container_width=True)

# -----------------------
# GRU Prediction
# -----------------------
if st.checkbox("Show GRU Forecast"):
    lookback = 20
    scaler = MinMaxScaler(feature_range=(-1,1))
    price_data = df[['Close']].copy()
    price_data['Close'] = scaler.fit_transform(price_data)
    
    # Split data
    def split_data(stock, lookback):
        data_raw = stock.to_numpy()
        data = []
        for i in range(len(data_raw)-lookback):
            data.append(data_raw[i:i+lookback])
        data = np.array(data)
        test_size = int(np.round(0.2*data.shape[0]))
        train_size = data.shape[0]-test_size
        x_train = data[:train_size,:-1,:]
        y_train = data[:train_size,-1,:]
        x_test = data[train_size:,:-1,:]
        y_test = data[train_size:,-1,:]
        return x_train, y_train, x_test, y_test

    x_train, y_train, x_test, y_test = split_data(price_data, lookback)
    x_train_t = torch.from_numpy(x_train).float()
    x_test_t = torch.from_numpy(x_test).float()
    y_train_t = torch.from_numpy(y_train).float()
    y_test_t = torch.from_numpy(y_test).float()
    
    # GRU Model
    class GRU(nn.Module):
        def __init__(self, input_dim=1, hidden_dim=32, num_layers=2, output_dim=1):
            super(GRU, self).__init__()
            self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)
        def forward(self, x):
            h0 = torch.zeros(2, x.size(0), 32)
            out, hn = self.gru(x, h0)
            out = self.fc(out[:, -1, :])
            return out
    
    model = GRU()
    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Training
    epochs = 50
    for t in range(epochs):
        y_pred = model(x_train_t)
        loss = criterion(y_pred, y_train_t)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    
    # Predictions
    y_train_pred = scaler.inverse_transform(model(x_train_t).detach().numpy())
    y_test_pred = scaler.inverse_transform(model(x_test_t).detach().numpy())
    y_true = scaler.inverse_transform(y_test_t.numpy())
    
    # Plot
    df_pred = pd.DataFrame({'Train': y_train_pred.flatten(), 'Test': np.concatenate([np.full(lookback, np.nan), y_test_pred.flatten()]), 'Actual': scaler.inverse_transform(price_data).flatten()})
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(y=df_pred['Train'], mode='lines', name='Train Prediction'))
    fig_pred.add_trace(go.Scatter(y=df_pred['Test'], mode='lines', name='Test Prediction'))
    fig_pred.add_trace(go.Scatter(y=df_pred['Actual'], mode='lines', name='Actual'))
    fig_pred.update_layout(title=f"{stock_choice} GRU Forecast", xaxis_title='Time', yaxis_title='Close (USD)', template='plotly_dark')
    st.plotly_chart(fig_pred, use_container_width=True)
    
    train_rmse = math.sqrt(mean_squared_error(scaler.inverse_transform(y_train_t), y_train_pred))
    test_rmse = math.sqrt(mean_squared_error(y_true, y_test_pred))
    st.write(f"Train RMSE: {train_rmse:.2f} | Test RMSE: {test_rmse:.2f}")
