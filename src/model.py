"""
Core Financial Data Processing & Forecasting Engine
Stock Price Prediction Platform
Powered by Yahoo Finance (yfinance) and Meta Prophet
"""

from datetime import datetime, date
import numpy as np
import pandas as pd
import yfinance as yf
from prophet import Prophet


POPULAR_TICKERS = {
    "Apple Inc. (AAPL)": "AAPL",
    "Microsoft Corp. (MSFT)": "MSFT",
    "Alphabet Inc. (GOOGL)": "GOOGL",
    "Amazon.com Inc. (AMZN)": "AMZN",
    "Tesla Inc. (TSLA)": "TSLA",
    "NVIDIA Corp. (NVDA)": "NVDA",
    "Meta Platforms (META)": "META",
    "GameStop Corp. (GME)": "GME",
    "S&P 500 ETF (SPY)": "SPY",
    "Reliance Industries (NSE)": "RELIANCE.NS",
    "Tata Consultancy Services (NSE)": "TCS.NS",
    "Infosys Limited (NSE)": "INFY.NS"
}


def fetch_stock_data(ticker: str, start: str = "2018-01-01", end: str = None) -> pd.DataFrame:
    """
    Fetch historical daily OHLCV stock data via yfinance.
    
    Args:
        ticker: Equity ticker symbol (e.g. 'AAPL', 'MSFT', 'TCS.NS')
        start: Start date string (YYYY-MM-DD)
        end: End date string (YYYY-MM-DD), defaults to today
        
    Returns:
        Pandas DataFrame with Date, Open, High, Low, Close, Volume
    """
    if not end:
        end = date.today().strftime("%Y-%m-%d")
        
    ticker_clean = ticker.strip().upper()
    df = yf.download(ticker_clean, start=start, end=end, progress=False)
    
    if df is None or df.empty:
        raise ValueError(f"No financial data returned for ticker symbol '{ticker}'. Please verify the symbol.")
        
    # Flatten MultiIndex columns if present (from recent yfinance versions)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
        
    df.reset_index(inplace=True)
    
    # Ensure Date column is datetime
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    elif "Datetime" in df.columns:
        df["Date"] = pd.to_datetime(df["Datetime"]).dt.tz_localize(None)
        
    # Drop rows with NaN close prices
    df = df.dropna(subset=["Close"]).reset_index(drop=True)
    return df


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute essential technical indicators:
    - Simple Moving Averages: SMA 20, SMA 50, SMA 200
    - Exponential Moving Average: EMA 20
    - Relative Strength Index (RSI 14)
    - Daily Return (%) and Rolling 30-day Volatility (%)
    """
    data = df.copy()
    close = data["Close"]
    
    data["SMA_20"] = close.rolling(window=20).mean()
    data["SMA_50"] = close.rolling(window=50).mean()
    data["SMA_200"] = close.rolling(window=200).mean()
    data["EMA_20"] = close.ewm(span=20, adjust=False).mean()
    
    # Daily Percentage Return
    data["Daily_Return"] = close.pct_change() * 100
    
    # 30-day Annualized Volatility
    data["Volatility_30D"] = data["Daily_Return"].rolling(window=30).std() * np.sqrt(252)
    
    # Relative Strength Index (RSI - 14)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    data["RSI_14"] = 100 - (100 / (1 + rs))
    
    return data


def prepare_prophet_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format dataframe into Prophet's required schema: 'ds' (datestamp) and 'y' (target price).
    """
    if "Date" not in df.columns or "Close" not in df.columns:
        raise ValueError("DataFrame must contain 'Date' and 'Close' columns.")
        
    prophet_df = pd.DataFrame({
        "ds": pd.to_datetime(df["Date"]),
        "y": df["Close"].astype(float)
    })
    return prophet_df.dropna().sort_values("ds").reset_index(drop=True)


def train_prophet_model(df_prophet: pd.DataFrame, changepoint_prior_scale: float = 0.05, seasonality_mode: str = "additive") -> Prophet:
    """
    Instantiate and train a Meta Prophet time-series model.
    """
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_mode=seasonality_mode
    )
    model.fit(df_prophet)
    return model


def generate_forecast(model: Prophet, periods: int = 365) -> pd.DataFrame:
    """
    Generate future date predictions using the fitted Prophet model.
    
    Returns:
        DataFrame containing 'ds', 'yhat' (prediction), 'yhat_lower', 'yhat_upper'
    """
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast


def evaluate_forecast(df: pd.DataFrame, test_days: int = 60) -> dict:
    """
    Backtest the Prophet model on the most recent `test_days` to evaluate forecast accuracy.
    Computes MAE, RMSE, and MAPE metrics.
    """
    prophet_df = prepare_prophet_data(df)
    
    if len(prophet_df) <= test_days + 30:
        return {"error": "Insufficient historical data for backtesting."}
        
    train_df = prophet_df.iloc[:-test_days]
    test_df = prophet_df.iloc[-test_days:].copy()
    
    model = train_prophet_model(train_df)
    future = model.make_future_dataframe(periods=test_days)
    forecast = model.predict(future)
    
    test_forecast = forecast.iloc[-test_days:].reset_index(drop=True)
    y_true = test_df["y"].values
    y_pred = test_forecast["yhat"].values
    
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100)
    
    return {
        "test_days": test_days,
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape": round(mape, 2),
        "actual_dates": test_df["ds"].dt.strftime("%Y-%m-%d").tolist(),
        "y_true": [round(float(v), 2) for v in y_true],
        "y_pred": [round(float(v), 2) for v in y_pred]
    }
