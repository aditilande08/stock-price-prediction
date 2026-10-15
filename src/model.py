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


def render_prophet_forecast_plot(model: Prophet, forecast: pd.DataFrame, ticker: str = "Stock"):
    """
    Render custom interactive Plotly forecast plot compatible with Pandas 2+.
    Avoids deprecated 'assert m.history' in prophet.plot.plot_plotly.
    """
    import plotly.graph_objects as go
    fig = go.Figure()

    # Actual historical points
    if hasattr(model, 'history') and model.history is not None:
        fig.add_trace(go.Scatter(
            x=model.history['ds'],
            y=model.history['y'],
            name='Actual Close Price',
            mode='markers',
            marker=dict(color='#0f172a', size=3)
        ))

    # Upper bound
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_upper'],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Lower bound with fill
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_lower'],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(37, 99, 235, 0.18)',
        name='80% Confidence Band'
    ))

    # Projected trendline
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        name='Projected Trend (yhat)',
        mode='lines',
        line=dict(color='#2563eb', width=2.5)
    ))

    fig.update_layout(
        title=f"Meta Prophet Price Forecast: {ticker}",
        xaxis_title="Date",
        yaxis_title="Stock Price (USD)",
        template="plotly_white",
        hovermode="x unified",
        xaxis_rangeslider_visible=True,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


def render_prophet_components_plot(model: Prophet, forecast: pd.DataFrame):
    """
    Render custom 3-subplot Plotly decomposition figure: Trend, Weekly, and Yearly.
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    components = ["trend"]
    titles = ["Overall Secular Trend"]

    if "weekly" in forecast.columns:
        components.append("weekly")
        titles.append("Weekly Seasonality (Day-of-Week Effect)")
    if "yearly" in forecast.columns:
        components.append("yearly")
        titles.append("Yearly Seasonality (Month-of-Year Effect)")

    fig = make_subplots(
        rows=len(components), cols=1,
        shared_xaxes=False,
        vertical_spacing=0.14,
        subplot_titles=titles
    )

    for idx, comp in enumerate(components, start=1):
        if comp == "trend":
            fig.add_trace(
                go.Scatter(x=forecast['ds'], y=forecast['trend'], name="Trend", line=dict(color="#2563eb", width=2)),
                row=idx, col=1
            )
        elif comp == "weekly":
            temp = forecast.copy()
            temp['day_name'] = temp['ds'].dt.day_name()
            order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
            weekly_mean = temp.groupby('day_name')['weekly'].mean().reindex(order).dropna()
            fig.add_trace(
                go.Scatter(x=weekly_mean.index, y=weekly_mean.values, name="Weekly", line=dict(color="#10b981", width=2)),
                row=idx, col=1
            )
        elif comp == "yearly":
            temp = forecast.copy()
            temp['day_of_year'] = temp['ds'].dt.dayofyear
            yearly_curve = temp.groupby('day_of_year')['yearly'].mean().sort_index().rolling(7, min_periods=1).mean()
            ref_dates = pd.date_range("2024-01-01", periods=len(yearly_curve), freq="D")
            fig.add_trace(
                go.Scatter(x=ref_dates, y=yearly_curve.values, name="Yearly", line=dict(color="#f59e0b", width=2)),
                row=idx, col=1
            )

    fig.update_layout(
        template="plotly_white",
        height=260 * len(components),
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

