"""
Stock Price Prediction Package
"""
from src.model import (
    POPULAR_TICKERS,
    fetch_stock_data,
    calculate_technical_indicators,
    prepare_prophet_data,
    train_prophet_model,
    generate_forecast,
    evaluate_forecast
)

__all__ = [
    "POPULAR_TICKERS",
    "fetch_stock_data",
    "calculate_technical_indicators",
    "prepare_prophet_data",
    "train_prophet_model",
    "generate_forecast",
    "evaluate_forecast"
]
