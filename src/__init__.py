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
    evaluate_forecast,
    render_prophet_forecast_plot,
    render_prophet_components_plot
)

__all__ = [
    "POPULAR_TICKERS",
    "fetch_stock_data",
    "calculate_technical_indicators",
    "prepare_prophet_data",
    "train_prophet_model",
    "generate_forecast",
    "evaluate_forecast",
    "render_prophet_forecast_plot",
    "render_prophet_components_plot"
]
