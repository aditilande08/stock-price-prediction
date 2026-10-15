"""
Unit Tests for Stock Price Prediction & Forecasting Platform
Verifies data preparation, indicator calculations, and Prophet model training.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.model import (
    calculate_technical_indicators,
    prepare_prophet_data,
    train_prophet_model,
    generate_forecast,
    evaluate_forecast
)


class TestStockModel(unittest.TestCase):
    """Test suite for core modeling and financial indicators."""

    def setUp(self):
        """Generate synthetic mock daily stock price data."""
        np.random.seed(42)
        n_days = 200
        dates = pd.date_range(start="2024-01-01", periods=n_days, freq="B")
        
        # Synthetic random walk
        prices = 150 + np.cumsum(np.random.randn(n_days) * 2)
        
        self.mock_df = pd.DataFrame({
            "Date": dates,
            "Open": prices + np.random.randn(n_days) * 0.5,
            "High": prices + 2.0,
            "Low": prices - 2.0,
            "Close": prices,
            "Volume": np.random.randint(1000000, 5000000, size=n_days)
        })

    def test_technical_indicators_created(self):
        """Check moving averages, volatility, and RSI calculations."""
        df_ind = calculate_technical_indicators(self.mock_df)
        self.assertIn("SMA_20", df_ind.columns)
        self.assertIn("SMA_50", df_ind.columns)
        self.assertIn("EMA_20", df_ind.columns)
        self.assertIn("Daily_Return", df_ind.columns)
        self.assertIn("Volatility_30D", df_ind.columns)
        self.assertIn("RSI_14", df_ind.columns)
        
        # Verify valid numerical ranges
        valid_rsi = df_ind["RSI_14"].dropna()
        self.assertTrue((valid_rsi >= 0).all() and (valid_rsi <= 100).all())

    def test_prepare_prophet_data(self):
        """Check conversion to ds and y format required by Prophet."""
        prophet_df = prepare_prophet_data(self.mock_df)
        self.assertIn("ds", prophet_df.columns)
        self.assertIn("y", prophet_df.columns)
        self.assertEqual(len(prophet_df), len(self.mock_df))
        self.assertFalse(prophet_df["y"].isna().any())

    def test_prophet_training_and_forecast(self):
        """Verify model fits and generates forecast dataframe."""
        prophet_df = prepare_prophet_data(self.mock_df)
        model = train_prophet_model(prophet_df)
        forecast = generate_forecast(model, periods=30)
        
        self.assertIn("ds", forecast.columns)
        self.assertIn("yhat", forecast.columns)
        self.assertIn("yhat_lower", forecast.columns)
        self.assertIn("yhat_upper", forecast.columns)
        self.assertEqual(len(forecast), len(prophet_df) + 30)

    def test_backtest_evaluation(self):
        """Verify backtest returns MAE, RMSE, and MAPE."""
        res = evaluate_forecast(self.mock_df, test_days=30)
        self.assertNotIn("error", res)
        self.assertIn("mae", res)
        self.assertIn("rmse", res)
        self.assertIn("mape", res)
        self.assertGreater(res["mae"], 0)
        self.assertGreater(res["rmse"], 0)


if __name__ == "__main__":
    unittest.main()
