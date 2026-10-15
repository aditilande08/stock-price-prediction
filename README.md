# 📈 Real-Time Stock Price Prediction & Forecasting Platform

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Prophet](https://img.shields.io/badge/Prophet-Meta%20AI-008080.svg)](https://facebook.github.io/prophet/)
[![yfinance](https://img.shields.io/badge/yfinance-Yahoo%20Finance-7B1FA2.svg)](https://pypi.org/project/yfinance/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F51B5.svg?logo=plotly&logoColor=white)](https://plotly.com/)
[![Tests](https://img.shields.io/badge/Tests-4%20Passed-brightgreen.svg)]()

An interactive, end-to-end financial time-series forecasting web application built with **Python**, **yfinance**, Meta's **Prophet** algorithm, and **Streamlit**. 

The platform enables users to select any global equity or index (e.g., AAPL, NVDA, TSLA, S&P 500, Reliance, TCS), inspect technical indicators (SMA, EMA, RSI, Volatility), project future price trajectories up to 4 years into the future with uncertainty bounds, and backtest model performance using out-of-sample error metrics (MAE, RMSE, MAPE).

*Inspired by Patrick Loeber's (Python Engineer) stock prediction architecture, modernized and expanded with enterprise-grade modularity, technical analysis, and model backtesting.*

---

## 📌 Table of Contents

- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [The Science: How Meta Prophet Works](#-the-science-how-meta-prophet-works)
- [Technical Indicators & Analysis](#-technical-indicators--analysis)
- [Technology Stack](#-technology-stack)
- [Project Directory Structure](#-project-directory-structure)
- [Installation & Quickstart](#-installation--quickstart)
- [Application Walkthrough](#-application-walkthrough)
- [Automated Unit Testing](#-automated-unit-testing)
- [Author & Acknowledgments](#-author--acknowledgments)

---

## 🚀 Key Features

1. **Live Financial Data Ingestion:** Real-time and multi-year historical OHLCV data streaming from the Yahoo Finance API (`yfinance`) with automatic time-zone normalization and missing-value handling.
2. **Flexible Asset Coverage:** Pre-configured with prominent US tech equities (Apple, Microsoft, Alphabet, Amazon, Tesla, Nvidia, Meta, GameStop), benchmark ETFs (S&P 500 `SPY`), Indian equities (NSE: `RELIANCE.NS`, `TCS.NS`, `INFY.NS`), and custom ticker entry for any listed asset.
3. **Prophet Time-Series Projections:** Configurable forecast horizons from **30 days to 4 years** (with quick-select buttons for 1M, 6M, 1Y, 2Y) generating median projections (`yhat`) alongside 80% upper and lower uncertainty intervals.
4. **Decomposition & Seasonality:** Automated breakdown of financial signals into secular multi-year trend lines, day-of-week calendar effects, and yearly seasonality patterns.
5. **Technical Indicators:** Moving averages (20-day SMA, 50-day SMA, 20-day EMA), 14-day Relative Strength Index (RSI), daily percentage returns, and 30-day annualized rolling volatility.
6. **Empirical Out-of-Sample Backtesting:** Withholds the most recent 60 trading sessions to evaluate forecast accuracy, reporting **Mean Absolute Error (MAE)**, **Root Mean Squared Error (RMSE)**, and **Mean Absolute Percentage Error (MAPE)**.
7. **Interactive Visualizations:** Responsive Plotly charts with candlestick overlays, volume bar charts, range slider zoom, and one-click CSV export.

---

## 🏗️ System Architecture

```
                 [ User Selection: Ticker Symbol & Horizon ]
                                     │
                                     ▼
        ┌────────────────────────────────────────────────────────┐
        │ 1. DATA INGESTION ENGINE (src/model.py)                │
        │    - yfinance API daily OHLCV query                    │
        │    - MultiIndex flattening & timezone stripping        │
        │    - Streamlit TTL caching (1 hour)                    │
        └────────────────────────────┬───────────────────────────┘
                                     │
                                     ▼
        ┌────────────────────────────────────────────────────────┐
        │ 2. QUANTITATIVE ANALYSIS & FEATURE PIPELINE            │
        │    - SMA 20, SMA 50, EMA 20 moving averages            │
        │    - RSI 14-period momentum calculation                │
        │    - 30-day rolling annualized volatility              │
        │    - Target transformation: ds (date) & y (close)      │
        └────────────────────────────┬───────────────────────────┘
                                     │
                     ┌───────────────┴───────────────┐
                     ▼                               ▼
┌────────────────────────────────────────┐ ┌────────────────────────────────────────┐
│ 3. PROPHET TIME-SERIES FORECASTING     │ │ 4. OUT-OF-SAMPLE BACKTESTING           │
│    - Generalized Additive Model (GAM)  │ │    - 60-day train/test split           │
│    - Piecewise linear trend modeling   │ │    - Actual vs. predicted evaluation   │
│    - Fourier series seasonality        │ │    - Metrics: MAE, RMSE, MAPE          │
│    - Output: yhat, yhat_lower, upper   │ │    - Validation curve plotting         │
└───────────────────┬────────────────────┘ └───────────────────┬────────────────────┘
                    │                                          │
                    └────────────────────┬─────────────────────┘
                                         │
                                         ▼
        ┌────────────────────────────────────────────────────────┐
        │ 5. INTERACTIVE STREAMLIT DASHBOARD (app.py)            │
        │    - Key Performance Indicator (KPI) cards             │
        │    - Plotly Candlestick & Volume Charts                │
        │    - Interactive Forecast with Uncertainty Bands       │
        │    - Component Decomposition & CSV Data Export         │
        └────────────────────────────────────────────────────────┘
```

---

## 🔬 The Science: How Meta Prophet Works

Prophet models time-series data using a **Generalized Additive Model (GAM)** with three main components:

$$y(t) = g(t) + s(t) + h(t) + \epsilon_t$$

- **$g(t)$ — Trend:** Piecewise linear growth model that automatically identifies changepoints where the growth rate shifts.
- **$s(t)$ — Seasonality:** Periodic changes (e.g., weekly day-of-week effects and annual seasonality) modeled using standard Fourier series.
- **$h(t)$ — Holiday Effects:** Captures recurring anomalous spikes around market holidays.
- **$\epsilon_t$ — Error Term:** Normally distributed idiosyncratic noise with variance estimated via Markov Chain Monte Carlo (MCMC).

---

## 🛠️ Technology Stack

| Component | Library | Role in System |
|---|---|---|
| **Language** | Python 3.9+ | Application logic and statistical modeling |
| **Data Ingestion** | `yfinance` | Daily OHLCV financial queries |
| **Forecasting** | `prophet` | Additive time-series trend and seasonality estimation |
| **Plotting & UI** | `plotly` | Dynamic candlestick, line, and forecast visualizations |
| **Web Framework** | `streamlit` | Responsive web dashboard and reactive state management |
| **Data Processing** | `pandas`, `numpy` | Financial transformations, RSI, volatility, moving averages |
| **Unit Testing** | `unittest` | Automated verification of data pipeline and indicator math |

---

## 📁 Project Directory Structure

```
stock-price-prediction/
├── app.py                      # Interactive Streamlit Web Application
├── requirements.txt            # Project dependencies (Streamlit, yfinance, etc.)
├── README.md                   # Comprehensive project documentation
├── .gitignore                  # Git ignore rules for virtualenvs and artifacts
├── src/
│   ├── __init__.py             # Package initializer
│   └── model.py                # Data fetching, indicators, and Prophet engine
└── tests/
    ├── __init__.py             # Test package initializer
    └── test_stock_model.py     # Automated unit test suite
```

---

## ⚙️ Installation & Quickstart

### Prerequisites
- Python 3.9, 3.10, 3.11, or 3.12 installed.

### 1. Clone or Navigate to the Directory
```powershell
cd C:\Users\aditi\.gemini\antigravity-ide\scratch\stock-price-prediction
```

### 2. Set Up a Virtual Environment (Recommended)
```powershell
python -m venv venv

# Activate on Windows PowerShell:
.\venv\Scripts\Activate.ps1

# (Or on Command Prompt):
# .\venv\Scripts\activate.bat
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Launch the Web Application
```powershell
streamlit run app.py
```
*The app will automatically open in your default browser at `http://localhost:8501`.*

---

## 🖥️ Application Walkthrough

1. **Asset Selection:** Choose a preconfigured stock from the sidebar (e.g. Apple, Nvidia, S&P 500) or enter any valid ticker symbol into the custom text box (e.g. `AMD`, `GME`, `TCS.NS`, `BTC-USD`).
2. **Forecast Horizon:** Use the slider or quick buttons (1M, 6M, 1Y, 2Y) to set the prediction target.
3. **KPI Metrics:** Observe the real-time card displays for Latest Close Price, Daily % Change, 52-Week High, 52-Week Low, and 30-Day Annualized Volatility.
4. **Interactive Charts:**
   - **Prophet Forecast Tab:** Inspect median forecast trajectory and uncertainty envelopes.
   - **Historical Chart Tab:** Toggle between Candlestick and Line views; overlay 20-Day SMA, 50-Day SMA, or 20-Day EMA.
   - **Seasonality Tab:** Explore long-term trend, day-of-week performance, and monthly seasonal bias.
   - **Backtest Tab:** Review MAE, RMSE, and MAPE errors on held-out test data.
   - **Data Tab:** Export historical and forecasted data to CSV.

---

## 🧪 Automated Unit Testing

To verify data preprocessing, indicators, and model fitting:

```powershell
python -m unittest discover -s tests
```

### Verified Test Cases:
- `test_technical_indicators_created`: Validates moving average columns and RSI bounds $[0, 100]$.
- `test_prepare_prophet_data`: Validates formatting into Prophet's required `ds` and `y` schema.
- `test_prophet_training_and_forecast`: Validates model fitting and forecast generation.
- `test_backtest_evaluation`: Validates computation of out-of-sample MAE, RMSE, and MAPE metrics.

---

## 👤 Author & Acknowledgments

- **Author:** Aditi Ashok Lande
- **Institution:** VIT Bhopal University
- **Specialization:** Computer Science & Engineering
- **GitHub:** [@aditilande08](https://github.com/aditilande08)
- **LinkedIn:** [Aditi Lande](http://linkedin.com/in/aditi-lande-9293a02ba)

### Tutorial Reference
- Inspired by Patrick Loeber's (*Python Engineer*) tutorial: [Build A Stock Prediction Web App In Python](https://youtu.be/0E_31WqVzCY).
