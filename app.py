"""
Stock Price Prediction & Time-Series Forecasting Platform
Built with Python, Streamlit, yfinance, Meta Prophet, and Plotly
"""

import os
import sys
from datetime import date, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(__file__))
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

# Page Configuration
st.set_page_config(
    page_title="Stock Price Prediction & Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    * { font-family: 'Inter', sans-serif; }
    
    .main-title {
        font-size: 28px;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 2px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .sub-title {
        font-size: 14px;
        color: #64748b;
        margin-bottom: 20px;
        font-weight: 500;
    }
    .stat-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .stat-label {
        font-size: 12px;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stat-val {
        font-size: 26px;
        font-weight: 800;
        color: #0f172a;
        margin-top: 4px;
    }
    .stat-change-pos {
        color: #16a34a;
        font-size: 13px;
        font-weight: 700;
    }
    .stat-change-neg {
        color: #dc2626;
        font-size: 13px;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# ─── Sidebar Controls ─────────────────────────────────────────
st.sidebar.markdown("### ⚙️ Asset & Forecast Settings")

asset_mode = st.sidebar.radio("Select Asset Input:", ["Popular Assets", "Custom Ticker Symbol"], horizontal=True)

if asset_mode == "Popular Assets":
    selected_label = st.sidebar.selectbox("Choose Equity / Index:", list(POPULAR_TICKERS.keys()), index=0)
    selected_ticker = POPULAR_TICKERS[selected_label]
else:
    custom_input = st.sidebar.text_input("Enter Ticker (e.g. BTC-USD, AMD, TSLA):", value="NVDA")
    selected_ticker = custom_input.strip().upper() if custom_input else "AAPL"
    selected_label = selected_ticker

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔮 Forecast Horizon")

col_p1, col_p2 = st.sidebar.columns(2)
with col_p1:
    if st.button("1 Month (30d)"):
        st.session_state["days_horizon"] = 30
    if st.button("6 Months (180d)"):
        st.session_state["days_horizon"] = 180
with col_p2:
    if st.button("1 Year (365d)"):
        st.session_state["days_horizon"] = 365
    if st.button("2 Years (730d)"):
        st.session_state["days_horizon"] = 730

default_days = st.session_state.get("days_horizon", 365)
forecast_days = st.sidebar.slider(
    "Prediction Horizon (Days):",
    min_value=30,
    max_value=1460,
    value=default_days,
    step=30
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🛠️ Prophet Hyperparameters")
changepoint_scale = st.sidebar.slider(
    "Changepoint Prior Scale (Flexibility):",
    min_value=0.01,
    max_value=0.50,
    value=0.05,
    step=0.01,
    help="Higher values make trend more flexible to fluctuations; lower values prevent overfitting."
)
seasonality_mode = st.sidebar.selectbox(
    "Seasonality Mode:",
    ["additive", "multiplicative"],
    index=0,
    help="Use additive for constant seasonality magnitude; multiplicative when seasonality grows with price."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About Project")
st.sidebar.info("""
**Stock Price Prediction Platform**
- **Model:** Meta Prophet Time-Series
- **Source:** Yahoo Finance (Real-Time API)
- **Author:** Aditi Ashok Lande
- **University:** VIT Bhopal University
""")


# ─── Header ───────────────────────────────────────────────────
st.markdown("<div class='main-title'>📈 Stock Price Prediction & Forecasting Platform</div>", unsafe_allow_html=True)
st.markdown(f"<div class='sub-title'>Time-Series Analysis & Trend Projections for <b>{selected_label}</b> via Meta Prophet</div>", unsafe_allow_html=True)


# ─── Load Data with Caching ───────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_cached_stock(ticker: str):
    start_date = "2018-01-01"
    today_str = date.today().strftime("%Y-%m-%d")
    df = fetch_stock_data(ticker, start=start_date, end=today_str)
    df_indicators = calculate_technical_indicators(df)
    return df_indicators


with st.spinner(f"Fetching real-time market data for {selected_ticker}..."):
    try:
        raw_df = load_cached_stock(selected_ticker)
    except Exception as e:
        st.error(f"❌ Error fetching stock data for '{selected_ticker}': {str(e)}")
        st.info("Tip: Verify the ticker symbol on Yahoo Finance (e.g. 'AAPL' for Apple, 'TCS.NS' for Tata Consultancy Services).")
        st.stop()


# ─── Top KPI Metric Cards ─────────────────────────────────────
latest = raw_df.iloc[-1]
prev = raw_df.iloc[-2] if len(raw_df) > 1 else latest

current_price = latest["Close"]
daily_change = current_price - prev["Close"]
daily_pct = (daily_change / (prev["Close"] + 1e-9)) * 100
high_52 = raw_df["Close"].tail(252).max()
low_52 = raw_df["Close"].tail(252).min()
volatility = latest["Volatility_30D"] if not np.isnan(latest["Volatility_30D"]) else 0.0

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class='stat-card'>
        <div class='stat-label'>Latest Close</div>
        <div class='stat-val'>${current_price:,.2f}</div>
        <div class='{"stat-change-pos" if daily_change >= 0 else "stat-change-neg"}'>
            {'▲' if daily_change >= 0 else '▼'} ${abs(daily_change):.2f} ({daily_pct:+.2f}%)
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='stat-card'>
        <div class='stat-label'>52-Week High</div>
        <div class='stat-val'>${high_52:,.2f}</div>
        <div style='color:#64748b; font-size:12px; margin-top:2px;'>Peak Resistance</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class='stat-card'>
        <div class='stat-label'>52-Week Low</div>
        <div class='stat-val'>${low_52:,.2f}</div>
        <div style='color:#64748b; font-size:12px; margin-top:2px;'>Key Support</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class='stat-card'>
        <div class='stat-label'>30D Volatility</div>
        <div class='stat-val'>{volatility:.1f}%</div>
        <div style='color:#64748b; font-size:12px; margin-top:2px;'>Annualized StDev</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class='stat-card'>
        <div class='stat-label'>Historical Records</div>
        <div class='stat-val'>{len(raw_df):,}</div>
        <div style='color:#64748b; font-size:12px; margin-top:2px;'>Trading Sessions</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)


# ─── Main Tabs ────────────────────────────────────────────────
tab_forecast, tab_history, tab_components, tab_eval, tab_data = st.tabs([
    "🔮 Prophet Forecast",
    "📈 Historical Chart & Indicators",
    "📊 Seasonality & Trend Components",
    "🎯 Model Evaluation & Backtest",
    "📋 Raw Dataset"
])


# ─── Tab 1: Prophet Forecast ──────────────────────────────────
with tab_forecast:
    st.markdown(f"#### 🔮 Projected Price Path ({forecast_days} Days Ahead)")
    
    with st.spinner("Training Meta Prophet model on historical time-series..."):
        prophet_input = prepare_prophet_data(raw_df)
        model = train_prophet_model(
            prophet_input,
            changepoint_prior_scale=changepoint_scale,
            seasonality_mode=seasonality_mode
        )
        forecast_df = generate_forecast(model, periods=forecast_days)
        
    # Projected targets
    target_future = forecast_df.iloc[-1]
    expected_price = target_future["yhat"]
    expected_lower = target_future["yhat_lower"]
    expected_upper = target_future["yhat_upper"]
    overall_growth = ((expected_price - current_price) / current_price) * 100
    
    col_fc1, col_fc2, col_fc3 = st.columns(3)
    with col_fc1:
        st.metric(
            label=f"Projected Price ({target_future['ds'].strftime('%b %d, %Y')})",
            value=f"${expected_price:,.2f}",
            delta=f"{overall_growth:+.2f}% vs. Today"
        )
    with col_fc2:
        st.metric(
            label="Confidence Interval (Lower Bound)",
            value=f"${expected_lower:,.2f}",
            delta="Pessimistic Scenario"
        )
    with col_fc3:
        st.metric(
            label="Confidence Interval (Upper Bound)",
            value=f"${expected_upper:,.2f}",
            delta="Optimistic Scenario"
        )
        
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    
    # Interactive Plotly Chart
    fig_fc = render_prophet_forecast_plot(model, forecast_df, selected_ticker)
    st.plotly_chart(fig_fc, use_container_width=True)
    
    st.caption("ℹ️ Black dots represent actual historical close prices; dark blue line is Prophet's median expected trend; light blue shaded area indicates 80% uncertainty bounds.")


# ─── Tab 2: Historical Chart & Indicators ─────────────────────
with tab_history:
    st.markdown("#### 📈 Price History & Technical Indicators")
    
    col_toggles1, col_toggles2 = st.columns([1, 1])
    with col_toggles1:
        chart_type = st.radio("Chart Style:", ["Candlestick", "Line Chart"], horizontal=True)
    with col_toggles2:
        show_sma20 = st.checkbox("Show 20-Day SMA", value=True)
        show_sma50 = st.checkbox("Show 50-Day SMA", value=True)
        show_ema20 = st.checkbox("Show 20-Day EMA", value=False)
        
    fig_hist = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Price Movement & Moving Averages", "Daily Trading Volume"),
        row_heights=[0.75, 0.25]
    )
    
    if chart_type == "Candlestick" and {"Open", "High", "Low", "Close"}.issubset(raw_df.columns):
        fig_hist.add_trace(
            go.Candlestick(
                x=raw_df["Date"],
                open=raw_df["Open"],
                high=raw_df["High"],
                low=raw_df["Low"],
                close=raw_df["Close"],
                name="OHLC"
            ),
            row=1, col=1
        )
    else:
        fig_hist.add_trace(
            go.Scatter(
                x=raw_df["Date"],
                y=raw_df["Close"],
                name="Close Price",
                line=dict(color="#2563eb", width=2)
            ),
            row=1, col=1
        )
        
    if show_sma20 and "SMA_20" in raw_df.columns:
        fig_hist.add_trace(go.Scatter(x=raw_df["Date"], y=raw_df["SMA_20"], name="SMA 20", line=dict(color="#f59e0b", width=1.5)), row=1, col=1)
    if show_sma50 and "SMA_50" in raw_df.columns:
        fig_hist.add_trace(go.Scatter(x=raw_df["Date"], y=raw_df["SMA_50"], name="SMA 50", line=dict(color="#10b981", width=1.5)), row=1, col=1)
    if show_ema20 and "EMA_20" in raw_df.columns:
        fig_hist.add_trace(go.Scatter(x=raw_df["Date"], y=raw_df["EMA_20"], name="EMA 20", line=dict(color="#8b5cf6", width=1.5, dash="dot")), row=1, col=1)
        
    if "Volume" in raw_df.columns:
        colors = ['#16a34a' if c >= o else '#dc2626' for c, o in zip(raw_df["Close"], raw_df["Open"])]
        fig_hist.add_trace(go.Bar(x=raw_df["Date"], y=raw_df["Volume"], name="Volume", marker_color=colors), row=2, col=1)
        
    fig_hist.update_layout(
        template="plotly_white",
        height=620,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis2_rangeslider_visible=False
    )
    st.plotly_chart(fig_hist, use_container_width=True)


# ─── Tab 3: Seasonality & Trend Components ────────────────────
with tab_components:
    st.markdown("#### 📊 Prophet Decomposition: Trend & Seasonality")
    st.write("Prophet decomposes time-series into long-term secular growth, weekly calendar cycles, and yearly seasonality:")
    
    fig_comp = render_prophet_components_plot(model, forecast_df)
    st.plotly_chart(fig_comp, use_container_width=True)


# ─── Tab 4: Model Evaluation & Backtesting ────────────────────
with tab_eval:
    st.markdown("#### 🎯 Backtest Evaluation on Recent 60 Trading Days")
    st.write("To test reliability, we withhold the most recent 60 trading days, fit Prophet on past data, and measure accuracy against what actually occurred:")
    
    with st.spinner("Computing out-of-sample backtest metrics..."):
        eval_metrics = evaluate_forecast(raw_df, test_days=60)
        
    if "error" in eval_metrics:
        st.warning(eval_metrics["error"])
    else:
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Mean Absolute Error (MAE)", f"${eval_metrics['mae']}")
        with col_m2:
            st.metric("Root Mean Squared Error (RMSE)", f"${eval_metrics['rmse']}")
        with col_m3:
            st.metric("Mean Absolute % Error (MAPE)", f"{eval_metrics['mape']}%")
            
        fig_eval = go.Figure()
        fig_eval.add_trace(go.Scatter(
            x=eval_metrics["actual_dates"],
            y=eval_metrics["y_true"],
            name="Actual Close Price",
            line=dict(color="#0f172a", width=2.5)
        ))
        fig_eval.add_trace(go.Scatter(
            x=eval_metrics["actual_dates"],
            y=eval_metrics["y_pred"],
            name="Prophet Backtest Prediction",
            line=dict(color="#3b82f6", width=2, dash="dash")
        ))
        fig_eval.update_layout(
            title="Out-of-Sample Backtest: Actual vs. Predicted (Last 60 Days)",
            xaxis_title="Date",
            yaxis_title="Stock Price (USD)",
            template="plotly_white",
            height=450,
            hovermode="x unified"
        )
        st.plotly_chart(fig_eval, use_container_width=True)


# ─── Tab 5: Raw Dataset ───────────────────────────────────────
with tab_data:
    st.markdown("#### 📋 Raw Stock Quotes & Predictions")
    
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("**Historical Market Records (Recent 10 Days)**")
        st.dataframe(raw_df.tail(10), use_container_width=True)
        csv_hist = raw_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Historical Data (CSV)",
            data=csv_hist,
            file_name=f"{selected_ticker}_historical_data.csv",
            mime="text/csv"
        )
        
    with col_d2:
        st.markdown("**Forecast Projections (Next 10 Days)**")
        st.dataframe(forecast_df[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(10), use_container_width=True)
        csv_fc = forecast_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Forecast Data (CSV)",
            data=csv_fc,
            file_name=f"{selected_ticker}_prophet_forecast.csv",
            mime="text/csv"
        )
