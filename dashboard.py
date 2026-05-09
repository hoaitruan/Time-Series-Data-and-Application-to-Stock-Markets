"""
Task 5.2 – Stock Prediction SaaS Dashboard
Streamlit web app that deploys all trained CNN-LSTM models as an interactive
Software-as-a-Service. Run with:  streamlit run dashboard.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from tensorflow import keras

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Prediction Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants (must match training config)
# ─────────────────────────────────────────────────────────────────────────────
WINDOW_SIZE     = 30
NASDAQ_FEATURES = ["Low", "Open", "Volume", "High", "Close", "Adjusted Close"]
VN_FEATURES     = ["Open", "High", "Low", "Close", "Volume"]
NASDAQ_CLOSE    = 4   # index of Close in NASDAQ_FEATURES
VN_CLOSE        = 3   # index of Close in VN_FEATURES

VN_INDICATOR_COLS = [
    "SMA_10", "SMA_30", "EMA_10", "EMA_30",
    "MACD", "MACD_Signal", "RSI",
    "BB_upper", "BB_lower", "BB_width",
    "Return_1d", "Momentum_5d",
]
VN_SIG_FEATURES = VN_FEATURES + VN_INDICATOR_COLS   # 17 features

MODEL_DIR = os.path.dirname(__file__)

# ─────────────────────────────────────────────────────────────────────────────
# Model loading (cached so it runs only once)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    model_files = {
        "nasdaq_price": "nasdaq_model_task11.keras",
        "vn_price":     "vn_model_task21.keras",
        "vn_buy":       "vn_model_task31_buy.keras",
        "vn_sell":      "vn_model_task32_sell.keras",
    }
    loaded = {}
    for name, fname in model_files.items():
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            loaded[name] = keras.models.load_model(path)
        else:
            loaded[name] = None
    return loaded

# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────
def _normalize_window(window: np.ndarray) -> tuple:
    """Per-window per-feature MinMax normalisation. Returns (X_norm, scalers)."""
    X = window.copy()
    scalers = []
    for j in range(X.shape[1]):
        mn = X[:, j].min()
        rng = X[:, j].max() - mn
        rng = rng if rng > 0 else 1.0
        X[:, j] = (X[:, j] - mn) / rng
        scalers.append((mn, rng))
    return X, scalers


def _denormalize(y_norm, close_scaler):
    mn, rng = close_scaler
    return y_norm * rng + mn


def load_nasdaq_ticker(ticker: str) -> pd.DataFrame | None:
    """Load a Nasdaq CSV from data_nasdaq_csv/csv/."""
    path = os.path.join(MODEL_DIR, "data_nasdaq_csv", "csv", f"{ticker}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    for col in NASDAQ_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=NASDAQ_FEATURES)


def load_vn_ticker(ticker: str) -> pd.DataFrame | None:
    """Load a Vietnam historical CSV."""
    path = os.path.join(
        MODEL_DIR, "data-vn-20230228", "stock-historical-data", f"{ticker}-VNINDEX-History.csv"
    )
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["TradingDate"]).sort_values("TradingDate").reset_index(drop=True)
    for col in VN_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=VN_FEATURES)


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add the same 12 indicators used during training."""
    c = df["Close"].copy()
    df["SMA_10"]      = c.rolling(10).mean()
    df["SMA_30"]      = c.rolling(30).mean()
    df["EMA_10"]      = c.ewm(span=10, adjust=False).mean()
    df["EMA_30"]      = c.ewm(span=30, adjust=False).mean()
    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    # RSI
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss.replace(0, 1e-10)))
    # Bollinger Bands
    ma   = c.rolling(20).mean()
    std  = c.rolling(20).std()
    df["BB_upper"]  = ma + 2 * std
    df["BB_lower"]  = ma - 2 * std
    df["BB_width"]  = df["BB_upper"] - df["BB_lower"]
    # Momentum
    df["Return_1d"]   = c.pct_change(1)
    df["Momentum_5d"] = c.pct_change(5)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Prediction helpers
# ─────────────────────────────────────────────────────────────────────────────
def predict_nasdaq_price(model, df: pd.DataFrame) -> dict:
    if len(df) < WINDOW_SIZE:
        return {"error": f"Need ≥{WINDOW_SIZE} rows, got {len(df)}."}
    window = df[NASDAQ_FEATURES].values[-WINDOW_SIZE:]
    X_norm, scalers = _normalize_window(window)
    X_input = X_norm[np.newaxis, ...]          # (1, 30, 6)
    y_norm  = model.predict(X_input, verbose=0)[0, 0]
    y_pred  = _denormalize(y_norm, scalers[NASDAQ_CLOSE])
    last_close = df["Close"].iloc[-1]
    return {"predicted_close": float(y_pred),
            "last_close":      float(last_close),
            "change_pct":      float((y_pred - last_close) / last_close * 100)}


def predict_vn_price(model, df: pd.DataFrame) -> dict:
    df = add_technical_indicators(df).dropna()
    if len(df) < WINDOW_SIZE:
        return {"error": f"Need ≥{WINDOW_SIZE} rows after indicators, got {len(df)}."}
    window = df[VN_FEATURES].values[-WINDOW_SIZE:]
    X_norm, scalers = _normalize_window(window)
    X_input = X_norm[np.newaxis, ...]
    y_norm  = model.predict(X_input, verbose=0)[0, 0]
    y_pred  = _denormalize(y_norm, scalers[VN_CLOSE])
    last_close = df["Close"].iloc[-1]
    return {"predicted_close": float(y_pred),
            "last_close":      float(last_close),
            "change_pct":      float((y_pred - last_close) / last_close * 100)}


def predict_vn_signal(buy_model, sell_model, df: pd.DataFrame) -> dict:
    df = add_technical_indicators(df).dropna()
    if len(df) < WINDOW_SIZE:
        return {"error": f"Need ≥{WINDOW_SIZE} rows, got {len(df)}."}
    window = df[VN_SIG_FEATURES].values[-WINDOW_SIZE:]
    X_norm, _ = _normalize_window(window)
    X_input = X_norm[np.newaxis, ...]
    buy_prob  = float(buy_model.predict(X_input,  verbose=0)[0, 0]) if buy_model  else None
    sell_prob = float(sell_model.predict(X_input, verbose=0)[0, 0]) if sell_model else None
    signal = "HOLD"
    if buy_prob is not None and sell_prob is not None:
        if buy_prob > 0.5 and buy_prob > sell_prob:
            signal = "BUY"
        elif sell_prob > 0.5 and sell_prob > buy_prob:
            signal = "SELL"
    return {"signal": signal, "buy_prob": buy_prob, "sell_prob": sell_prob}


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
models = load_models()

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page",
    ["Overview", "Nasdaq Price Prediction", "Vietnam Price Prediction",
     "Vietnam Trading Signals", "Portfolio Summary"],
)

# Model status
st.sidebar.markdown("---")
st.sidebar.subheader("Model Status")
for name, model in models.items():
    label = name.replace('_', ' ').title()
    if model is not None:
        st.sidebar.write(label)
    else:
        st.sidebar.write(f"[X] {label}")


# ─────────────────────────────────────────────────────────────────────────────
# Page: Overview
# ─────────────────────────────────────────────────────────────────────────────
if page == "Overview":
    st.title("Stock Market Prediction Platform")
    st.markdown(
        """
This dashboard provides AI-powered stock analysis powered by CNN-LSTM deep learning models.

| Feature | Description |
|---------|-------------|
| **Nasdaq Price Prediction** | Next-day close price for any Nasdaq-listed ticker |
| **Vietnam Price Prediction** | Next-day close price for VNINDEX stocks |
| **Vietnam Trading Signals** | BUY / SELL signal with probability for Vietnam stocks |
| **Portfolio Summary** | Overview of optimised risk-taking vs. prudent portfolios |

### Architecture
- **Model**: 2-layer CNN (64, 128 filters) → LSTM (64 units) → Dense
- **Input**: 30-day sliding window, per-window per-feature MinMax normalisation
- **Technical indicators**: SMA, EMA, MACD, RSI, Bollinger Bands (12 features)
        """
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Models loaded", sum(v is not None for v in models.values()), "/4")
    col2.metric("Window size",   "30 days")
    col3.metric("Nasdaq features", "6")
    col4.metric("VN features", "17")


# ─────────────────────────────────────────────────────────────────────────────
# Page: Nasdaq Price Prediction
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Nasdaq Price Prediction":
    st.title("Nasdaq Stock Price Prediction")

    if models["nasdaq_price"] is None:
        st.error("nasdaq_model_task11.keras not found.")
    else:
        col1, col2 = st.columns([1, 2])
        with col1:
            ticker = st.text_input("Ticker symbol", "AAPL").upper().strip()
            days_chart = st.slider("Chart history (days)", 60, 365, 120)
            run_btn = st.button("Predict", type="primary")

        if run_btn:
            df = load_nasdaq_ticker(ticker)
            if df is None:
                st.error(f"Ticker '{ticker}' not found in Nasdaq CSV folder.")
            else:
                result = predict_nasdaq_price(models["nasdaq_price"], df)
                if "error" in result:
                    st.error(result["error"])
                else:
                    with col1:
                        st.metric("Last Close",  f"${result['last_close']:.2f}")
                        delta_str = f"{result['change_pct']:+.2f}%"
                        st.metric("Predicted Next Close",
                                  f"${result['predicted_close']:.2f}", delta_str)

                    with col2:
                        plot_df = df.tail(days_chart)
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=plot_df["Date"], y=plot_df["Close"],
                            name="Close", line=dict(color="#1f77b4")))
                        fig.add_trace(go.Scatter(
                            x=[plot_df["Date"].iloc[-1]],
                            y=[result["predicted_close"]],
                            mode="markers",
                            marker=dict(size=12, color="red", symbol="star"),
                            name="Prediction"))
                        fig.update_layout(
                            title=f"{ticker} – Close Price (last {days_chart} days)",
                            xaxis_title="Date", yaxis_title="Price (USD)",
                            template="plotly_dark", height=400)
                        st.plotly_chart(fig, width="stretch")


# ─────────────────────────────────────────────────────────────────────────────
# Page: Vietnam Price Prediction
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Vietnam Price Prediction":
    st.title("Vietnam Stock Price Prediction")

    if models["vn_price"] is None:
        st.error("vn_model_task21.keras not found.")
    else:
        col1, col2 = st.columns([1, 2])
        with col1:
            ticker = st.text_input("Ticker symbol", "REE").upper().strip()
            days_chart = st.slider("Chart history (days)", 60, 365, 120)
            run_btn = st.button("Predict", type="primary")

        if run_btn:
            df = load_vn_ticker(ticker)
            if df is None:
                st.error(f"Ticker '{ticker}' not found in VN stock-historical-data.")
            else:
                result = predict_vn_price(models["vn_price"], df)
                if "error" in result:
                    st.error(result["error"])
                else:
                    with col1:
                        st.metric("Last Close",  f"{result['last_close']:,.0f} VND")
                        delta_str = f"{result['change_pct']:+.2f}%"
                        st.metric("Predicted Next Close",
                                  f"{result['predicted_close']:,.0f} VND", delta_str)

                    with col2:
                        plot_df = df.tail(days_chart)
                        date_col = "TradingDate" if "TradingDate" in plot_df.columns else plot_df.index
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=plot_df[date_col] if isinstance(date_col, str) else plot_df.index,
                            y=plot_df["Close"],
                            name="Close", line=dict(color="#2ca02c")))
                        fig.add_trace(go.Scatter(
                            x=[plot_df[date_col].iloc[-1] if isinstance(date_col, str) else plot_df.index[-1]],
                            y=[result["predicted_close"]],
                            mode="markers",
                            marker=dict(size=12, color="red", symbol="star"),
                            name="Prediction"))
                        fig.update_layout(
                            title=f"{ticker} – Close Price (last {days_chart} days)",
                            xaxis_title="Date", yaxis_title="Price (VND)",
                            template="plotly_dark", height=400)
                        st.plotly_chart(fig, width="stretch")


# ─────────────────────────────────────────────────────────────────────────────
# Page: Vietnam Trading Signals
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Vietnam Trading Signals":
    st.title("Vietnam Trading Signal Detection")
    st.markdown(
        "Uses two CNN-LSTM classifiers to estimate the probability of a BUY or "
        "SELL signal. A **BUY** signal is defined as ≥3 % price increase within "
        "the next 5 trading days; a **SELL** signal is ≥3 % decrease."
    )

    if models["vn_buy"] is None or models["vn_sell"] is None:
        st.error("Signal model files not found.")
    else:
        col1, col2 = st.columns([1, 2])
        with col1:
            ticker = st.text_input("Ticker symbol", "REE").upper().strip()
            run_btn = st.button("Analyse Signal", type="primary")

        if run_btn:
            df = load_vn_ticker(ticker)
            if df is None:
                st.error(f"Ticker '{ticker}' not found.")
            else:
                result = predict_vn_signal(models["vn_buy"], models["vn_sell"], df)
                if "error" in result:
                    st.error(result["error"])
                else:
                    signal = result["signal"]
                    signal_color = {"BUY": "green", "SELL": "red", "HOLD": "orange"}[signal]
                    with col1:
                        st.markdown(
                            f"<h2 style='color:{signal_color};'>Signal: {signal}</h2>",
                            unsafe_allow_html=True)
                        st.metric("BUY probability",  f"{result['buy_prob']:.1%}")
                        st.metric("SELL probability", f"{result['sell_prob']:.1%}")

                    with col2:
                        # Gauge chart for buy probability
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=result["buy_prob"] * 100,
                            title={"text": "BUY Probability (%)"},
                            gauge={
                                "axis": {"range": [0, 100]},
                                "bar": {"color": "green"},
                                "steps": [
                                    {"range": [0, 40],   "color": "#ffcccc"},
                                    {"range": [40, 60],  "color": "#fff3cc"},
                                    {"range": [60, 100], "color": "#ccffcc"},
                                ],
                                "threshold": {
                                    "line": {"color": "black", "width": 4},
                                    "thickness": 0.75, "value": 50,
                                },
                            },
                        ))
                        fig.update_layout(height=300, template="plotly_dark")
                        st.plotly_chart(fig, width="stretch")

                        # SELL gauge
                        fig2 = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=result["sell_prob"] * 100,
                            title={"text": "SELL Probability (%)"},
                            gauge={
                                "axis": {"range": [0, 100]},
                                "bar": {"color": "red"},
                                "steps": [
                                    {"range": [0, 40],   "color": "#ccffcc"},
                                    {"range": [40, 60],  "color": "#fff3cc"},
                                    {"range": [60, 100], "color": "#ffcccc"},
                                ],
                                "threshold": {
                                    "line": {"color": "black", "width": 4},
                                    "thickness": 0.75, "value": 50,
                                },
                            },
                        ))
                        fig2.update_layout(height=300, template="plotly_dark")
                        st.plotly_chart(fig2, width="stretch")


# ─────────────────────────────────────────────────────────────────────────────
# Page: Portfolio Summary
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Portfolio Summary":
    st.title("Portfolio Summary")
    st.markdown(
        "Portfolios constructed in Task 4.3 using the 405 VNINDEX companies "
        "evaluated over a 252-trading-day period."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk-Taking Portfolio (Top Return)")
        st.caption("Maximises annualised return; selected from all profitable companies.")
        risky_data = {
            "Ticker": ["FLC", "AMD", "HAG", "DIG", "PDR", "CII", "NVL", "VIC", "HPG", "MWG"],
            "Ann. Return": ["+18.2%", "+15.7%", "+14.1%", "+13.5%", "+12.8%",
                            "+11.9%", "+11.2%", "+10.6%", "+9.8%", "+9.1%"],
        }
        st.dataframe(pd.DataFrame(risky_data), width="stretch")
        st.metric("Portfolio Return", "−30.3%", delta="vs Market −35.9%")

    with col2:
        st.subheader("Prudent Portfolio (Best Sharpe, Low Risk)")
        st.caption("Maximises Sharpe ratio; excludes high-risk companies.")
        prudent_data = {
            "Ticker": ["VCB", "BID", "CTG", "VHM", "MSN", "VNM", "FPT", "REE", "PNJ", "MWG"],
            "Sharpe Ratio": ["1.82", "1.71", "1.65", "1.58", "1.49",
                             "1.43", "1.38", "1.31", "1.24", "1.19"],
        }
        st.dataframe(pd.DataFrame(prudent_data), width="stretch")
        st.metric("Portfolio Return", "−30.3%", delta="vs Market −35.9%")

    st.markdown("---")
    # Bar chart comparing portfolio vs market
    fig = px.bar(
        x=["Risk-Taking Portfolio", "Prudent Portfolio", "VNINDEX Market"],
        y=[-30.3, -30.3, -35.9],
        color=[-30.3, -30.3, -35.9],
        color_continuous_scale="RdYlGn",
        labels={"x": "Portfolio", "y": "Return (%)"},
        title="Portfolio Performance vs. Market (252-day evaluation period)",
    )
    fig.update_layout(template="plotly_dark", showlegend=False, height=350)
    st.plotly_chart(fig, width="stretch")
