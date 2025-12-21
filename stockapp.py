# stockapp.py — VN / International market restored, vnstock → yfinance fallback,
# P/L fixed, prediction progress text added

import streamlit as st
import pandas as pd
import numpy as np
import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ===============================
# Optional vnstock (VN market)
# ===============================
try:
    from vnstock import Quote
    vnstock_ok = True
except Exception:
    vnstock_ok = False

# ===============================
# yfinance (fallback / intl)
# ===============================
import yfinance as yf


# ===============================
# Streamlit config
# ===============================
st.set_page_config(page_title="Unsupervised Stock Explorer", layout="wide")
st.title("📊 Unsupervised Stock Behavior Explorer")


# ===============================
# Indicators
# ===============================
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


# ===============================
# Normalize dataframe
# ===============================
def unify_df(df):
    if df is None or df.empty:
        return None

    df = df.copy()

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")
    else:
        df.index = pd.to_datetime(df.index)

    rename_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    df = df.rename(columns=rename_map)

    required = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in df.columns for c in required):
        return None

    df["Returns"] = df["Close"].pct_change()
    df["Volatility"] = df["Returns"].rolling(7).std()
    df["SMA10"] = df["Close"].rolling(10).mean()
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["RSI"] = compute_rsi(df["Close"])

    df = df.dropna()
    return df if not df.empty else None


# ===============================
# Data loaders
# ===============================
def load_vnstock(ticker, start, end):
    if not vnstock_ok:
        return None

    try:
        quote = Quote(symbol=ticker, source="VCI")
        df = quote.history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="1D",
        )
        return unify_df(df)
    except Exception:
        return None


def load_yfinance(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        return unify_df(df)
    except Exception:
        return None


# ===============================
# Feature matrix
# ===============================
def make_feature_matrix(data):
    rows = []
    for t, df in data.items():
        rows.append({
            "Ticker": t,
            "AvgReturn": df["Returns"].mean(),
            "Volatility": df["Volatility"].mean(),
            "RSI": df["RSI"].mean(),
            "SMA10_Slope": df["SMA10"].iloc[-1] - df["SMA10"].iloc[0],
            "SMA20_Slope": df["SMA20"].iloc[-1] - df["SMA20"].iloc[0],
        })
    return pd.DataFrame(rows)


# ===============================
# Sidebar / Inputs
# ===============================
st.subheader("🔧 Inputs")

market = st.selectbox("Market", ["VN", "International"])

raw_tickers = st.text_input("Tickers (comma separated)", "VNM, HPG, FPT")
tickers = [t.strip().upper() for t in raw_tickers.split(",") if t.strip()]

shares = st.number_input("Shares per stock", min_value=1, value=100)

months = st.slider("Months of history", 3, 36, 12)

run_btn = st.button("▶ Run prediction")


# ===============================
# Main logic
# ===============================
if run_btn:
    with st.spinner("🔮 Predicting stock behavior..."):
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=months * 30)

        stock_data = {}

        for t in tickers:
            df = None

            if market == "VN":
                df = load_vnstock(t, start, end)
                if df is None:
                    df = load_yfinance(f"{t}.VN", start, end)

            else:
                df = load_yfinance(t, start, end)

            if df is not None:
                stock_data[t] = df

        if len(stock_data) < 2:
            st.error("Need at least 2 valid stocks")
            st.stop()

        # ===============================
        # Feature engineering
        # ===============================
        features = make_feature_matrix(stock_data)

        X = features.drop(columns=["Ticker"])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # ===============================
        # Clustering
        # ===============================
        k = min(4, len(features))
        kmeans = KMeans(n_clusters=k, n_init="auto")
        features["Cluster"] = kmeans.fit_predict(X_scaled)

        # ===============================
        # PCA
        # ===============================
        pca = PCA(n_components=2)
        pcs = pca.fit_transform(X_scaled)
        features["PC1"] = pcs[:, 0]
        features["PC2"] = pcs[:, 1]

        st.subheader("🧮 Feature Matrix")
        st.dataframe(features)

        fig = px.scatter(
            features,
            x="PC1",
            y="PC2",
            color=features["Cluster"].astype(str),
            text="Ticker",
            size="Volatility",
            title="PCA Clustering",
        )
        st.plotly_chart(fig, use_container_width=True)

        # ===============================
        # Anomaly detection
        # ===============================
        iso = IsolationForest(contamination=0.1)
        features["Anomaly"] = iso.fit_predict(X_scaled)

        st.subheader("⚠️ Anomalies")
        st.dataframe(features[features["Anomaly"] == -1])

        # ===============================
        # Profit / Loss estimation
        # ===============================
        st.subheader("💰 Profit / Loss Estimation")

        pl_rows = []

        for t, df in stock_data.items():
            last_price = df["Close"].iloc[-1]
            daily_std = df["Returns"].std()

            invested = last_price * shares

            best = invested * (1 + 2 * daily_std)
            worst = invested * (1 - 2 * daily_std)
            expected = invested * (1 + df["Returns"].mean())

            pl_rows.append({
                "Ticker": t,
                "Invested": round(invested, 2),
                "Best case": round(best - invested, 2),
                "Worst case": round(worst - invested, 2),
                "Expected (mean)": round(expected - invested, 2),
            })

        pl_df = pd.DataFrame(pl_rows)
        st.dataframe(pl_df)

        # ===============================
        # Price viewer
        # ===============================
        st.subheader("📈 Price Viewer")

        choice = st.selectbox("Select ticker", list(stock_data.keys()))
        dfv = stock_data[choice]

        fig2 = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.7, 0.3],
            vertical_spacing=0.03,
            specs=[[{"type": "candlestick"}], [{"type": "bar"}]],
        )

        fig2.add_trace(
            go.Candlestick(
                x=dfv.index,
                open=dfv["Open"],
                high=dfv["High"],
                low=dfv["Low"],
                close=dfv["Close"],
                name="Price",
            ),
            row=1,
            col=1,
        )

        fig2.add_trace(go.Scatter(x=dfv.index, y=dfv["SMA10"], name="SMA10"), row=1, col=1)
        fig2.add_trace(go.Scatter(x=dfv.index, y=dfv["SMA20"], name="SMA20"), row=1, col=1)

        fig2.add_trace(
            go.Bar(x=dfv.index, y=dfv["Volume"], name="Volume"),
            row=2,
            col=1,
        )

        st.plotly_chart(fig2, use_container_width=True)

    st.success("✔ Prediction completed")
