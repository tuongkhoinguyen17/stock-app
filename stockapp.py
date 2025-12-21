# stockapp.py
import streamlit as st
import numpy as np
import pandas as pd
from datetime import date, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import plotly.graph_objects as go

# =========================
# Data sources
# =========================
try:
    from vnstock import Quote
    vnstock_ok = True
except Exception:
    vnstock_ok = False

import yfinance as yf

# =========================
# App config
# =========================
st.set_page_config(page_title="📈 Stock ML Predictor", layout="wide")
st.title("📊 Stock Market Prediction Dashboard")

# =========================
# USER INPUT (TOP)
# =========================
market = st.radio("Select Market", ["VN (Việt Nam)", "INTL (International)"])
ticker = st.text_input("Stock Code", "FPT").upper().strip()
num_shares = st.number_input("Number of shares to buy", min_value=1, value=100)
months_back = st.slider("Months of Data", 3, 24, 6)
indicator = st.selectbox("Technical Indicator", ["SMA", "EMA"])

# =========================
# Helpers
# =========================
def load_vn_stock(ticker, start, end):
    if not vnstock_ok:
        return None, None
    try:
        q = Quote(symbol=ticker, source="VCI")
        df = q.history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="1D"
        )
        if df is not None and not df.empty:
            return df, "vnstock"
    except Exception:
        pass
    return None, None

def load_yf(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end)
        if df is not None and not df.empty:
            return df, "yfinance"
    except Exception:
        pass
    return None, None

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# =========================
# RUN
# =========================
if st.button("Run Prediction 🚀"):
    start = date.today() - timedelta(days=months_back * 30)
    end = date.today()

    df = None
    source = None

    if "VN" in market:
        df, source = load_vn_stock(ticker, start, end)
        if df is None:
            df, source = load_yf(ticker + ".VN", start, end)
    else:
        df, source = load_yf(ticker, start, end)

    if df is None or df.empty:
        st.error("❌ Failed to load data.")
        st.stop()

    # normalize columns
    df = df.rename(columns=lambda x: x.lower())
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")
    else:
        df.index = pd.to_datetime(df.index)

    df = df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume"
    })

    st.success(f"✅ Loaded {len(df)} rows via {source}")

    # =========================
    # Indicators
    # =========================
    if indicator == "EMA":
        df["fast"] = df["Close"].ewm(span=5).mean()
        df["slow"] = df["Close"].ewm(span=20).mean()
    else:
        df["fast"] = df["Close"].rolling(5).mean()
        df["slow"] = df["Close"].rolling(20).mean()

    df["Diff"] = df["fast"] - df["slow"]
    df["RSI"] = compute_rsi(df["Close"])

    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Hist"] = df["MACD"] - df["MACD"].ewm(span=9).mean()

    mid = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()
    df["BB_Width"] = (mid + 2 * std) - (mid - 2 * std)

    df["Target"] = np.where(df["Diff"] > 0, 2,
                    np.where(df["Diff"] < 0, 0, 1))

    df.dropna(inplace=True)

    X = df[["Diff", "Volume", "RSI", "MACD", "MACD_Hist", "BB_Width"]]
    y = df["Target"].astype(int).values

    classes = np.unique(y)
    st.write(f"ℹ Classes present in data: {classes}")
    if len(classes) < 2:
        st.error("❌ Not enough class diversity.")
        st.stop()

    # =========================
    # Train
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "KNN": KNeighborsClassifier(5),
        "Decision Tree": DecisionTreeClassifier(max_depth=5),
        "Random Forest": RandomForestClassifier(100),
        "XGBoost": XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            use_label_encoder=False
        )
    }

    results = []
    trained = {}

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            results.append([
                name,
                accuracy_score(y_test, pred),
                f1_score(y_test, pred, average="macro")
            ])
            trained[name] = model
        except Exception:
            pass

    res_df = pd.DataFrame(results, columns=["Model", "Accuracy", "F1"])
    st.dataframe(res_df, use_container_width=True)

    # =========================
    # Select model
    # =========================
    model_name = st.selectbox("Select model", res_df["Model"])
    model = trained[model_name]

    last_X = scaler.transform(X.iloc[-1].values.reshape(1, -1))
    probs = model.predict_proba(last_X)[0]

    prob_map = {cls: 0.0 for cls in [0, 1, 2]}
    for i, cls in enumerate(model.classes_):
        prob_map[int(cls)] = probs[i]

    labels = {0: "DOWN", 1: "SAME", 2: "UP"}
    pred_class = max(prob_map, key=prob_map.get)

    st.subheader("📈 Prediction")
    st.metric("Direction", labels[pred_class])
    st.write({labels[k]: f"{v*100:.2f}%" for k, v in prob_map.items()})

    # =========================
    # PROFIT / LOSS
    # =========================
    last_price = df["Close"].iloc[-1]
    change = {0: -0.02, 1: 0.0, 2: 0.02}

    expected_price = last_price * (
        1 + sum(prob_map[c] * change[c] for c in prob_map)
    )

    pl_weighted = (expected_price - last_price) * num_shares
    pl_best = last_price * 0.02 * num_shares
    pl_worst = -last_price * 0.02 * num_shares

    st.subheader("💰 Profit / Loss Estimation")
    st.write(pd.DataFrame({
        "Scenario": ["Probability-weighted", "Best case", "Worst case"],
        "P/L": [pl_weighted, pl_best, pl_worst]
    }))

    # =========================
    # Chart
    # =========================
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close"))
    st.plotly_chart(fig, use_container_width=True)
