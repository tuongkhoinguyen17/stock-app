# stock_prediction_app.py
# VNStock primary → yfinance fallback
# Market selector restored
# P/L table fixed
# Spinner + status text added
# User input (shares) placed at top

import streamlit as st
import numpy as np
import pandas as pd
from datetime import date, timedelta

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from xgboost import XGBClassifier
import yfinance as yf
import plotly.graph_objects as go

# =============================
# Optional vnstock
# =============================
try:
    from vnstock import Vnstock
    vnstock_ok = True
except Exception:
    vnstock_ok = False

# =============================
# App config
# =============================
st.set_page_config(page_title="📈 Stock ML Predictor", layout="wide")
st.title("📊 Stock Market Prediction Dashboard")

# =============================
# USER INPUT (TOP)
# =============================
col1, col2, col3 = st.columns(3)

with col1:
    market = st.selectbox("Market", ["VN (Việt Nam)", "INTL (International)"])

with col2:
    ticker = st.text_input("Stock Code", "AAPL").upper().strip()

with col3:
    shares = st.number_input("Number of shares", min_value=1, value=100)

months_back = st.slider("Months of Data", 3, 24, 6)
indicator = st.selectbox("Indicator Type", ["SMA", "EMA"])

run_btn = st.button("🚀 Run Prediction")

if "trained" not in st.session_state:
    st.session_state.trained = False

# =============================
# Helpers
# =============================
def load_vnstock(ticker, start, end):
    if not vnstock_ok:
        return None, None

    sources = ["VCI", "FPT", "SSI", "VCBS"]
    for src in sources:
        try:
            stock = Vnstock().stock(symbol=ticker, source=src)
            df = stock.quote.history(
                start=str(start),
                end=str(end),
            )
            if df is not None and not df.empty:
                return df, f"vnstock-{src}"
        except Exception:
            continue
    return None, None


def load_yf(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        return df
    except Exception:
        return None


# =============================
# RUN
# =============================
if run_btn:
    with st.spinner("🔮 Predicting... Please wait"):
        start_date = date.today() - timedelta(days=months_back * 30)
        end_date = date.today()

        df, src = None, None

        # ---- VN primary → yfinance fallback
        if "VN" in market:
            df, src = load_vnstock(ticker, start_date, end_date)
            if df is None:
                df = load_yf(f"{ticker}.VN", start_date, end_date)
                src = "yfinance"
        else:
            df = load_yf(ticker, start_date, end_date)
            src = "yfinance"

        if df is None or df.empty:
            st.error("❌ Unable to load data")
            st.stop()

        df = df.copy()
        df.rename(columns=lambda x: x.title(), inplace=True)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        st.success(f"✅ Data loaded from {src}")

        # =============================
        # Indicators
        # =============================
        if indicator == "EMA":
            df["Fast"] = df["Close"].ewm(span=5).mean()
            df["Slow"] = df["Close"].ewm(span=20).mean()
        else:
            df["Fast"] = df["Close"].rolling(5).mean()
            df["Slow"] = df["Close"].rolling(20).mean()

        df["Diff"] = df["Fast"] - df["Slow"]

        # Target: binary (FIXES CLASS ERROR)
        df["Target"] = np.where(df["Diff"] > 0, 1, 0)

        delta = df["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        df["RSI"] = 100 - (100 / (1 + gain.rolling(14).mean() / loss.rolling(14).mean()))

        ema12 = df["Close"].ewm(span=12).mean()
        ema26 = df["Close"].ewm(span=26).mean()
        df["MACD"] = ema12 - ema26

        mid = df["Close"].rolling(20).mean()
        std = df["Close"].rolling(20).std()
        df["BB_Width"] = (mid + 2 * std) - (mid - 2 * std)

        df.dropna(inplace=True)

        features = ["Diff", "Volume", "RSI", "MACD", "BB_Width"]
        X = df[features]
        y = df["Target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "KNN": KNeighborsClassifier(5),
            "Decision Tree": DecisionTreeClassifier(max_depth=5),
            "Random Forest": RandomForestClassifier(100),
            "XGBoost": XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                use_label_encoder=False,
            ),
        }

        results = []
        trained = {}

        for name, m in models.items():
            try:
                m.fit(X_train_s, y_train)
                pred = m.predict(X_test_s)
                results.append([
                    name,
                    accuracy_score(y_test, pred),
                    f1_score(y_test, pred),
                ])
                trained[name] = m
            except Exception:
                continue

        results_df = pd.DataFrame(
            results, columns=["Model", "Accuracy", "F1 Score"]
        ).sort_values("F1 Score", ascending=False)

        st.subheader("🧠 Model Performance")
        st.dataframe(results_df, use_container_width=True)

        st.session_state.update({
            "trained": True,
            "df": df,
            "X": X,
            "scaler": scaler,
            "models": trained,
            "results": results_df,
            "ticker": ticker,
            "shares": shares,
        })

    st.success("✔ Prediction finished")

# =============================
# PREDICTION + P/L
# =============================
if st.session_state.trained:
    st.subheader("📈 Final Prediction")

    model_name = st.selectbox(
        "Choose model",
        st.session_state.results["Model"].tolist(),
    )
    model = st.session_state.models[model_name]

    last_row = st.session_state.X.iloc[-1].values.reshape(1, -1)
    last_scaled = st.session_state.scaler.transform(last_row)

    prob_up = (
        model.predict_proba(last_scaled)[0][1]
        if hasattr(model, "predict_proba")
        else float(model.predict(last_scaled)[0])
    )

    last_price = float(st.session_state.df["Close"].iloc[-1])
    shares = st.session_state.shares

    expected_change = (prob_up * 0.02) - ((1 - prob_up) * 0.02)
    expected_price = last_price * (1 + expected_change)

    pl_expected = (expected_price - last_price) * shares
    pl_best = (last_price * 1.02 - last_price) * shares
    pl_worst = (last_price * 0.98 - last_price) * shares

    st.metric("Probability of UP", f"{prob_up*100:.2f}%")

    pl_df = pd.DataFrame({
        "Scenario": ["Probability-weighted", "Best case", "Worst case"],
        "P / L": [
            round(pl_expected, 2),
            round(pl_best, 2),
            round(pl_worst, 2),
        ],
    })

    st.subheader("💰 Profit / Loss")
    st.dataframe(pl_df, use_container_width=True)

    st.subheader("📊 Price Chart")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=st.session_state.df.index,
        y=st.session_state.df["Close"],
        name="Close",
    ))
    st.plotly_chart(fig, use_container_width=True)
