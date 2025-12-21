# stockapp.py
import streamlit as st
import numpy as np
import pandas as pd
from datetime import date, timedelta

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from vnstock import Quote
import yfinance as yf
import plotly.graph_objects as go

# ---------------------------
# App config
# ---------------------------
st.set_page_config(page_title="📈 Stock ML Predictor", layout="wide")
st.title("📊 Stock Market Prediction Dashboard")

# ---------------------------
# USER INPUT
# ---------------------------
market = st.radio("Select Market", ["VN (Việt Nam)", "INTL (International)"])
ticker = st.text_input("Enter Stock Code (e.g., FPT or AAPL)", "AAPL").upper().strip()
months_back = st.slider("Months of Data", 3, 24, 6)
indicator = st.selectbox("Technical Indicator", ["SMA", "EMA"])

if "trained" not in st.session_state:
    st.session_state.trained = False

# ---------------------------
# Helpers
# ---------------------------
def format_currency(value, symbol, name):
    try:
        if name == "VND":
            return f"{symbol}{int(round(value)):,}"
        return f"{symbol}{value:,.2f}"
    except Exception:
        return str(value)

def get_currency(market):
    if "VN" in market:
        return "₫", "VND", 1
    return "$", "USD", 1

# ---------------------------
# Load data
# ---------------------------
def load_data(ticker, market, start_date, end_date):
    if "VN" in market:
        try:
            quote = Quote(symbol=ticker, source="VCI")
            df = quote.history(
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                interval="1D"
            )
            if df is not None and not df.empty:
                df["time"] = pd.to_datetime(df["time"])
                df.set_index("time", inplace=True)
                df.rename(columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume"
                }, inplace=True)
                return df, "vnstock Quote()"
        except Exception:
            pass

    # yfinance fallback ONLY
    yf_ticker = ticker if "INTL" in market else f"{ticker}.VN"
    df = yf.download(yf_ticker, start=start_date, end=end_date)
    if df is not None and not df.empty:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df, "yfinance"

    return None, None

# ---------------------------
# RUN
# ---------------------------
if st.button("Run Prediction 🚀"):
    start_date = date.today() - timedelta(days=months_back * 30)
    end_date = date.today()

    df, source = load_data(ticker, market, start_date, end_date)
    if df is None or df.empty:
        st.error("❌ Failed to load data.")
        st.stop()

    currency_symbol, currency_name, multiplier = get_currency(market)
    st.success(f"✅ Loaded {len(df)} rows ({source})")

    # ---------------------------
    # Indicators
    # ---------------------------
    if indicator == "EMA":
        df["MA_5"] = df["Close"].ewm(span=5).mean()
        df["MA_20"] = df["Close"].ewm(span=20).mean()
    else:
        df["MA_5"] = df["Close"].rolling(5).mean()
        df["MA_20"] = df["Close"].rolling(20).mean()

    df["Diff"] = df["MA_5"] - df["MA_20"]
    df["Target"] = np.where(df["Diff"] > 0, 2, 0)

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Hist"] = df["MACD"] - df["MACD"].ewm(span=9).mean()

    mid = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()
    df["BB_Width"] = (mid + 2 * std) - (mid - 2 * std)

    df.dropna(inplace=True)

    features = ["Diff", "Volume", "RSI", "MACD", "MACD_Hist", "BB_Width"]
    X = df[features]
    y = df["Target"].astype(int).values

    classes = np.unique(y)
    st.write(f"ℹ Classes present in data: {classes}")

    if len(classes) < 2:
        st.error("❌ Not enough class diversity.")
        st.stop()

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
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "XGBoost": XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False
        )
    }

    results = []
    trained = {}

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            acc = accuracy_score(y_test, pred)
            f1 = f1_score(y_test, pred, average="macro")
            trained[name] = model
            results.append([name, acc, f1])
        except Exception:
            pass

    if not results:
        st.error("❌ No model could be trained.")
        st.stop()

    results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "F1"]).sort_values("F1", ascending=False)
    st.dataframe(results_df, use_container_width=True)

    st.session_state.update({
        "trained": True,
        "models": trained,
        "results": results_df,
        "df": df,
        "X": X,
        "scaler": scaler,
        "ticker": ticker,
        "currency_symbol": currency_symbol,
        "currency_name": currency_name
    })

# ---------------------------
# PREDICTION
# ---------------------------
if st.session_state.trained:
    st.subheader("🔮 Final Prediction")

    model_name = st.selectbox(
        "Select model",
        st.session_state.results["Model"].tolist()
    )
    model = st.session_state.models[model_name]

    last_X = st.session_state.scaler.transform(
        st.session_state.X.iloc[-1:].values
    )
    prob = model.predict_proba(last_X)[0][1]
    direction = "UP" if prob >= 0.5 else "DOWN"

    last_price = st.session_state.df["Close"].iloc[-1]
    st.metric("Predicted Direction", direction)
    st.metric("Confidence", f"{prob*100:.2f}%")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=st.session_state.df.index[-60:],
        y=st.session_state.df["Close"].iloc[-60:],
        mode="lines",
        name="Close"
    ))
    st.plotly_chart(fig, use_container_width=True)
