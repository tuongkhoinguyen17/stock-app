# stock_ml_app.py
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
import yfinance as yf
import plotly.graph_objects as go

# ======================================================
# App config
# ======================================================
st.set_page_config(page_title="📈 Stock ML Predictor", layout="wide")
st.title("📊 Stock Market Prediction Dashboard")

# ======================================================
# USER INPUT (TOP)
# ======================================================
col1, col2, col3 = st.columns(3)

with col1:
    ticker = st.text_input("Stock Code", "FPT.VN").upper().strip()

with col2:
    months_back = st.slider("Months of Data", 3, 24, 6)

with col3:
    num_shares = st.number_input("Number of shares", min_value=1, value=10)

indicator = st.selectbox("Technical Indicator", ["SMA", "EMA"])

if "trained" not in st.session_state:
    st.session_state.trained = False

# ======================================================
# Helpers
# ======================================================
def format_currency(v, symbol):
    return f"{symbol}{v:,.0f}"

def load_data_yf(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.rename(columns=lambda x: x.title(), inplace=True)
    return df

# ======================================================
# RUN PREDICTION
# ======================================================
if st.button("🚀 Run Prediction"):
    with st.spinner("🔮 Predicting stock movement, please wait..."):
        start_date = date.today() - timedelta(days=months_back * 30)
        end_date = date.today()

        df = load_data_yf(ticker, start_date, end_date)
        if df is None:
            st.error("❌ Failed to load data from yfinance.")
            st.stop()

        st.success(f"✅ Loaded {len(df)} data points")

        # ---------------- Indicators ----------------
        if indicator == "EMA":
            df["Fast"] = df["Close"].ewm(span=5).mean()
            df["Slow"] = df["Close"].ewm(span=20).mean()
        else:
            df["Fast"] = df["Close"].rolling(5).mean()
            df["Slow"] = df["Close"].rolling(20).mean()

        df["Diff"] = df["Fast"] - df["Slow"]
        df["Target"] = np.where(df["Diff"] > 0, 2, np.where(df["Diff"] < 0, 0, 1))

        delta = df["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        df["RSI"] = 100 - (100 / (1 + gain.rolling(14).mean() / loss.rolling(14).mean()))

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

        if len(np.unique(y)) < 2:
            st.error("❌ Not enough class diversity.")
            st.stop()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        models = {
            "Logistic Regression": LogisticRegression(max_iter=500),
            "KNN": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(max_depth=5),
            "Random Forest": RandomForestClassifier(n_estimators=100),
            "XGBoost": XGBClassifier(
                objective="multi:softprob",
                eval_metric="mlogloss",
                use_label_encoder=False
            ),
        }

        results = []
        trained = {}

        for name, model in models.items():
            try:
                model.fit(X_train_s, y_train)
                pred = model.predict(X_test_s)
                results.append([
                    name,
                    accuracy_score(y_test, pred),
                    f1_score(y_test, pred, average="macro")
                ])
                trained[name] = model
            except:
                pass

        results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "F1"])
        results_df.sort_values("F1", ascending=False, inplace=True)

        st.session_state.update({
            "df": df,
            "X": X,
            "scaler": scaler,
            "models": trained,
            "results": results_df,
            "trained": True
        })

    st.success("🎉 Training complete!")

# ======================================================
# PREDICTION & P/L
# ======================================================
if st.session_state.trained:
    st.subheader("🧠 Model Performance")
    st.dataframe(st.session_state.results, use_container_width=True)

    model_name = st.selectbox("Select model", st.session_state.results["Model"])
    model = st.session_state.models[model_name]

    last_scaled = st.session_state.scaler.transform(
        st.session_state.X.iloc[-1:].values
    )

    probs = model.predict_proba(last_scaled)[0]
    labels = ["DOWN", "SAME", "UP"]

    pred_idx = int(np.argmax(probs))
    st.metric("Predicted Direction", labels[pred_idx])

    last_price = float(st.session_state.df["Close"].iloc[-1])
    change = {"UP": 0.02, "DOWN": -0.02, "SAME": 0.0}

    expected_price = last_price * (
        1 + sum(probs[i] * change[labels[i]] for i in range(3))
    )

    pl_weighted = float((expected_price - last_price) * num_shares)
    pl_best = float(last_price * 0.02 * num_shares)
    pl_worst = float(-last_price * 0.02 * num_shares)

    summary = pd.DataFrame({
        "Scenario": ["Probability-weighted", "Best case", "Worst case"],
        "Estimated P/L": [pl_weighted, pl_best, pl_worst]
    })

    summary["Estimated P/L"] = summary["Estimated P/L"].apply(
        lambda x: format_currency(x, "$")
    )

    st.subheader("📊 Profit / Loss Estimation")
    st.dataframe(summary, use_container_width=True)

    # ---------------- Chart ----------------
    st.subheader("📈 Price Chart")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=st.session_state.df.index[-60:],
        y=st.session_state.df["Close"].iloc[-60:],
        name="Close"
    ))
    st.plotly_chart(fig, use_container_width=True)
