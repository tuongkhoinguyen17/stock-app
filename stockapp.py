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
# Helper functions
# ---------------------------
def format_currency(value, currency_symbol, currency_name):
    try:
        if currency_name == "VND":
            v = int(round(value))
            return f"{currency_symbol}{v:,}"
        else:
            return f"{currency_symbol}{value:,.2f}"
    except Exception:
        return str(value)

def get_currency_and_multiplier(market, ticker):
    if "VN" in market or ticker.upper().endswith(".VN"):
        return "₫", "VND", 1000
    else:
        return "$", "USD", 1

# ---------------------------
# Load VN stock with fallback
# ---------------------------
def load_vn_stock(ticker, start_date, end_date):
    from vnstock import Vnstock
    sources = ["VCI", "FPT", "SSI", "VCBS"]
    for src in sources:
        try:
            stock = Vnstock().stock(symbol=ticker, source=src)
            df = stock.quote.history(start=str(start_date), end=str(end_date))
            if df is not None and not df.empty:
                return df, src
        except Exception:
            continue
    try:
        # fallback to yfinance
        df = yf.download(ticker + ".VN", start=start_date, end=end_date)
        if df is not None and not df.empty:
            return df, "yfinance"
    except Exception:
        pass
    return None, None

# ---------------------------
# RUN PREDICTION BUTTON
# ---------------------------
if st.button("Run Prediction 🚀"):
    start_date = date.today() - timedelta(days=months_back * 30)
    end_date = date.today()

    st.write("📥 Loading data...")

    df = None
    if "VN" in market:
        df, src = load_vn_stock(ticker, start_date, end_date)
        if df is None:
            st.error("❌ Unable to load VN stock data from all sources.")
            st.stop()
        df.rename(columns=lambda x: x.title(), inplace=True)
        st.success(f"✅ Loaded VN stock data (Source: {src})")
    else:
        try:
            df = yf.download(ticker, start=start_date, end=end_date)
        except Exception as e:
            st.error(f"❌ Failed to fetch data: {e}")
            st.stop()

    if df is None or df.empty:
        st.error("❌ No data found. Check ticker or timeframe.")
        st.stop()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if "Close" not in df.columns:
        df.rename(columns=lambda x: x.title(), inplace=True)

    currency_symbol, currency_name, vnd_multiplier = get_currency_and_multiplier(market, ticker)
    st.success(f"✅ Loaded {len(df)} data points for {ticker}")

    # ---------------------------
    # Indicators
    # ---------------------------
    if indicator == "EMA":
        df['Ema_5'] = df['Close'].ewm(span=5).mean()
        df['Ema_20'] = df['Close'].ewm(span=20).mean()
        df['Diff'] = df['Ema_5'] - df['Ema_20']
    else:
        df['Sma_5'] = df['Close'].rolling(5).mean()
        df['Sma_20'] = df['Close'].rolling(20).mean()
        df['Diff'] = df['Sma_5'] - df['Sma_20']

    df['Target'] = np.where(df['Diff'] > 1, 2, np.where(df['Diff'] < -1, 0, 1))

    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    df['RSI'] = 100 - (100 / (1 + (gain.rolling(14).mean() / loss.rolling(14).mean())))
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']
    df['Middle'] = df['Close'].rolling(20).mean()
    df['Upper'] = df['Middle'] + 2 * df['Close'].rolling(20).std()
    df['Lower'] = df['Middle'] - 2 * df['Close'].rolling(20).std()
    df['BB_Width'] = df['Upper'] - df['Lower']

    df.dropna(inplace=True)

    features = ['Diff', 'Volume', 'RSI', 'MACD', 'MACD_Hist', 'BB_Width']
    X = df[[f for f in features if f in df.columns]]
    if X.empty:
        st.error("❌ No usable features available after indicator calculation.")
        st.stop()

    y = df['Target'].astype(int).values

    unique_classes = np.unique(y)
    st.write(f"ℹ Classes present in data: {unique_classes}")
    if len(unique_classes) < 2:
        st.warning("⚠ Not enough class diversity for predictions.")
        st.stop()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    candidate_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs'),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(max_depth=5),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "XGBoost": XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', use_label_encoder=False)
    }

    st.write("🧠 Training models...")
    trained_models, results, skipped = {}, [], []

    for name, model in candidate_models.items():
        try:
            model.fit(X_train_scaled, y_train)
            trained_models[name] = model
            try:
                pred = model.predict(X_test_scaled)
                acc = accuracy_score(y_test, pred)
                f1 = f1_score(y_test, pred, average='macro')
            except Exception:
                acc = np.nan
                f1 = np.nan
            results.append([name, acc, f1])
        except Exception as e:
            skipped.append(f"{name} (skipped: {e})")

    if skipped:
        st.warning("⚠ Some models were skipped:\n" + "\n".join(skipped))
    if not results:
        st.error("❌ No models could be trained.")
        st.stop()

    results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "F1 Score"]).sort_values(by="F1 Score", ascending=False, na_position='last')
    st.dataframe(results_df, use_container_width=True)

    st.session_state.update({
        'results_df': results_df,
        'trained_models': trained_models,
        'scaler': scaler,
        'X': X,
        'df': df,
        'ticker': ticker,
        'currency_symbol': currency_symbol,
        'currency_name': currency_name,
        'vnd_multiplier': vnd_multiplier,
        'trained': True
    })

    st.success("🎉 Training complete!")

# ---------------------------
# SELECT MODEL & PREDICT
# ---------------------------
if st.session_state.trained:
    st.subheader("🧠 Choose Model for Final Prediction")
    model_choice = st.selectbox("Select a model:", st.session_state.results_df["Model"].tolist(), index=0)
    chosen_model = st.session_state.trained_models.get(model_choice, None)
    st.info(f"🔍 Selected Model: **{model_choice}**")

    st.subheader("💰 Profit/Loss Estimation")
    num_shares = st.number_input("Enter number of shares you own:", min_value=1, value=10)

    last_price = st.session_state.df['Close'].iloc[-1] * st.session_state.vnd_multiplier

    if st.button("🔮 Predict with Selected Model"):
        if chosen_model is None:
            st.error("❌ Selected model unavailable.")
            st.stop()

        last_scaled = st.session_state.scaler.transform(st.session_state.X.iloc[-1].values.reshape(1, -1))
        probs_map = {0: 0.0, 1: 0.0, 2: 0.0}

        try:
            if hasattr(chosen_model, "predict_proba"):
                probs_raw = chosen_model.predict_proba(last_scaled)[0]
                model_classes = getattr(chosen_model, "classes_", np.arange(len(probs_raw)))
                for idx, cls in enumerate(model_classes):
                    probs_map[int(cls)] = float(probs_raw[idx])
            else:
                pred = chosen_model.predict(last_scaled)[0]
                probs_map[int(pred)] = 1.0
        except Exception as e:
            st.warning(f"⚠ Model could not generate probabilities: {e}")
            st.stop()

        probs_full = np.array([probs_map.get(i, 0.0) for i in range(3)])
        labels = {0: "DOWN", 1: "SAME", 2: "UP"}
        pred_label = int(np.argmax(probs_full))

        st.subheader(f"📈 Prediction for {st.session_state.ticker}")
        st.metric("Predicted Direction", labels[pred_label])
        st.write({labels[i]: f"{probs_full[i]*100:.2f}%" for i in range(3)})

        change_pct = {"UP": 0.02, "DOWN": -0.02, "SAME": 0.0}
        expected_price = last_price * (1 + sum(probs_full[i] * change_pct[labels[i]] for i in range(3)))
        pl_weighted = (expected_price - last_price) * num_shares

        pl_best = (last_price * (1 + max(change_pct.values())) - last_price) * num_shares
        pl_worst = (last_price * (1 + min(change_pct.values())) - last_price) * num_shares

        current_value = last_price * num_shares
        total_weighted = current_value + pl_weighted
        total_best = current_value + pl_best
        total_worst = current_value + pl_worst

        st.subheader("📊 Price Chart with Bollinger Bands")
        fig = go.Figure()
        slice_len = min(60, len(st.session_state.df))
        fig.add_trace(go.Scatter(x=st.session_state.df.index[-slice_len:], y=st.session_state.df['Close'][-slice_len:], name='Close', mode='lines'))
        if 'Upper' in st.session_state.df.columns and 'Lower' in st.session_state.df.columns:
            fig.add_trace(go.Scatter(x=st.session_state.df.index[-slice_len:], y=st.session_state.df['Upper'][-slice_len:], name='Upper Band', mode='lines', line=dict(dash='dot')))
            fig.add_trace(go.Scatter(x=st.session_state.df.index[-slice_len:], y=st.session_state.df['Lower'][-slice_len:], name='Lower Band', mode='lines', line=dict(dash='dot')))
        fig.update_layout(height=500, hovermode='x unified', legend=dict(x=0, y=1))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📋 Investment Summary Table")
        summary_df = pd.DataFrame({
            "Scenario": ["Probability-Weighted", "Best-Case", "Worst-Case"],
            f"Estimated P/L ({st.session_state.currency_name})": [pl_weighted, pl_best, pl_worst],
            f"Estimated Total Value ({st.session_state.currency_name})": [total_weighted, total_best, total_worst],
            f"Current Value ({st.session_state.currency_name})": [current_value]*3
        })

        for c in summary_df.columns[1:]:
            summary_df[c] = summary_df[c].apply(lambda v: format_currency(v, st.session_state.currency_symbol, st.session_state.currency_name))

        def color_pl(val):
            try:
                num = float(str(val).replace(st.session_state.currency_symbol, '').replace(',', ''))
            except Exception:
                num = 0.0
            return 'color: green; font-weight:bold;' if num >= 0 else 'color: red; font-weight:bold;'

        st.dataframe(summary_df.style.applymap(color_pl, subset=[c for c in summary_df.columns if c != 'Scenario']), use_container_width=True)
