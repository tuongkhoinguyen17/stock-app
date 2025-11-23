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
# Small helper: currency formatting
# ---------------------------
def format_currency(value, currency_symbol, currency_name):
    """Format a numeric value for display based on currency.
    VND -> no decimals, use ₫
    USD (and others) -> two decimals, use $
    """
    if currency_name == "VND":
        try:
            v = int(round(value))
        except Exception:
            v = int(value)
        return f"{currency_symbol}{v:,}"
    else:
        return f"{currency_symbol}{value:,.2f}"

# ---------------------------
# RUN PREDICTION BUTTON
# ---------------------------
if st.button("Run Prediction 🚀"):
    start_date = date.today() - timedelta(days=months_back * 30)
    end_date = date.today()

    st.write("📥 Loading data...")

    # Load stock data
    if "VN" in market:
        try:
            from vnstock import Vnstock
            stock = Vnstock().stock(symbol=ticker, source="VCI")
            df = stock.quote.history(start=str(start_date), end=str(end_date))
            df.rename(columns=lambda x: x.title(), inplace=True)
        except Exception as e:
            st.error(f"❌ Error loading VN stock data: {e}")
            st.stop()
    else:
        df = yf.download(ticker, start=start_date, end=end_date)

    if df is None or df.empty:
        st.error("❌ No data found. Check ticker or timeframe.")
        st.stop()

    # handle yfinance multiindex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if "Close" not in df.columns:
        df.rename(columns=lambda x: x.title(), inplace=True)
        
    # -----------------------------
    # Detect currency automatically
    # -----------------------------
    # prefer explicit VN ticker marker (e.g., VNM.VN). You can extend detection if needed.
    if ticker.upper().endswith(".VN") or "VN" in market:
        currency_symbol = "₫"
        currency_name = "VND"
    else:
        currency_symbol = "$"
        currency_name = "USD"

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

    # dynamic target threshold: keep your original thresholds but you can adjust later
    df['Target'] = np.where(df['Diff'] > 1, 2, np.where(df['Diff'] < -1, 0, 1))

    # Additional features
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

    # Features & target
    features = ['Diff', 'Volume', 'RSI', 'MACD', 'MACD_Hist', 'BB_Width']
    existing_features = [f for f in features if f in df.columns]
    if not existing_features:
        st.error("❌ No usable features available after indicator calculation.")
        st.stop()

    X = df[existing_features]
    y = df['Target'].astype(int).values

    # ---------------------------
    # Check class diversity
    # ---------------------------
    unique_classes = np.unique(y)
    st.write(f"ℹ Classes present in data: {unique_classes}")
    if len(unique_classes) < 2:
        st.warning(
            "⚠ This stock hasn't moved enough in the selected timeframe.\n"
            "Prediction models require at least 2 classes (UP/DOWN/SAME).\n"
            "Try a longer timeframe or a more volatile stock."
        )
        st.stop()

    # --------------------------- 
    # Split & scale
    # ---------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---------------------------
    # Define candidate models
    # ---------------------------
    candidate_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs'),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(max_depth=5),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "XGBoost": XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', use_label_encoder=False)
    }

    st.write("🧠 Training models (skipping ones that error)...")
    trained_models = {}   # only models that trained successfully
    results = []
    skipped = []

    for name, model in candidate_models.items():
        try:
            model.fit(X_train_scaled, y_train)
            trained_models[name] = model

            # evaluate
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
        st.error("❌ No models could be trained. Try another stock or timeframe.")
        st.stop()

    # Show results
    results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "F1 Score"])
    results_df = results_df.sort_values(by="F1 Score", ascending=False, na_position='last')
    st.dataframe(results_df, use_container_width=True)

    # Save session
    st.session_state.results_df = results_df
    st.session_state.trained_models = trained_models
    st.session_state.scaler = scaler
    st.session_state.X = X
    st.session_state.df = df
    st.session_state.ticker = ticker
    st.session_state.currency_symbol = currency_symbol
    st.session_state.currency_name = currency_name
    st.session_state.trained = True

    st.success("🎉 Training complete! Select a model below to continue.")

# ---------------------------
# SELECT MODEL & PREDICT
# ---------------------------
if st.session_state.trained:
    st.subheader("🧠 Choose Model for Final Prediction")
    model_choice = st.selectbox(
        "Select a model:",
        st.session_state.results_df["Model"].tolist(),
        index=0
    )

    trained_models = st.session_state.trained_models
    if model_choice not in trained_models:
        st.warning("⚠ The selected model was not trained successfully. Choose another model.")
    chosen_model = trained_models.get(model_choice, None)

    st.info(f"🔍 Selected Model: **{model_choice}**")

    # User input for shares
    st.subheader("💰 Profit/Loss Estimation")
    num_shares = st.number_input("Enter number of shares you own:", min_value=1, value=10)

    last_price = st.session_state.df['Close'].iloc[-1]

    currency_symbol = st.session_state.currency_symbol
    currency_name = st.session_state.currency_name

    # ------------------------------------------------------
    # Apply VND multiplier (VN market uses prices * 1000)
    # ------------------------------------------------------
    if "VN" in market or ticker.endswith(".VN"):
        vnd_multiplier = 1000
    else:
        vnd_multiplier = 1

    last_price *= vnd_multiplier

    if st.button("🔮 Predict with Selected Model"):
        if chosen_model is None:
            st.error("❌ Selected model unavailable. Pick a trained model.")
            st.stop()

        X = st.session_state.X
        scaler = st.session_state.scaler
        df = st.session_state.df
        ticker = st.session_state.ticker

        # last row features and scaling
        last = X.iloc[-1].values.reshape(1, -1)
        last_scaled = scaler.transform(last)

        # Try to get predicted probabilities in a safe way
        probs_map = {0: 0.0, 1: 0.0, 2: 0.0}  # default probs for DOWN(0),SAME(1),UP(2)
        model_classes = None

        try:
            if hasattr(chosen_model, "predict_proba"):
                probs_raw = chosen_model.predict_proba(last_scaled)[0]  # shape (n_classes_model,)
                model_classes = getattr(chosen_model, "classes_", None)
                if model_classes is None:
                    model_classes = np.arange(len(probs_raw))
                for idx, cls in enumerate(model_classes):
                    probs_map[int(cls)] = float(probs_raw[idx])
            else:
                pred = chosen_model.predict(last_scaled)[0]
                probs_map[int(pred)] = 1.0
        except Exception as e:
            st.warning(f"⚠ Model could not generate probabilities: {e}")
            st.stop()

        # Build probs array in order [0,1,2] (DOWN,SAME,UP)
        probs_full = np.array([probs_map.get(0, 0.0), probs_map.get(1, 0.0), probs_map.get(2, 0.0)])
        pred_label = int(np.argmax(probs_full))
        labels = {0: "DOWN", 1: "SAME", 2: "UP"}

        # Display prediction and probabilities
        st.subheader(f"📈 Prediction for {ticker}")
        st.metric("Predicted Direction", labels[pred_label])
        st.write({labels[i]: f"{probs_full[i]*100:.2f}%" for i in range(3)})

        # P/L calculations using only available probability mass (missing classes treated as 0)
        change_pct = {"UP": 0.02, "DOWN": -0.02, "SAME": 0.0}
        expected_rel_change = sum(probs_full[i] * change_pct[labels[i]] for i in range(3))
        expected_price = last_price * (1 + expected_rel_change)
        pl_weighted = (expected_price - last_price) * num_shares

        # -----------------------------
        # Multiply scenarios for VND
        # -----------------------------
        if vnd_multiplier == 1000:
            # already applied to last_price, so only multiply deltas that came from original 1x inputs
            pass  # nothing else needed

        # Best/worst (use canonical UP/SAME/DOWN)
        best_change = max(change_pct.values())
        worst_change = min(change_pct.values())
        pl_best = (last_price * (1 + best_change) - last_price) * num_shares
        pl_worst = (last_price * (1 + worst_change) - last_price) * num_shares

        current_value = last_price * num_shares
        total_weighted = current_value + pl_weighted
        total_best = current_value + pl_best
        total_worst = current_value + pl_worst

        # Price Chart
        st.subheader("📊 Price Chart with Bollinger Bands")
        fig = go.Figure()
        slice_len = min(60, len(df))
        fig.add_trace(go.Scatter(x=df.index[-slice_len:], y=df['Close'][-slice_len:], name='Close', mode='lines'))
        if 'Upper' in df.columns and 'Lower' in df.columns:
            fig.add_trace(go.Scatter(x=df.index[-slice_len:], y=df['Upper'][-slice_len:], name='Upper Band', mode='lines', line=dict(dash='dot')))
            fig.add_trace(go.Scatter(x=df.index[-slice_len:], y=df['Lower'][-slice_len:], name='Lower Band', mode='lines', line=dict(dash='dot')))
        fig.update_layout(height=500, hovermode='x unified', legend=dict(x=0, y=1))
        st.plotly_chart(fig, use_container_width=True)

        # Summary table (numbers formatted according to currency)
        st.subheader("📋 Investment Summary Table")
        currency_symbol = "₫" if vnd_multiplier == 1000 else "$"
        currency_name = "VND" if vnd_multiplier == 1000 else "USD"

        summary_df = pd.DataFrame({
            "Scenario": ["Probability-Weighted", "Best-Case", "Worst-Case"],
            f"Estimated P/L ({currency_name})": [pl_weighted, pl_best, pl_worst],
            f"Estimated Total Value ({currency_name})": [total_weighted, total_best, total_worst],
            f"Current Value ({currency_name})": [current_value]*3,
            "Probability (%)": [
                "—",
                f"{probs_full[np.argmax([change_pct[l] for l in labels.values()])]*100:.2f}%",
                f"{probs_full[np.argmin([change_pct[l] for l in labels.values()])]*100:.2f}%"
            ]
        })


        # Format numeric columns with currency formatting
        numeric_cols = [c for c in summary_df.columns if c.startswith("Estimated") or c.startswith("Current")]
        for c in numeric_cols:
            summary_df[c] = summary_df[c].apply(lambda v: format_currency(v, currency_symbol, currency_name))

        def color_pl(val):
            # val here is formatted string like "$1,234.56" or "₫1,234"
            # We parse numeric for coloring
            try:
                # remove non-numeric characters except minus and dot and comma
                s = str(val).replace(currency_symbol, "").replace(",", "").strip()
                num = float(s)
            except Exception:
                num = 0.0
            color = "green" if num >= 0 else "red"
            return f'color: {color}; font-weight:bold;'

        st.dataframe(
            summary_df.style.applymap(color_pl, subset=[c for c in summary_df.columns if c.startswith("Estimated") or c.startswith("Current")]),
            use_container_width=True
        )
