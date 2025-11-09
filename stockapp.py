import streamlit as st
import numpy as np
import pandas as pd
from datetime import date, timedelta
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import yfinance as yf

st.set_page_config(page_title="📈 Stock ML Predictor", layout="wide")

st.title("📊 Stock Market Prediction Dashboard")

# --- User input ---
market = st.radio("Select Market", ["VN (Việt Nam)", "INTL (International)"])
ticker = st.text_input("Enter Stock Code (e.g., FPT or AAPL)", "AAPL").upper().strip()
months_back = st.slider("Months of Data", 3, 24, 6)
indicator = st.selectbox("Technical Indicator", ["SMA", "EMA"])

if st.button("Run Prediction 🚀"):
    start_date = date.today() - timedelta(days=months_back * 30)
    end_date = date.today()

    # --- Load data ---
    st.write("📥 Loading data...")
    if "VN" in market:
        import vnstock
        try:
            if hasattr(vnstock, "stock_historical_data"):
                df = vnstock.stock_historical_data(
                    symbol=ticker, start_date=str(start_date), end_date=str(end_date), resolution='1D'
                )
            else:
                from vnstock import Vnstock
                stock = Vnstock().stock(symbol=ticker, source='VCI')
                df = stock.quote.history(start=str(start_date), end=str(end_date))
        except Exception as e:
            st.error(f"❌ Lỗi khi tải dữ liệu VN: {e}")
            st.stop()
    else:
        df = yf.download(ticker, start=start_date, end=end_date)

    if df.empty:
        st.error("❌ No data available. Please check symbol or timeframe.")
        st.stop()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [col.title() for col in df.columns]

    st.success(f"✅ Loaded {len(df)} data points for {ticker}")

    # --- Indicators ---
    if indicator == "EMA":
        df['Ema_5'] = df['Close'].ewm(span=5, adjust=False).mean()
        df['Ema_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['Diff'] = df['Ema_5'] - df['Ema_20']
    else:
        df['Sma_5'] = df['Close'].rolling(window=5).mean()
        df['Sma_20'] = df['Close'].rolling(window=20).mean()
        df['Diff'] = df['Sma_5'] - df['Sma_20']

    df['Target'] = np.where(df['Diff'] > 1, 2, np.where(df['Diff'] < -1, 0, 1))

    # --- RSI, MACD, Bollinger ---
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

    # --- Prepare data ---
    X = df[['Diff', 'Volume', 'RSI', 'MACD', 'MACD_Hist', 'BB_Width']]
    y = df['Target']
    y_adj = y - y.min()

    X_train, X_test, y_train, y_test = train_test_split(X, y_adj, test_size=0.25, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Models ---
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(objective='multi:softprob', num_class=3, eval_metric='mlogloss')
    }

    st.write("🧠 Training models...")
    results = []
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        results.append([name, acc, f1])

    results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "F1 Score"]).sort_values(by="F1 Score", ascending=False)
    st.dataframe(results_df, use_container_width=True)

    best_model_name = results_df.iloc[0, 0]
    best_model = models[best_model_name]
    st.success(f"✅ Best Model: {best_model_name}")

    # --- Prediction ---
    latest = X.iloc[-1].values.reshape(1, -1)
    scaled_latest = scaler.transform(latest)
    probs = best_model.predict_proba(scaled_latest)[0]
    pred = np.argmax(probs)
    labels = {0: "DOWN", 1: "SAME", 2: "UP"}

    st.subheader(f"📈 Prediction for {ticker}")
    st.metric("Predicted Direction", labels[pred])
    st.write({labels[i]: f"{p*100:.2f}%" for i, p in enumerate(probs)})

    # --- Chart ---
    st.subheader("📊 Price Chart (with Bollinger Bands)")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df['Close'][-60:], label='Close')
    ax.plot(df['Upper'][-60:], linestyle='--', alpha=0.5, label='Upper BB')
    ax.plot(df['Lower'][-60:], linestyle='--', alpha=0.5, label='Lower BB')
    ax.legend()
    st.pyplot(fig)
