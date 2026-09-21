import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.data_loader import get_stock_data
from src.features import add_features, TECHNICAL_FEATURES
from src.model import walk_forward_evaluate, compare_models
from src.backtest import backtest, benchmark_equity, calculate_metrics
from src.signals import generate_signal

st.set_page_config(page_title="MarketSignal", page_icon="📈", layout="wide")
st.title("MarketSignal")
st.caption("Walk-forward machine-learning research for 5-day stock direction")

with st.sidebar:
    ticker = st.text_input("Ticker", "AAPL").upper().strip()
    confidence_threshold = st.slider("Signal confidence threshold", 0.50, 0.80, 0.60, 0.01)
    retrain_days = st.selectbox("Retrain frequency", [21, 42, 63], index=0)
    transaction_cost_bps = st.number_input("Transaction cost (bps)", 0.0, 100.0, 10.0, 1.0)
    slippage_bps = st.number_input("Slippage (bps)", 0.0, 100.0, 5.0, 1.0)

@st.cache_data(ttl=3600)
def load_features(symbol):
    return add_features(get_stock_data(symbol))

try:
    df = load_features(ticker)
except Exception as exc:
    st.error(f"Could not load {ticker}: {exc}")
    st.stop()

st.write(f"Using {len(df):,} labeled trading days from {df.index.min().date()} to {df.index.max().date()}.")

with st.spinner("Running walk-forward evaluation..."):
    model, predictions, metrics = walk_forward_evaluate(
        df, TECHNICAL_FEATURES, retrain_every=retrain_days, model_name="XGBoost"
    )

signal_row = predictions.iloc[-1]
signal = generate_signal(int(signal_row["Prediction"]), float(signal_row["Probability"]), confidence_threshold)
up_probability = float(signal_row["Probability"])
down_probability = 1 - up_probability

c1, c2, c3, c4 = st.columns(4)
c1.metric("5-day UP probability", f"{up_probability:.1%}")
c2.metric("5-day DOWN probability", f"{down_probability:.1%}")
c3.metric("Signal", signal)
c4.metric("Last close", f"${df["Close"].iloc[-1]:,.2f}")

st.subheader("Out-of-sample classification performance")
metric_cols = st.columns(6)
for col, label, key in zip(metric_cols, ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Log loss"], ["accuracy", "precision", "recall", "f1", "roc_auc", "log_loss"]):
    value = metrics[key]
    col.metric(label, f"{value:.3f}" if pd.notna(value) else "N/A")

st.caption(f"Expanding-window walk-forward test: first 70% for initial training; model retrained every {retrain_days} trading days. Predictions are out-of-sample.")

st.subheader("Strategy backtest vs SPY")
equity, backtest_data = backtest(predictions, starting_capital=10000, transaction_cost_bps=transaction_cost_bps, slippage_bps=slippage_bps, confidence_threshold=confidence_threshold)
spy = benchmark_equity(df.loc[equity.index, "SPY_Close"], starting_capital=10000)
strategy_metrics = calculate_metrics(equity)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Strategy return", f"{strategy_metrics["total_return"]:.1%}")
m2.metric("Annualized return", f"{strategy_metrics["annualized_return"]:.1%}")
m3.metric("Sharpe", f"{strategy_metrics["sharpe"]:.2f}")
m4.metric("Max drawdown", f"{strategy_metrics["max_drawdown"]:.1%}")

fig = go.Figure()
fig.add_trace(go.Scatter(x=equity.index, y=equity.values, name="MarketSignal"))
fig.add_trace(go.Scatter(x=spy.index, y=spy.values, name="SPY buy & hold"))
fig.update_layout(yaxis_title="Portfolio value ($)", xaxis_title="Date", hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

st.caption(f"Backtest assumptions: {transaction_cost_bps:.0f} bps transaction cost + {slippage_bps:.0f} bps slippage. Signals formed at the close affect the following day return.")

with st.expander("Model comparison"):
    with st.spinner("Comparing models..."):
        comparison = compare_models(df, TECHNICAL_FEATURES)
    st.dataframe(comparison.style.format({
        "accuracy": "{:.3f}", "precision": "{:.3f}", "recall": "{:.3f}", "f1": "{:.3f}", "roc_auc": "{:.3f}", "log_loss": "{:.3f}"
    }), use_container_width=True)

st.subheader("Feature importance")
try:
    importance = pd.Series(model.feature_importances_, index=TECHNICAL_FEATURES).sort_values(ascending=False)
    st.bar_chart(importance.head(12))
except AttributeError:
    st.info("Feature importance is available for tree-based models.")

st.subheader("Recent price history")
recent = df.tail(180)
price_fig = go.Figure()
price_fig.add_trace(go.Scatter(x=recent.index, y=recent["Close"], name=ticker))
price_fig.update_layout(yaxis_title="Close", xaxis_title="Date")
st.plotly_chart(price_fig, use_container_width=True)

st.warning("Research/education only. Historical backtests do not establish future performance and this app is not financial advice.")