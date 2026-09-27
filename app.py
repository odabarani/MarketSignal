import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.backtest import backtest, calculate_metrics, equity_from_returns
from src.data_loader import get_stock_data
from src.features import TECHNICAL_FEATURES, add_features
from src.model import compare_models, predict_latest, walk_forward_evaluate
from src.signals import generate_signal

st.set_page_config(page_title="MarketSignal Research", page_icon="📈", layout="wide")
st.markdown("""
<style>
    .block-container {max-width: 1180px; padding-top: 2rem;}
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #101a2d, #15243d);
        border: 1px solid #263b5d; border-radius: 14px; padding: 14px;
    }
    .research-note {
        background: #0e1b2e; border-left: 4px solid #38bdf8;
        border-radius: 8px; padding: 12px 16px; margin: 8px 0 18px;
    }
    .eyebrow {color: #38bdf8; font-size: .78rem; font-weight: 700;
        letter-spacing: .12em; text-transform: uppercase;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="eyebrow">Machine-learning research dashboard</div>', unsafe_allow_html=True)
st.title("MarketSignal")
st.caption("A transparent 5-day stock-direction experiment with walk-forward testing.")

with st.sidebar:
    st.header("Research settings")
    ticker = st.text_input("Ticker", "AAPL").upper().strip()
    confidence_threshold = st.slider(
        "Confidence threshold", 0.50, 0.80, 0.60, 0.01,
        help="UP or DOWN needs at least this probability. Otherwise the position stays unchanged.",
    )
    retrain_days = st.selectbox(
        "Retrain every", [21, 42, 63], index=0,
        format_func=lambda value: f"{value} trading days",
    )
    transaction_cost_bps = st.number_input("Transaction cost (bps)", 0.0, 100.0, 10.0, 1.0)
    slippage_bps = st.number_input("Slippage (bps)", 0.0, 100.0, 5.0, 1.0)
    st.divider()
    st.caption("Research only · not financial advice")


@st.cache_data(ttl=3600)
def load_market(symbol):
    raw = get_stock_data(symbol)
    return raw, add_features(raw)


@st.cache_resource(show_spinner=False)
def run_research(feature_data, retrain_frequency):
    return walk_forward_evaluate(
        feature_data, TECHNICAL_FEATURES,
        retrain_every=retrain_frequency, model_name="XGBoost",
    )


@st.cache_resource(show_spinner=False)
def run_current_model(feature_data):
    return predict_latest(feature_data, TECHNICAL_FEATURES, model_name="XGBoost")


try:
    raw, features = load_market(ticker)
    with st.spinner("Running the walk-forward study…"):
        _, predictions, metrics = run_research(features, retrain_days)
        current_model, latest, up_probability = run_current_model(features)
except Exception as exc:
    st.error(f"Research run stopped: {exc}")
    st.info("Try a liquid ticker with several years of price history.")
    st.stop()

down_probability = 1 - up_probability
signal = generate_signal(int(up_probability >= 0.5), up_probability, confidence_threshold)
as_of = latest.name.date()

st.markdown(
    f'<div class="research-note"><b>Current research snapshot · {as_of}</b><br>'
    "The model uses information available at this close. Any simulated trade enters at the "
    "<b>next market open</b>.</div>",
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("UP probability", f"{up_probability:.1%}")
c2.metric("DOWN probability", f"{down_probability:.1%}")
c3.metric("Research signal", signal)
c4.metric("Close as of date", f"USD {latest['Close']:,.2f}")

overview_tab, validation_tab, method_tab = st.tabs(
    ["Overview", "Validation", "How it works"]
)

with overview_tab:
    st.subheader("Out-of-sample strategy")
    try:
        equity, backtest_data = backtest(
            predictions,
            starting_capital=10000,
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
            confidence_threshold=confidence_threshold,
        )
        realization_dates = pd.DatetimeIndex(backtest_data["RealizationDate"])
        initial_date = equity.index[0]
        spy_returns = pd.Series(
            backtest_data["benchmark_return"].to_numpy(), index=realization_dates
        )
        stock_returns = pd.Series(
            backtest_data["market_return"].to_numpy(), index=realization_dates
        )
        spy = equity_from_returns(spy_returns, 10000, initial_date)
        stock = equity_from_returns(stock_returns, 10000, initial_date)
        strategy_metrics = calculate_metrics(equity)
    except Exception as exc:
        st.error(f"Backtest could not run: {exc}")
        st.stop()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total return", f"{strategy_metrics['total_return']:.1%}")
    m2.metric("Annual return", f"{strategy_metrics['annualized_return']:.1%}")
    m3.metric("Sharpe ratio", f"{strategy_metrics['sharpe']:.2f}")
    m4.metric("Max drawdown", f"{strategy_metrics['max_drawdown']:.1%}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity.index, y=equity, name="MarketSignal", line=dict(width=3, color="#38bdf8")))
    fig.add_trace(go.Scatter(x=stock.index, y=stock, name=f"{ticker} buy & hold", line=dict(color="#a78bfa")))
    fig.add_trace(go.Scatter(x=spy.index, y=spy, name="SPY buy & hold", line=dict(color="#94a3b8")))
    fig.update_layout(
        yaxis_title="Portfolio value (USD)", xaxis_title=None, hovermode="x unified",
        legend=dict(orientation="h", y=1.08), margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Next-open execution · {transaction_cost_bps:.0f} bps cost + "
        f"{slippage_bps:.0f} bps slippage · all curves use the same dates."
    )

    st.subheader("Recent market history")
    recent = raw.tail(180)
    price_fig = go.Figure(go.Scatter(
        x=recent.index, y=recent["Close"], name=ticker,
        line=dict(color="#38bdf8", width=2),
    ))
    price_fig.update_layout(yaxis_title="Close (USD)", xaxis_title=None, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(price_fig, use_container_width=True)

with validation_tab:
    st.subheader("Walk-forward validation")
    metric_specs = [
        ("Accuracy", "accuracy", "Correct UP/DOWN classifications."),
        ("Majority baseline", "majority_accuracy", "Accuracy from always choosing the common class."),
        ("Precision", "precision", "Share of predicted UP cases that were UP."),
        ("Recall", "recall", "Share of actual UP cases the model found."),
        ("F1", "f1", "Balance between precision and recall."),
        ("ROC-AUC", "roc_auc", "Ranking quality across thresholds."),
        ("Brier score", "brier", "Probability calibration; lower is better."),
        ("Log loss", "log_loss", "Penalty for confident mistakes; lower is better."),
    ]
    for row_start in (0, 4):
        cols = st.columns(4)
        for col, (label, key, help_text) in zip(cols, metric_specs[row_start:row_start + 4]):
            value = metrics[key]
            col.metric(label, f"{value:.3f}" if pd.notna(value) else "N/A", help=help_text)
    st.caption(
        f"The first 70% starts training. Predictions then move forward in time, "
        f"retraining every {retrain_days} sessions. A 5-session label embargo prevents future outcomes from leaking into training."
    )

    st.subheader("Probability calibration")
    calibration = predictions.assign(
        probability_band=pd.cut(predictions["Probability"], bins=[0, .2, .4, .6, .8, 1], include_lowest=True)
    ).groupby("probability_band", observed=True).agg(
        predicted_up=("Probability", "mean"), actual_up=("Target", "mean"), observations=("Target", "size")
    )
    calibration_fig = go.Figure()
    calibration_fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], name="Perfect calibration",
        line=dict(color="#64748b", dash="dash"),
    ))
    calibration_fig.add_trace(go.Scatter(
        x=calibration["predicted_up"], y=calibration["actual_up"],
        mode="lines+markers", name="XGBoost", marker=dict(size=10, color="#38bdf8"),
    ))
    calibration_fig.update_layout(
        xaxis_title="Average predicted UP probability", yaxis_title="Observed UP rate",
        xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]), margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(calibration_fig, use_container_width=True)

    if st.button("Compare all models", type="secondary"):
        with st.spinner("Running the same folds for each model…"):
            comparison = compare_models(features, TECHNICAL_FEATURES, retrain_every=retrain_days)
        st.dataframe(
            comparison.style.format({
                key: "{:.3f}" for key in
                ["accuracy", "majority_accuracy", "precision", "recall", "f1", "roc_auc", "brier", "log_loss"]
            }),
            use_container_width=True, hide_index=True,
        )

    st.subheader("Feature importance")
    importance = pd.Series(
        current_model.feature_importances_, index=TECHNICAL_FEATURES
    ).sort_values(ascending=True).tail(12)
    st.bar_chart(importance, horizontal=True, color="#38bdf8")

with method_tab:
    st.subheader("Research method")
    st.markdown("""
### Target
`UP` means the close five trading sessions later is higher. The newest five rows have no known target yet, but they can still produce the current snapshot.

### Walk-forward validation
Training begins with the oldest 70% of labeled data. Time only moves forward. **Label embargo** removes any training row whose five-day result was not known at model-fit time.

### Signal rule
The model reports `P(UP)`. A strong UP probability enters long, a strong `P(DOWN) = 1 − P(UP)` exits to cash, and the middle range keeps the prior position.

### Backtest
A decision made after today's close executes at the **next open**. Returns begin at that open. Strategy, ticker buy-and-hold, and SPY use identical dates.

### Limits
Yahoo Finance data can be revised. Costs are simplified. The model is not probability-calibrated after training, so the **Brier score** and calibration chart should be read alongside accuracy.
""")

st.caption("MarketSignal is an educational research project. Historical results do not predict future performance.")
