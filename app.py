import os
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html

from src.backtest import backtest, calculate_metrics, equity_from_returns
from src.data_loader import get_stock_data
from src.features import TECHNICAL_FEATURES, add_features
from src.model import compare_models, predict_latest, walk_forward_evaluate
from src.signals import generate_signal

BASE_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
app = Dash(
    __name__,
    title="MarketSignal Research",
    assets_folder=str(BASE_DIR / "assets"),
)
server = app.server


@lru_cache(maxsize=16)
def load_market(symbol):
    raw = get_stock_data(symbol)
    return raw, add_features(raw)


@lru_cache(maxsize=16)
def run_model_research(symbol, retrain_days):
    """Cache the expensive model fit separately from cheap backtest settings."""
    raw, features = load_market(symbol)
    _, predictions, metrics = walk_forward_evaluate(
        features, TECHNICAL_FEATURES,
        retrain_every=int(retrain_days), model_name="XGBoost",
    )
    current_model, latest, up_probability = predict_latest(
        features, TECHNICAL_FEATURES, model_name="XGBoost"
    )
    return raw, features, predictions, metrics, current_model, latest, up_probability


@lru_cache(maxsize=16)
def run_model_comparison(symbol, retrain_days):
    _, features = load_market(symbol)
    return compare_models(
        features, TECHNICAL_FEATURES, retrain_every=int(retrain_days)
    )


def metric_card(label, value, note=None):
    return html.Div([
        html.Div(label, className="metric-label"),
        html.Div(value, className="metric-value"),
        html.Div(note, className="metric-note") if note else None,
    ], className="metric-card")


def research_figure(equity, stock, spy, ticker):
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=equity.index, y=equity, name="MarketSignal",
        line=dict(width=3, color="#38bdf8"),
    ))
    figure.add_trace(go.Scatter(
        x=stock.index, y=stock, name=f"{ticker} buy & hold",
        line=dict(color="#a78bfa"),
    ))
    figure.add_trace(go.Scatter(
        x=spy.index, y=spy, name="SPY buy & hold",
        line=dict(color="#94a3b8"),
    ))
    figure.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", yaxis_title="Portfolio value (USD)",
        xaxis_title=None, hovermode="x unified",
        legend=dict(orientation="h", y=1.08),
        margin=dict(l=20, r=10, t=55, b=20),
    )
    return figure


def calibration_figure(predictions):
    calibration = predictions.assign(
        probability_band=pd.cut(
            predictions["Probability"], bins=[0, .2, .4, .6, .8, 1],
            include_lowest=True,
        )
    ).groupby("probability_band", observed=True).agg(
        predicted_up=("Probability", "mean"),
        actual_up=("Target", "mean"),
    )
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], name="Perfect calibration",
        line=dict(color="#64748b", dash="dash"),
    ))
    figure.add_trace(go.Scatter(
        x=calibration["predicted_up"], y=calibration["actual_up"],
        mode="lines+markers", name="XGBoost",
        marker=dict(size=10, color="#38bdf8"),
    ))
    figure.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Average predicted UP probability",
        yaxis_title="Observed UP rate",
        xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]),
        margin=dict(l=20, r=10, t=25, b=20),
    )
    return figure


def price_figure(raw, ticker):
    recent = raw.tail(180)
    figure = go.Figure(go.Scatter(
        x=recent.index, y=recent["Close"], name=ticker,
        line=dict(color="#38bdf8", width=2),
    ))
    figure.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", yaxis_title="Close (USD)",
        xaxis_title=None, margin=dict(l=20, r=10, t=25, b=20),
    )
    return figure


def importance_figure(model):
    importance = pd.Series(
        model.feature_importances_, index=TECHNICAL_FEATURES
    ).sort_values().tail(12)
    figure = go.Figure(go.Bar(
        x=importance.values, y=importance.index,
        orientation="h", marker_color="#38bdf8",
    ))
    figure.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", xaxis_title="Relative importance",
        margin=dict(l=20, r=10, t=25, b=20),
    )
    return figure


def build_research_dashboard(ticker, threshold, retrain_days, cost_bps, slippage_bps):
    ticker = (ticker or "").upper().strip()
    (
        raw, _, predictions, metrics, current_model, latest, up_probability
    ) = run_model_research(
        ticker, int(retrain_days)
    )
    down_probability = 1 - up_probability
    signal = generate_signal(
        int(up_probability >= 0.5), up_probability, float(threshold)
    )

    equity, backtest_data = backtest(
        predictions, starting_capital=10000,
        transaction_cost_bps=float(cost_bps),
        slippage_bps=float(slippage_bps),
        confidence_threshold=float(threshold),
    )
    realization_dates = pd.DatetimeIndex(backtest_data["RealizationDate"])
    spy_returns = pd.Series(
        backtest_data["benchmark_return"].to_numpy(), index=realization_dates
    )
    stock_returns = pd.Series(
        backtest_data["market_return"].to_numpy(), index=realization_dates
    )
    spy = equity_from_returns(spy_returns, 10000, equity.index[0])
    stock = equity_from_returns(stock_returns, 10000, equity.index[0])
    strategy_metrics = calculate_metrics(equity)

    classification_cards = [
        ("Accuracy", "accuracy", "Correct UP/DOWN calls"),
        ("Majority baseline", "majority_accuracy", "Always choose the common class"),
        ("Precision", "precision", "Quality of UP calls"),
        ("Recall", "recall", "Share of UP moves found"),
        ("F1", "f1", "Precision and recall balance"),
        ("ROC-AUC", "roc_auc", "Ranking quality"),
        ("Brier score", "brier", "Calibration · lower is better"),
        ("Log loss", "log_loss", "Confident errors · lower is better"),
    ]

    overview = html.Div([
        html.H2("Out-of-sample strategy"),
        html.Div([
            metric_card("Total return", f"{strategy_metrics['total_return']:.1%}"),
            metric_card("Annual return", f"{strategy_metrics['annualized_return']:.1%}"),
            metric_card("Sharpe ratio", f"{strategy_metrics['sharpe']:.2f}"),
            metric_card("Max drawdown", f"{strategy_metrics['max_drawdown']:.1%}"),
        ], className="metrics-grid"),
        html.Div(dcc.Graph(
            figure=research_figure(equity, stock, spy, ticker),
            config={"displayModeBar": False}, responsive=True,
        ), className="chart-card"),
        html.P(
            f"Next-open execution · {float(cost_bps):.0f} bps cost + "
            f"{float(slippage_bps):.0f} bps slippage · identical comparison dates.",
            className="figure-note",
        ),
        html.H2("Recent market history"),
        html.Div(dcc.Graph(
            figure=price_figure(raw, ticker),
            config={"displayModeBar": False}, responsive=True,
        ), className="chart-card"),
    ])

    validation = html.Div([
        html.H2("Walk-forward validation"),
        html.Div([
            metric_card(label, f"{metrics[key]:.3f}", note)
            for label, key, note in classification_cards
        ], className="metrics-grid"),
        html.P(
            f"The oldest 70% starts training. The model retrains every "
            f"{int(retrain_days)} sessions. A 5-session label embargo blocks "
            "outcomes that were not known at fit time.",
            className="figure-note",
        ),
        html.Div([
            html.Div([
                html.H2("Probability calibration"),
                dcc.Graph(
                    figure=calibration_figure(predictions),
                    config={"displayModeBar": False}, responsive=True,
                ),
            ], className="chart-card"),
            html.Div([
                html.H2("Feature importance"),
                dcc.Graph(
                    figure=importance_figure(current_model),
                    config={"displayModeBar": False}, responsive=True,
                ),
            ], className="chart-card"),
        ], className="two-column"),
        html.Button("Compare all models", id="compare-button", className="secondary-button"),
        dcc.Loading(html.Div(id="comparison-results"), type="circle"),
    ])

    method = html.Div([
        html.H2("Research method"),
        html.Div([
            html.H3("Target"),
            dcc.Markdown(
                "`UP` means the close five sessions later is higher. The newest "
                "five rows can produce a current estimate even though their targets are unknown."
            ),
            html.H3("Walk-forward validation"),
            dcc.Markdown(
                "Time only moves forward. **Label embargo** removes any result "
                "that was unavailable when a model was fitted."
            ),
            html.H3("Signal rule"),
            dcc.Markdown(
                "The model reports `P(UP)`. A strong UP estimate enters long, "
                "a strong `P(DOWN) = 1 − P(UP)` exits to cash, and the middle "
                "range keeps the prior position."
            ),
            html.H3("Backtest"),
            dcc.Markdown(
                "A decision after today's close enters at the **next open**. "
                "MarketSignal, the selected stock, and SPY use the same dates."
            ),
            html.H3("Limits"),
            dcc.Markdown(
                "Yahoo Finance data can be revised and costs are simplified. "
                "Use the **Brier score** and calibration chart with accuracy."
            ),
        ], className="method-card"),
    ])

    return html.Div([
        html.Div([
            html.Strong(f"Current research snapshot · {latest.name.date()}"),
            html.Span(
                "Model information ends at this close. A simulated trade enters "
                "at the next market open."
            ),
        ], className="research-note"),
        html.Div([
            metric_card("UP probability", f"{up_probability:.1%}"),
            metric_card("DOWN probability", f"{down_probability:.1%}"),
            metric_card("Research signal", signal),
            metric_card("Close as of date", f"USD {latest['Close']:,.2f}"),
        ], className="metrics-grid headline-metrics"),
        dcc.Tabs([
            dcc.Tab(overview, label="Overview"),
            dcc.Tab(validation, label="Validation"),
            dcc.Tab(method, label="How it works"),
        ], className="research-tabs"),
    ])


def comparison_table(frame):
    columns = [
        ("Model", "Model"), ("Accuracy", "accuracy"),
        ("Baseline", "majority_accuracy"), ("ROC-AUC", "roc_auc"),
        ("Brier", "brier"), ("Log loss", "log_loss"),
    ]
    return html.Div([
        html.Table([
            html.Thead(html.Tr([html.Th(label) for label, _ in columns])),
            html.Tbody([
                html.Tr([
                    html.Td(row[key] if key == "Model" else f"{row[key]:.3f}")
                    for _, key in columns
                ]) for _, row in frame.iterrows()
            ]),
        ], className="comparison-table"),
    ], className="table-card")


app.layout = html.Div([
    html.Header([
        html.Div("Machine-learning research dashboard", className="eyebrow"),
        html.H1("MarketSignal"),
        html.P(
            "A transparent 5-day stock-direction experiment with walk-forward testing.",
            className="subtitle",
        ),
    ], className="page-header"),
    html.Main([
        html.Aside([
            html.H2("Research settings"),
            html.Label("Ticker"),
            dcc.Input(id="ticker", value="AAPL", type="text", debounce=True),
            html.Label("Confidence threshold"),
            dcc.Slider(
                id="threshold", min=.5, max=.8, step=.01, value=.6,
                marks={
                    value: {"label": label, "style": {"color": "#91a4bf"}}
                    for value, label in [(.5, "50%"), (.6, "60%"),
                                         (.7, "70%"), (.8, "80%")]
                },
                tooltip={"placement": "bottom"},
            ),
            html.Label("Retrain frequency"),
            dcc.RadioItems(
                id="retrain-days", value=21,
                options=[
                    {"label": "21d", "value": 21},
                    {"label": "42d", "value": 42},
                    {"label": "63d", "value": 63},
                ],
                inline=True, className="retrain-picker",
            ),
            html.Label("Transaction cost (bps)"),
            dcc.Input(id="cost-bps", value=10, type="number", min=0, max=100),
            html.Label("Slippage (bps)"),
            dcc.Input(id="slippage-bps", value=5, type="number", min=0, max=100),
            html.Button("Run research", id="run-button", n_clicks=0, className="primary-button"),
            html.P("Research only · not financial advice", className="disclaimer"),
        ], className="control-panel"),
        html.Section([
            dcc.Loading(
                html.Div(id="research-results"),
                type="circle", color="#38bdf8",
            )
        ], className="results-panel"),
    ], className="app-shell"),
    html.Footer(
        "MarketSignal is an educational research project. Historical results "
        "do not predict future performance."
    ),
])


@app.callback(
    Output("research-results", "children"),
    Input("run-button", "n_clicks"),
    State("ticker", "value"),
    State("threshold", "value"),
    State("retrain-days", "value"),
    State("cost-bps", "value"),
    State("slippage-bps", "value"),
)
def update_research(_, ticker, threshold, retrain_days, cost_bps, slippage_bps):
    try:
        return build_research_dashboard(
            ticker, threshold, retrain_days, cost_bps, slippage_bps
        )
    except Exception as exc:
        return html.Div([
            html.H3("Research run stopped"),
            html.P(str(exc)),
            html.P("Try a liquid ticker with several years of price history."),
        ], className="error-card")


@app.callback(
    Output("comparison-results", "children"),
    Input("compare-button", "n_clicks"),
    State("ticker", "value"),
    State("retrain-days", "value"),
    prevent_initial_call=True,
)
def update_comparison(_, ticker, retrain_days):
    try:
        comparison = run_model_comparison(
            (ticker or "").upper().strip(), int(retrain_days)
        )
        return comparison_table(comparison)
    except Exception as exc:
        return html.Div(str(exc), className="error-card")


if __name__ == "__main__":
    app.run(
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8050")),
        debug=False,
    )
