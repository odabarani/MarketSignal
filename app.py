import os
import sys
import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html, no_update
from flask import g, has_request_context

from src.backtest import backtest, calculate_metrics, equity_from_returns, trading_metrics
from src.data_loader import get_stock_data
from src.features import TECHNICAL_FEATURES, add_features
from src.model import compare_models, predict_latest, walk_forward_evaluate
from src.signals import generate_signal
from src.snapshots import SnapshotStore, normalize_symbol
from src.reports import ReportStore, build_report
from src.jobs import JobManager, execute_research
from src.observability import register_observability
from src.market_calendar import session_quality, validate_us_sessions
from src.errors import DataQualityError
from src.settings import validate_settings
from src import study_ui, library
from src.installable import register_installable

BASE_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
app = Dash(
    __name__,
    title="MarketSignal Research",
    assets_folder=str(BASE_DIR / "assets"),
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1, viewport-fit=cover"},
        {"name": "theme-color", "content": "#07101f"},
        {"name": "apple-mobile-web-app-capable", "content": "yes"},
        {"name": "apple-mobile-web-app-status-bar-style", "content": "black-translucent"},
        {"name": "apple-mobile-web-app-title", "content": "MarketSignal"},
    ],
)
app.index_string = app.index_string.replace("{%favicon%}", '''{%favicon%}
        <link rel="manifest" href="/manifest.webmanifest">
        <link rel="icon" type="image/png" href="/assets/icons/app-192.png">
        <link rel="apple-touch-icon" href="/assets/icons/app-180.png">''')
server = app.server
register_installable(server, BASE_DIR)
MARKET_STORE = SnapshotStore(Path(os.getenv("MARKETSIGNAL_DATA_DIR", BASE_DIR / "data")) / "snapshots.sqlite")
JOBS = JobManager(execute_research)
register_observability(server, lambda: MARKET_STORE, lambda: JOBS)


def load_market(symbol):
    raw = MARKET_STORE.get(symbol, get_stock_data)
    if session_quality(raw.index.max())["lag_sessions"] > 2:
        raw = MARKET_STORE.get(symbol, get_stock_data, force_refresh=True)
        if session_quality(raw.index.max())["lag_sessions"] > 2:
            raise DataQualityError("Market data is more than two completed sessions behind. Research stopped until the provider has fresh data.")
    validate_us_sessions(raw.index)
    return raw, add_features(raw)


def run_model_research(symbol, retrain_days):
    raw, _ = load_market(symbol)
    return fit_snapshot(raw.attrs["snapshot"]["id"], retrain_days)


@lru_cache(maxsize=16)
def fit_snapshot(snapshot_id, retrain_days):
    """Reuse fits only for the exact retained data snapshot and settings."""
    raw = MARKET_STORE.load(snapshot_id)
    features = add_features(raw)
    _, predictions, metrics = walk_forward_evaluate(
        features, TECHNICAL_FEATURES,
        retrain_every=int(retrain_days), model_name="XGBoost",
    )
    current_model, latest, up_probability = predict_latest(
        features, TECHNICAL_FEATURES, model_name="XGBoost"
    )
    return raw, features, predictions, metrics, current_model, latest, up_probability


@lru_cache(maxsize=16)
def run_model_comparison(snapshot_id, retrain_days):
    features = add_features(MARKET_STORE.load(snapshot_id))
    return compare_models(
        features, TECHNICAL_FEATURES, retrain_every=int(retrain_days)
    )


def metric_card(label, value, note=None):
    return html.Div([
        html.Div(label, className="metric-label"),
        html.Div(value, className="metric-value"),
        html.Div(note, className="metric-note") if note else None,
    ], className="metric-card")


def research_welcome():
    return html.Div([
        html.Div("Research preview", className="eyebrow"),
        html.H2("Explore the evidence behind a stock-direction estimate."),
        html.P("Choose a US stock ticker and select Run research to compare a "
               "simulated strategy with the stock and SPY."),
        html.Ol([
            html.Li("Inspect the data date and evaluation sample."),
            html.Li("Compare returns after your chosen costs and slippage."),
            html.Li("Review validation and methodology before interpreting the estimate."),
        ]),
        html.P("Daily historical data · 5-session forecast · long or cash simulation. "
               "No orders are placed. Estimates are experimental and can be wrong.",
               className="figure-note"),
    ], className="welcome-card")


def research_evidence(raw, predictions, latest, as_of=None):
    quality_as_of = pd.Timestamp(as_of) + pd.Timedelta(hours=23, minutes=59) if as_of is not None else None
    as_of = as_of or datetime.now(timezone.utc).date()
    last_date = raw.index.max().date()
    age = (as_of - last_date).days
    quality = session_quality(last_date, quality_as_of)
    return html.Div([
        html.H2("Evidence for this run"),
        html.Div([
            metric_card("Latest market date", str(last_date), "Yahoo Finance · daily history"),
            metric_card("Evaluation period",
                        f"{predictions.index.min().date()} – {predictions.index.max().date()}",
                        "Walk-forward prediction dates"),
            metric_card("Evaluated predictions", f"{len(predictions):,}",
                        "Overlapping 5-session outcomes; not independent trials"),
            metric_card("History rows", f"{len(raw):,}",
                        f"From {raw.index.min().date()} · estimate uses {latest.name.date()}"),
        ], className="metrics-grid evidence-grid"),
        html.P(
            f"Latest completed US market session: {quality['latest_completed_session']} · "
            f"Data lag: {quality['lag_sessions']} completed sessions. "
            "History more than two completed sessions behind cannot start new research.",
            className="freshness-warning" if quality["lag_sessions"] else "figure-note",
        ),
        html.P(
            f"Data is {age} calendar days old. Check the trading calendar before "
            "interpreting this estimate. These results do not describe today's market."
            if age > 7 else
            "Dates refer to historical market sessions, not a real-time quote. "
            "Weekends and market holidays can delay the latest available session.",
            className="freshness-warning" if age > 7 else "figure-note",
            role="status",
        ),
        html.P("Probabilities are model estimates, not guarantees or measured confidence "
               "intervals. Changing settings after seeing results can overfit this evaluation.",
               className="figure-note"),
        html.P(
            f"Dataset ID: {raw.attrs['snapshot']['id']} · "
            f"First retained: {raw.attrs['snapshot']['acquired_at']} · "
            "Provider checked at least hourly when research is requested.",
            className="snapshot-note",
        ) if "snapshot" in raw.attrs else None,
    ], className="evidence-panel")


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
    ticker = normalize_symbol(ticker)
    threshold, retrain_days, cost_bps, slippage_bps = validate_settings(
        threshold, retrain_days, cost_bps, slippage_bps
    )
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
    strategy_metrics.update(trading_metrics(backtest_data))
    report = build_report(
        raw, predictions, latest, up_probability, signal,
        {"ticker": ticker, "threshold": threshold, "retrain_days": retrain_days,
         "transaction_cost_bps": cost_bps, "slippage_bps": slippage_bps,
         "starting_capital_usd": 10000, "liquidate_at_end": True,
         "benchmark_transaction_cost_bps": 0},
        metrics, strategy_metrics, equity, stock, spy, BASE_DIR,
    )
    report_id = ReportStore(MARKET_STORE).save(report)

    classification_cards = [
        ("Accuracy", "accuracy", "Correct UP/DOWN calls"),
        ("Training baseline", "training_baseline_accuracy", "Majority class known at each fit"),
        ("Majority comparator", "majority_accuracy", "Most common class in this evaluation sample"),
        ("Precision", "precision", "Quality of UP calls"),
        ("Recall", "recall", "Share of UP moves found"),
        ("F1", "f1", "Precision and recall balance"),
        ("ROC-AUC", "roc_auc", "Ranking quality"),
        ("Brier score", "brier", "Calibration · lower is better"),
        ("Log loss", "log_loss", "Confident errors · lower is better"),
    ]

    overview = html.Div([
        html.H2("Exploratory strategy evaluation"),
        html.Div([
            metric_card("Total return", f"{strategy_metrics['total_return']:.1%}"),
            metric_card("Annual return", f"{strategy_metrics['annualized_return']:.1%}"),
            metric_card("Sharpe ratio", f"{strategy_metrics['sharpe']:.2f}"),
            metric_card("Max drawdown", f"{strategy_metrics['max_drawdown']:.1%}"),
        ], className="metrics-grid"),
        html.Div([
            metric_card("Stock buy & hold", f"{calculate_metrics(stock)['total_return']:.1%}", "Gross return including dividends"),
            metric_card("SPY buy & hold", f"{calculate_metrics(spy)['total_return']:.1%}", "Gross return including dividends"),
            metric_card("Entries / exits", f"{strategy_metrics['entries']} / {strategy_metrics['exits']}", "Includes final liquidation"),
            metric_card("Time in market", f"{strategy_metrics['time_in_market']:.1%}", "Share of simulated open-to-open intervals"),
        ], className="metrics-grid"),
        html.Div(dcc.Graph(
            figure=research_figure(equity, stock, spy, ticker),
            config={"displayModeBar": False}, responsive=True,
        ), className="chart-card"),
        html.P(
            f"Next-open execution · {float(cost_bps):.0f} bps cost + "
            f"{float(slippage_bps):.0f} bps slippage · final exit charged · cash dividends included. "
            "Benchmarks use identical dates, without trading costs.",
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
        dcc.Store(id="comparison-job"),
        dcc.Interval(id="comparison-poll", interval=1000, disabled=True),
        html.Div(id="comparison-submit-status", role="status"),
        html.Div(id="comparison-job-status", role="status"),
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
                "Yahoo OHLC prices are split-normalized historical prices; cash dividends "
                "are credited when a position is held into the ex-date. The forecast target "
                "remains price direction. Revised histories are not point-in-time archives. "
                "Costs and dividend reinvestment are simplified; benchmarks exclude costs. "
                "Use the **Brier score** and calibration chart with accuracy."
            ),
        ], className="method-card"),
    ])

    return html.Div([
        dcc.Store(id="research-context", data={
            "snapshot_id": raw.attrs["snapshot"]["id"],
            "ticker": ticker, "retrain_days": retrain_days,
            "report_id": report_id,
        }),
        html.Div([
            html.Button("Download research report", id="export-button", n_clicks=0,
                        className="secondary-button"),
            html.P("JSON report includes this run's settings, evidence, benchmark metrics "
                   "and simulated equity curves.", className="figure-note"),
            dcc.Download(id="report-download"),
            html.Div(id="export-status", role="status"),
        ], className="report-actions"),
        html.Div([
            html.Strong(f"Current research snapshot · {latest.name.date()}"),
            html.Span(
                "Model information ends at this close. A simulated trade enters "
                "at the next market open."
            ),
            html.Span("Settings are editable. This dashboard is not a locked final test."),
        ], className="research-note"),
        html.Div([
            metric_card("UP probability", f"{up_probability:.1%}"),
            metric_card("DOWN probability", f"{down_probability:.1%}"),
            metric_card("Research signal", signal),
            metric_card("Close as of date", f"USD {latest['Close']:,.2f}"),
        ], className="metrics-grid headline-metrics"),
        research_evidence(raw, predictions, latest),
        dcc.Tabs([
            dcc.Tab(overview, label="Overview"),
            dcc.Tab(validation, label="Validation"),
            dcc.Tab(method, label="How it works"),
        ], className="research-tabs"),
    ])


def comparison_table(frame):
    columns = [
        ("Model", "Model"), ("Accuracy", "accuracy"),
        ("Training baseline", "training_baseline_accuracy"), ("ROC-AUC", "roc_auc"),
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
    html.A("Skip to workspace", href="#workspace", className="skip-link"),
    html.Header([
        html.Div([
            html.Img(src="/assets/icons/mark.svg", className="brand-mark", alt=""),
            html.Div([
                html.Div("Independent market research", className="eyebrow"),
                html.H1("MarketSignal"),
            ]),
            html.Span("Research preview", className="release-badge"),
        ], className="brand-row"),
        html.P("Explore an idea. Test the evidence. Keep the record.", className="subtitle"),
        html.Details([
            html.Summary("Install app", className="install-summary"),
            html.Div([
                html.H2("Your research, within reach"),
                html.P("One app for your phone and computer", id="installation-state"),
                html.Button("Install MarketSignal", id="install-app-button", hidden=True, className="primary-button"),
                html.Dl([
                    html.Dt("iPhone / iPad"),
                    html.Dd("Open in Safari → Share → Add to Home Screen. Enable Open as Web App if shown."),
                    html.Dt("Android"),
                    html.Dd("Open in Chrome → menu → Install app or Add to Home screen."),
                    html.Dt("Windows / Mac"),
                    html.Dd("In Chrome or Edge, use the install icon in the address bar. On recent macOS, Safari also offers File → Add to Dock."),
                ]),
                html.P("Internet is required for data and calculations. Saved studies live on this server; there are no private accounts yet.", className="figure-note"),
            ], className="install-guide"),
        ], className="install-panel"),
    ], className="page-header"),
    html.Div("You’re offline. Reconnect to run research or refresh results. Submitted studies may continue on the server.",
             id="connection-banner", className="connection-banner", role="status", hidden=True),
    html.Main([
        html.Aside([
            html.Div("WORKSPACE", className="eyebrow"),
            dcc.RadioItems(
                id="workspace-mode", value="research",
                options=[{"label": "Research", "value": "research"}, {"label": "Studies", "value": "studies"}, {"label": "Saved", "value": "saved"}],
                className="workspace-switch", persistence=True, persistence_type="local",
            ),
            html.H2("Choose your market"),
            html.Label("US stock ticker", htmlFor="ticker"),
            dcc.Input(id="ticker", value="AAPL", type="text", debounce=True,
                      persistence=True, persistence_type="local", maxLength=12),
            html.P("Daily US stock data · USD", className="field-hint"),
            html.Div([
                html.Details([
                    html.Summary("Strategy settings"),
                    html.Label("Confidence threshold", htmlFor="threshold"),
                    dcc.Slider(
                        id="threshold", min=.5, max=.8, step=.01, value=.6,
                        marks={value: {"label": label, "style": {"color": "#91a4bf"}}
                               for value, label in [(.5, "50%"), (.6, "60%"), (.7, "70%"), (.8, "80%") ]},
                        tooltip={"placement": "bottom"},
                    ),
                    html.Label("Retrain frequency"),
                    dcc.RadioItems(
                        id="retrain-days", value=21,
                        options=[{"label": "21d", "value": 21}, {"label": "42d", "value": 42}, {"label": "63d", "value": 63}],
                        inline=True, className="retrain-picker",
                    ),
                    html.Label("Transaction cost (bps)", htmlFor="cost-bps"),
                    dcc.Input(id="cost-bps", value=10, type="number", min=0, max=100),
                    html.Label("Slippage (bps)", htmlFor="slippage-bps"),
                    dcc.Input(id="slippage-bps", value=5, type="number", min=0, max=100),
                ], className="strategy-settings"),
                html.Button("Run research", id="run-button", n_clicks=0, className="primary-button"),
            ], id="exploratory-controls"),
            html.P("Research only · not financial advice", className="disclaimer"),
        ], className="control-panel"),
        html.Section([
            html.Div([
                dcc.Store(id="research-job", storage_type="session"),
                dcc.Interval(id="research-poll", interval=1000, disabled=True),
                html.Div(id="research-submit-status", role="status"),
                html.Div(id="research-job-status", role="status"),
                dcc.Loading(html.Div(research_welcome(), id="research-results"), type="circle", color="#38bdf8"),
            ], id="research-workspace"),
            html.Div(study_ui.layout(), id="study-workspace", style={"display": "none"}),
            html.Div(library.layout(), id="library-workspace", style={"display": "none"}),
        ], className="results-panel", id="workspace", tabIndex=-1),
    ], className="app-shell"),
    html.Footer("MarketSignal · Research preview. Historical results do not predict future performance."),
])

app.clientside_callback(
    """function(mode) {
        const hidden = {display: 'none'};
        return [mode === 'research' ? {} : hidden, mode === 'studies' ? {} : hidden,
                mode === 'research' ? {} : hidden, mode === 'saved' ? {} : hidden];
    }""",
    Output("research-workspace", "style"), Output("study-workspace", "style"),
    Output("exploratory-controls", "style"), Output("library-workspace", "style"), Input("workspace-mode", "value"),
)


@app.callback(
    Output("research-job", "data"),
    Output("research-submit-status", "children"),
    Input("run-button", "n_clicks"),
    State("ticker", "value"),
    State("threshold", "value"),
    State("retrain-days", "value"),
    State("cost-bps", "value"),
    State("slippage-bps", "value"),
    prevent_initial_call=True,
)
def submit_research(clicks, ticker, threshold, retrain_days, cost_bps, slippage_bps):
    if not clicks:
        return no_update, no_update
    try:
        ticker = normalize_symbol(ticker)
        settings = validate_settings(threshold, retrain_days, cost_bps, slippage_bps)
        args = (ticker, *settings)
        job_id = JOBS.submit(("research", *args), {
            "kind": "research", "args": args, "store_path": str(MARKET_STORE.path),
            "request_id": g.request_id if has_request_context() else None,
        })
        return job_id, ""
    except ValueError as exc:
        return no_update, html.P(str(exc), className="freshness-warning")
    except Exception:
        server.logger.exception("research_submission_failed")
        return no_update, html.P("Research service is unavailable. Please try again.", className="freshness-warning")


@app.callback(
    Output("research-results", "children"),
    Output("research-job-status", "children"),
    Output("research-poll", "disabled"),
    Output("run-button", "disabled"),
    Input("research-job", "data"),
    Input("research-poll", "n_intervals"),
    prevent_initial_call=True,
)
def poll_research(job_id, _):
    return poll_job(job_id)


def poll_job(job_id):
    if not job_id:
        return no_update, "", True, False
    status = JOBS.status(job_id)
    if status["state"] in ("queued", "running"):
        text = "Research queued. Waiting for the worker…" if status["state"] == "queued" else "Research running. Downloading data and evaluating models…"
        return no_update, html.P(text, className="figure-note"), False, True
    if status["state"] == "completed":
        return status["result"], "Research complete.", True, False
    return no_update, html.P(status["result"], className="freshness-warning"), True, False


@app.callback(
    Output("comparison-job", "data"),
    Output("comparison-submit-status", "children"),
    Input("compare-button", "n_clicks"),
    State("research-context", "data"),
    prevent_initial_call=True,
)
def submit_comparison(clicks, context):
    if not clicks:
        return no_update, no_update
    try:
        if not isinstance(context, dict):
            raise ValueError("Run research before comparing models.")
        report = ReportStore(MARKET_STORE).load((context or {}).get("report_id"))
        snapshot_id = report["dataset"]["id"]
        retrain_days = report["settings"]["retrain_days"]
        job_id = JOBS.submit(("comparison", snapshot_id, retrain_days), {
            "kind": "comparison", "args": (snapshot_id, retrain_days),
            "store_path": str(MARKET_STORE.path),
            "request_id": g.request_id if has_request_context() else None,
        })
        return job_id, ""
    except ValueError as exc:
        return no_update, html.P(str(exc), className="freshness-warning")
    except Exception:
        server.logger.exception("comparison_submission_failed")
        return no_update, html.P("Research service is unavailable. Please try again.", className="freshness-warning")


@app.callback(
    Output("comparison-results", "children"),
    Output("comparison-job-status", "children"),
    Output("comparison-poll", "disabled"),
    Output("compare-button", "disabled"),
    Input("comparison-job", "data"),
    Input("comparison-poll", "n_intervals"),
    prevent_initial_call=True,
)
def poll_comparison(job_id, _):
    return poll_job(job_id)


@app.callback(
    Output("report-download", "data"),
    Output("export-status", "children"),
    Input("export-button", "n_clicks"),
    State("research-context", "data"),
    prevent_initial_call=True,
)
def download_report(clicks, context):
    if not clicks:
        return no_update, no_update
    try:
        report_id = (context or {}).get("report_id")
        report = ReportStore(MARKET_STORE).load(report_id)
        filename = f"MarketSignal-{report['settings']['ticker']}-{report_id[:12]}.json"
        content = json.dumps({"report_id": report_id, "report": report},
                             indent=2, sort_keys=True, allow_nan=False)
        return dcc.send_string(content, filename, type="application/json"), "Report downloaded."
    except ValueError as exc:
        return no_update, html.P(str(exc), className="freshness-warning")
    except Exception:
        server.logger.exception("Research report export failed")
        return no_update, html.P("Report download failed. Please try again.", className="freshness-warning")


study_ui.register(app, lambda: MARKET_STORE, JOBS)
library.register(app, lambda: MARKET_STORE)


if __name__ == "__main__":
    app.run(
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8050")),
        debug=False,
    )
