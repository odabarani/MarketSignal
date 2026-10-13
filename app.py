import os
import sys
import json
import math
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html
from flask import Response

from src.cache import ttl_cache
from src.stock_profile import decode_selection, search_options, local_search_options, get_analyst_snapshot
from src.backtest import backtest, calculate_metrics, equity_from_returns
from src.data_loader import get_stock_data
from src.features import TECHNICAL_FEATURES, add_features
from src.model import compare_models, predict_latest, walk_forward_evaluate
from src.signals import generate_signal

BASE_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
app = Dash(
    __name__,
    title="MarketSignal Research",
    suppress_callback_exceptions=True,
    assets_folder=str(BASE_DIR / "assets"),
)
server = app.server


@server.route("/research-lab")
def research_lab():
    """Serve the saved multi-stock experiment without fitting or downloading."""
    from src.research_report import render_report

    default_report = BASE_DIR / "data/walk_forward.json"
    if not default_report.exists():
        default_report = BASE_DIR / "docs/research-lab.json"
    report_path = Path(os.getenv("MARKETSIGNAL_RESEARCH_REPORT", default_report))
    if not report_path.exists():
        return Response(
            '<!doctype html><html lang="en"><meta name="viewport" content="width=device-width">'
            '<title>MarketSignal · Research Lab</title><link rel="stylesheet" href="/assets/style.css">'
            '<main class="page-header"><div class="eyebrow">Research Lab</div>'
            '<h1>No experiment yet</h1><p>Run the walk-forward research pipeline to create '
            'a saved report. This page will show performance and calibration charts.</p>'
            '<p>Setup instructions are in the repository README.</p><a href="/">Back to MarketSignal</a></main></html>',
            mimetype="text/html",
        )
    try:
        return Response(render_report(json.loads(report_path.read_text())), mimetype="text/html")
    except (ValueError, KeyError, TypeError, OSError, StopIteration):
        return Response("Research report could not be loaded. Regenerate it with the walk-forward pipeline.",
                        status=503, mimetype="text/plain")


@ttl_cache(seconds=900, maxsize=16)
def load_market(symbol):
    raw = get_stock_data(symbol)
    return raw, add_features(raw)


@ttl_cache(seconds=900, maxsize=16)
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


@ttl_cache(seconds=900, maxsize=16)
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
        plot_bgcolor="rgba(0,0,0,0)", yaxis_title="Growth of 10,000 units",
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
        line=dict(color="#67e8f9", width=3),
    ))
    figure.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", yaxis_title="Closing price",
        xaxis_title=None, margin=dict(l=20, r=10, t=25, b=20),
        hovermode="x unified",
        yaxis=dict(range=[float(recent["Close"].min()) * .92,
                          float(recent["Close"].max()) * 1.08]),
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


def validate_settings(threshold, retrain_days, cost_bps, slippage_bps):
    try:
        values = [float(value) for value in (threshold, retrain_days, cost_bps, slippage_bps)]
    except (TypeError, ValueError):
        raise ValueError("Fill in each research setting with a number.") from None
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Research settings must be finite numbers.")
    threshold, retrain_days, cost_bps, slippage_bps = values
    if not .5 <= threshold <= .8:
        raise ValueError("Choose a confidence threshold between 50% and 80%.")
    if retrain_days not in (21, 42, 63):
        raise ValueError("Choose a retraining interval of 21, 42, or 63 sessions.")
    if not all(0 <= value <= 100 for value in (cost_bps, slippage_bps)):
        raise ValueError("Trading costs and slippage must be between 0 and 100 basis points.")
    return threshold, int(retrain_days), cost_bps, slippage_bps


def build_research_dashboard(ticker, threshold, retrain_days, cost_bps, slippage_bps):
    ticker, company = decode_selection(ticker)
    if not ticker:
        raise ValueError("Choose a stock from the company search results.")
    company = company or ticker
    threshold, retrain_days, cost_bps, slippage_bps = validate_settings(
        threshold, retrain_days, cost_bps, slippage_bps
    )
    raw, _, predictions, metrics, current_model, latest, up_probability = run_model_research(ticker, retrain_days)
    signal = generate_signal(int(up_probability >= .5), up_probability, threshold)
    view = {"UP": "Upward tilt", "DOWN": "Downward tilt", "HOLD": "No strong signal"}[signal]
    explanation = {
        "UP": "The model's upward estimate clears your confidence threshold.",
        "DOWN": "The model's down-or-flat estimate clears your confidence threshold.",
        "HOLD": "Neither direction clears your confidence threshold. The model has no strong directional view.",
    }[signal]
    equity, backtest_data = backtest(
        predictions, starting_capital=10000, transaction_cost_bps=cost_bps,
        slippage_bps=slippage_bps, confidence_threshold=threshold,
    )
    realization_dates = pd.DatetimeIndex(backtest_data["RealizationDate"])
    spy = equity_from_returns(pd.Series(backtest_data["benchmark_return"].to_numpy(), index=realization_dates),
                              10000, equity.index[0])
    stock = equity_from_returns(pd.Series(backtest_data["market_return"].to_numpy(), index=realization_dates),
                                10000, equity.index[0])
    strategy_metrics = calculate_metrics(equity)
    stock_return = stock.iloc[-1] / stock.iloc[0] - 1
    price = float(raw.Close.iloc[-1])
    daily_change = float(raw.Close.iloc[-1] / raw.Close.iloc[-2] - 1)
    accuracy_gap = metrics["accuracy"] - metrics["majority_accuracy"]
    historical_readout = (
        f"It scored {abs(accuracy_gap) * 100:.1f} percentage points "
        f"{'above' if accuracy_gap > 0 else 'below' if accuracy_gap < 0 else 'level with'} "
        "the hindsight majority-class reference. This alone does not establish an edge."
    )
    classification_cards = [
        ("Accuracy", "accuracy", "Share of correct up/down-or-flat calls", True),
        ("Majority reference", "majority_accuracy", "Always pick the test period's most common outcome; hindsight reference", True),
        ("Precision", "precision", "When the model called UP, how often it was right", True),
        ("Recall", "recall", "Share of actual UP outcomes it found", True),
        ("F1 score", "f1", "Balance between precision and recall; higher is better", False),
        ("ROC-AUC", "roc_auc", "Ranking quality: 0.5 is chance; 1 is perfect", False),
        ("Brier score", "brier", "Probability error; lower is better", False),
        ("Log loss", "log_loss", "Penalizes confident mistakes; lower is better", False),
    ]
    overview = html.Div([
        html.H2("How the price has moved"),
        html.P("The last 180 trading sessions. Hover over the chart for a date and price.", className="figure-note"),
        html.Div(dcc.Graph(figure=price_figure(raw, ticker), config={"displayModeBar": False}, responsive=True), className="chart-card"),
        html.H2("What happened in the simulation?"),
        html.P(
            f"From {equity.index[0].date()} to {equity.index[-1].date()}, the strategy returned "
            f"{strategy_metrics['total_return']:+.1%}; simply holding {ticker} returned {stock_return:+.1%}. "
            "These are historical outcomes, not a forecast.", className="figure-note",
        ),
        html.Div([
            metric_card("Simulated return", f"{strategy_metrics['total_return']:+.1%}", "Total change after modeled costs"),
            metric_card("Annualized return", f"{strategy_metrics['annualized_return']:+.1%}", "The equivalent compounded rate per year"),
            metric_card("Sharpe ratio", f"{strategy_metrics['sharpe']:.2f}", "Return relative to volatility; assumes a 0% cash rate"),
            metric_card("Largest decline", f"{strategy_metrics['max_drawdown']:.1%}", "Max drawdown: worst fall from a prior peak"),
        ], className="metrics-grid"),
        html.Div(dcc.Graph(figure=research_figure(equity, stock, spy, ticker), config={"displayModeBar": False}, responsive=True), className="chart-card"),
        html.P(
            f"Each line starts at 10,000 units on identical dates. Trades enter at the next open. "
            f"Costs: {cost_bps:g} bps + {slippage_bps:g} bps slippage per position change "
            f"({(cost_bps + slippage_bps) / 100:.2f}%). Price returns exclude dividends and currency conversion.",
            className="figure-note",
        ),
    ])
    validation = html.Div([
        html.H2("How much trust has the model earned?"),
        html.P(historical_readout, className="figure-note"),
        html.Div([
            metric_card(label, "Unavailable" if pd.isna(metrics[key]) else
                        (f"{metrics[key]:.1%}" if percent else f"{metrics[key]:.3f}"), note)
            for label, key, note, percent in classification_cards
        ], className="metrics-grid"),
        html.P(
            f"Walk-forward validation starts training on the oldest 70% of usable observations, then "
            f"refits every {retrain_days} sessions. Each fit uses only outcomes known at that time. "
            "Five-day outcomes overlap, so the observations are not independent trials.", className="figure-note",
        ),
        html.Div([
            html.Div([
                html.H2("Does confidence match reality?"),
                html.P("Calibration: closer to the dotted line is better. This chart diagnoses confidence; it does not adjust it.", className="figure-note"),
                dcc.Graph(figure=calibration_figure(predictions), config={"displayModeBar": False}, responsive=True),
            ], className="chart-card"),
            html.Div([
                html.H2("What influenced the model?"),
                html.P("Feature importance shows which inputs the fitted model uses most. It does not prove cause and effect.", className="figure-note"),
                dcc.Graph(figure=importance_figure(current_model), config={"displayModeBar": False}, responsive=True),
            ], className="chart-card"),
        ], className="two-column"),
        html.Button("Compare model scores", id="compare-button", n_clicks=0, className="secondary-button"),
        dcc.Loading(html.Div(id="comparison-results"), type="circle", color="#67e8f9"),
    ])
    method = html.Div([
        html.H2("The research, in plain language"),
        html.Div([
            html.Div([html.H3("1. Predict a direction"), html.P("XGBoost estimates whether the close five trading sessions later will be higher. The other outcome includes a lower or unchanged close. It does not predict a target price.")]),
            html.Div([html.H3("2. Test through time"), html.P("Walk-forward validation trains on the past and scores a later period. A label embargo prevents training on outcomes that had not happened yet.")]),
            html.Div([html.H3("3. Simulate a rule"), html.P("An UP signal enters a long position at the next open; DOWN moves to cash. HOLD keeps the previous position. There is no short selling, and the simulation starts in cash.")]),
            html.Div([html.H3("4. Compare the evidence"), html.P("SPY is an exchange-traded fund tracking the S&P 500. It provides a broad US-market reference. Compare the strategy with SPY and simply holding the selected stock.")]),
        ], className="explainer-grid"),
        html.Details([
            html.Summary("Technical details and limitations"),
            html.P("The model uses price, volume and market indicators. Current analyst opinions are displayed separately and never enter historical model training. Threshold changes affect the trading rule, not the model's probabilities."),
            html.P("The five-session classification target differs from the daily holding returns used by the trading rule. Trading costs are simplified, cash earns zero, and a remaining position is marked to market without a final liquidation fee. Yahoo Finance histories can be revised; raw prices exclude dividends and do not model FX conversion."),
            html.P("These results have been inspected during development. They are not an untouched final test and do not establish a reliable trading advantage. The separate Research Lab compares three classes relative to SPY, which differ from this page's direction target."),
        ], className="method-card"),
    ])
    return html.Div([
        dcc.Store(id="viewed-stock", data={"symbol": ticker, "company": company, "retrain_days": retrain_days}),
        html.Div([
            html.Div([
                html.Div([html.Span(ticker, className="stock-pill"), html.Span("Stock overview", className="eyebrow")], className="stock-identity"),
                html.H2(company), html.P(f"You are viewing {company} ({ticker}).", className="subtitle"),
            ]),
            html.Div([html.Div("Latest daily close", className="metric-label"),
                      html.Div(f"{price:,.2f}", className="metric-value"),
                      html.Div(f"{daily_change:+.2%} vs previous session", className="positive" if daily_change >= 0 else "negative"),
                      html.Div([f"{raw.index[-1].date()} · ", html.Span("Checking currency…", id="stock-currency")], className="metric-note")]),
        ], className="stock-hero"),
        html.Div([
            metric_card("5-session model view", view, f"Signal rule: {signal} · threshold {threshold:.0%}"),
            metric_card("Estimated chance of UP", f"{up_probability:.1%}", "Uncalibrated model estimate; not proven odds"),
            metric_card("Down or unchanged", f"{1 - up_probability:.1%}", "The other possible direction outcome"),
            metric_card("Historical accuracy", f"{metrics['accuracy']:.1%}", f"Majority reference: {metrics['majority_accuracy']:.1%}"),
        ], className="metrics-grid headline-metrics"),
        html.Div([
            html.Div("The quick read", className="eyebrow"), html.H3(view),
            html.P(explanation + " A higher model probability does not guarantee a better outcome."),
            html.P(historical_readout, className="stat-note"),
            html.Small(f"Current research snapshot · {latest.name.date()} · daily data; not a live quote. Cached runs expire within 15 minutes."),
        ], className="insight-panel"),
        dcc.Loading(html.Div(id="analyst-results", **{"aria-live": "polite"}), type="circle", color="#c4b5fd"),
        dcc.Tabs([
            dcc.Tab(overview, label="Price & performance"),
            dcc.Tab(validation, label="Model quality"),
            dcc.Tab(method, label="How it works"),
        ], className="research-tabs"),
    ])


def analyst_panel(snapshot, company):
    symbol = snapshot["symbol"]
    votes = snapshot.get("votes")
    total = sum(votes.values()) if votes else 0
    targets = snapshot.get("price_targets") or {}
    currency = snapshot.get("currency")
    quote_note = currency or "quote units; currency not supplied"
    content = [
        html.Div("A second perspective", className="eyebrow"),
        html.H2(f"What analysts say about {company}"),
        html.P("Analyst ratings reflect outside opinions and usually a longer horizon than this model's five sessions. They are not used to calculate the model signal.", className="figure-note"),
    ]
    if total:
        labels = {"buy": "Buy / strong buy", "hold": "Hold", "sell": "Sell / strong sell"}
        largest = max(votes.values())
        leaders = [labels[key] for key, count in votes.items() if count == largest]
        headline = f"Most common view: {leaders[0]}" if len(leaders) == 1 else "Analyst views are tied"
        content.extend([
            html.H3(headline),
            html.P(f"{total} ratings in Yahoo Finance's current-period summary. Counts refer to ratings; price targets may cover a different set of analysts.", className="stat-note"),
            html.Div([html.Div(style={"width": f"{count / total * 100}%"}, className=f"vote-segment {key}", title=f"{labels[key]}: {count}") for key, count in votes.items() if count], className="vote-bar", **{"aria-hidden": "true"}),
            html.Div([metric_card(labels[key], str(count), f"{count / total:.0%} of ratings") for key, count in votes.items()], className="metrics-grid analyst-votes"),
        ])
    else:
        content.append(html.P("Current analyst ratings are unavailable for this stock. Missing coverage is not a neutral or bearish opinion.", className="empty-state"))
    if targets:
        content.extend([
            html.H3("Analyst price targets"),
            html.Div([metric_card(label, f"{targets[key]:,.2f}" if targets.get(key) is not None else "Unavailable", quote_note)
                      for label, key in [("Low target", "low"), ("Average target", "mean"), ("High target", "high")]], className="metrics-grid analyst-votes"),
            html.P("Targets are estimates, not promised prices. The provider does not supply a common publication date or guaranteed target horizon for this snapshot.", className="stat-note"),
        ])
    else:
        content.append(html.P("Price targets are not available from the provider for this stock.", className="stat-note"))
    checked = snapshot.get("checked_at")
    try:
        checked = pd.Timestamp(checked).tz_convert("UTC").strftime("%b %d, %Y at %H:%M UTC")
    except (TypeError, ValueError):
        checked = "an unknown time"
    content.append(html.P([
        html.A("Source: Yahoo Finance", href=f"https://finance.yahoo.com/quote/{symbol}/analysis/", target="_blank", rel="noopener noreferrer"),
        f" · Checked {checked}. Retrieval time is not the date each analyst issued a rating.",
    ], className="figure-note"))
    if snapshot.get("errors"):
        content.append(html.P("Some provider data could not be loaded. Available information is shown above; try again later for missing fields.", className="stat-note"))
    return html.Section(content, className="analyst-card")

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
        html.Div([
            html.Div("A clearer view of the market", className="eyebrow"),
            html.H1("MarketSignal"),
            html.P("Explore a company. Understand the signal. Check the evidence.", className="subtitle"),
        ]),
        html.A("Open Research Lab ↗", href="/research-lab", className="lab-link"),
    ], className="page-header"),
    html.Main([
        html.Aside([
            html.H2("Find a company"),
            html.Label("Company name or stock symbol", htmlFor="ticker"),
            dcc.Dropdown(
                id="ticker", value="NVDA|NVIDIA Corporation",
                options=local_search_options("", "NVDA|NVIDIA Corporation"),
                placeholder="Try NVIDIA or NVDA…", searchable=True, clearable=False,
                className="stock-picker", optionHeight=48,
            ),
            html.P("Type at least two letters and choose a match. Selecting a company loads its research automatically.", className="stat-note"),
            html.Details([
                html.Summary("Research settings"),
                html.Label("Signal confidence threshold"),
                dcc.Slider(id="threshold", min=.5, max=.8, step=.01, value=.6,
                           marks={v: {"label": label, "style": {"color": "#b0c4df"}} for v, label in [(.5,"50%"),(.6,"60%"),(.7,"70%"),(.8,"80%")]}),
                html.P("Minimum model probability for an UP or DOWN signal. Otherwise the rule holds its prior position.", className="stat-note"),
                html.Label("Retrain every"),
                dcc.RadioItems(id="retrain-days", value=21,
                               options=[{"label": "21 sessions", "value": 21}, {"label": "42 sessions", "value": 42}, {"label": "63 sessions", "value": 63}],
                               className="retrain-picker"),
                html.Label("Trading cost (basis points)", htmlFor="cost-bps"),
                dcc.Input(id="cost-bps", value=10, type="number", min=0, max=100),
                html.Label("Slippage (basis points)", htmlFor="slippage-bps"),
                dcc.Input(id="slippage-bps", value=5, type="number", min=0, max=100),
                html.P("1 basis point (bp) = 0.01%. Slippage represents a worse execution price than expected.", className="stat-note"),
            ], open=True),
            html.Button("Apply settings", id="run-button", n_clicks=0, className="primary-button"),
            html.P("Daily data · educational research", className="disclaimer"),
        ], className="control-panel"),
        html.Section([
            dcc.Loading(html.Div(id="research-results", **{"aria-live": "polite"}),
                        type="circle", color="#67e8f9", overlay_style={"visibility": "hidden"}),
        ], className="results-panel"),
    ], className="app-shell"),
    html.Footer("MarketSignal is a research project. Model estimates and analyst opinions can be wrong; historical performance does not establish future returns."),
])


@app.callback(Output("ticker", "options"), Input("ticker", "search_value"), State("ticker", "value"))
def update_stock_options(query, selected):
    options = search_options(query or "", selected)
    # Retain the selected label when the user searches for a different company.
    if selected and not any(option["value"] == selected for option in options):
        options += local_search_options("", selected)
    return options


@app.callback(
    Output("research-results", "children"), Input("run-button", "n_clicks"), Input("ticker", "value"),
    State("threshold", "value"), State("retrain-days", "value"), State("cost-bps", "value"), State("slippage-bps", "value"),
    running=[(Output("run-button", "disabled"), True, False)],
)
def update_research(_, ticker, threshold, retrain_days, cost_bps, slippage_bps):
    try:
        return build_research_dashboard(ticker, threshold, retrain_days, cost_bps, slippage_bps)
    except Exception as exc:
        symbol, company = decode_selection(ticker)
        return html.Div([
            html.Div("Research run stopped", className="eyebrow"),
            html.H3(f"Could not load {company or symbol or 'this stock'}"),
            html.P(str(exc)),
            html.P("Check the settings or choose another company. New listings may not have enough history; data services can also be temporarily unavailable."),
        ], className="error-card", role="alert")


@ttl_cache(seconds=900, maxsize=32)
def load_analysts(symbol):
    return get_analyst_snapshot(symbol)


@app.callback(Output("analyst-results", "children"), Output("stock-currency", "children"), Input("viewed-stock", "data"))
def update_analysts(viewed):
    if not viewed:
        return "", "Currency unavailable"
    try:
        snapshot = load_analysts(viewed["symbol"])
        return analyst_panel(snapshot, viewed["company"]), snapshot.get("currency") or "Currency unavailable"
    except Exception:
        return html.Section([
            html.Div("A second perspective", className="eyebrow"),
            html.H2(f"Analyst view · {viewed['company']}"),
            html.P("Analyst data is temporarily unavailable. The model results above are still available."),
        ], className="analyst-card"), "Currency unavailable"


@app.callback(
    Output("comparison-results", "children"), Input("compare-button", "n_clicks"),
    State("viewed-stock", "data"), prevent_initial_call=True,
)
def update_comparison(clicks, viewed):
    if not clicks or not viewed:
        return ""
    try:
        comparison = run_model_comparison(viewed["symbol"], int(viewed["retrain_days"]))
        return html.Div([html.H3(f"Model comparison · {viewed['company']} ({viewed['symbol']})"), comparison_table(comparison)])
    except Exception as exc:
        return html.Div(str(exc), className="error-card", role="alert")


if __name__ == "__main__":
    app.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8050")), debug=False)
