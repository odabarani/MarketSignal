"""Browse retained server records without fetching prices or refitting models."""
import json
import logging
import re

from dash import Input, Output, State, dcc, html, no_update
import plotly.graph_objects as go

from src.reports import ReportStore
from src.studies import StudyStore
from src.study_ui import render as render_study


def recent_records(snapshots):
    records = []
    with snapshots.connect() as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        # Fixed table names and bounded reads; browser input never becomes SQL.
        for table, kind in (("reports", "report"), ("studies", "study")):
            if table not in tables:
                continue
            ids = connection.execute(f"SELECT id FROM {table} ORDER BY rowid DESC LIMIT 30").fetchall()
            for (identity,) in ids:
                record = load_record(snapshots, f"{kind}:{identity}")
                if kind == "report":
                    symbol, date = record["settings"]["ticker"], record["created_at"]
                    label = "Research report"
                else:
                    symbol, date = record["protocol"]["dataset"]["symbol"], record["protocol"]["created_at"]
                    label = f"Study · {record['state'].replace('_', ' ')}"
                records.append({"label": f"{symbol} · {label} · {date[:16].replace('T', ' ')} UTC · {identity[:8]}",
                                "value": f"{kind}:{identity}", "date": date})
    return [{"label": record["label"], "value": record["value"]}
            for record in sorted(records, key=lambda record: record["date"], reverse=True)]


def load_record(snapshots, identity):
    if not isinstance(identity, str) or not re.fullmatch(r"(report:[a-f0-9]{64}|study:[a-f0-9]{32})", identity):
        raise ValueError("Choose a saved record.")
    kind, key = identity.split(":")
    return ReportStore(snapshots).load(key) if kind == "report" else StudyStore(snapshots.path).read(key)


def render_report(report):
    settings, estimate = report["settings"], report["latest_estimate"]
    metrics = report["metrics"]["strategy"]
    curves = report["equity_curves"]
    figure = go.Figure()
    for key, label, color in (("strategy", "Strategy (net)", "#38bdf8"),
                               ("stock_buy_hold", "Stock (gross)", "#a78bfa"),
                               ("spy_buy_hold", "SPY (gross)", "#94a3b8")):
        figure.add_trace(go.Scatter(x=curves["dates"], y=curves[key], name=label, line={"color": color}))
    figure.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         margin={"l": 15, "r": 10, "t": 65, "b": 25}, legend={"orientation": "h", "y": 1.15},
                         yaxis_title="Portfolio value (USD)", hovermode="x unified")
    cards = [("Model estimate as of", estimate["date"]), ("UP probability then", f"{estimate['up_probability']:.1%}"),
             ("Net total return", f"{metrics['total_return']:.1%}"), ("Max drawdown", f"{metrics['max_drawdown']:.1%}")]
    return html.Div([
        html.H3(f"{settings['ticker']} · saved research report"),
        html.P("Archived results, not current market data. Opening this record does not refresh prices or rerun the model.", className="research-note"),
        html.P(f"Saved {report['created_at'][:19].replace('T', ' ')} UTC. "
               f"Threshold {settings['threshold']:.0%}; retrain every {settings['retrain_days']} sessions."),
        html.Div([html.Div([html.Div(label, className="metric-label"), html.Div(value, className="metric-value")],
                          className="metric-card") for label, value in cards], className="metrics-grid"),
        html.Div(dcc.Graph(figure=figure, responsive=True, config={"displayModeBar": False}), className="chart-card"),
        html.P("Exploratory evaluation. Historical returns do not predict future performance. Benchmarks exclude trading costs.", className="figure-note"),
    ])


def layout():
    return html.Section([
        html.Div("SAVED EVIDENCE", className="eyebrow"),
        html.H2("Saved research"),
        html.P("Shared server history · available from every device using this app. These records are not private accounts."),
        html.P("The latest 30 reports and 30 studies are listed. Choose a record to inspect its original results."),
        html.Button("Refresh saved records", id="library-refresh", n_clicks=0),
        html.Div(id="library-list-status", role="status"),
        html.Label("Saved report or study", htmlFor="library-record", className="library-label"),
        dcc.Dropdown(id="library-record", options=[], placeholder="Choose a saved record", clearable=False),
        html.Div(id="library-view"),
        html.Button("Download saved record", id="library-export", n_clicks=0, disabled=True, className="secondary-button"),
        html.Div(id="library-export-status", role="status"),
        dcc.Download(id="library-download"),
    ], className="evidence-panel library-panel")


def register(app, market_store):
    @app.callback(Output("library-record", "options"), Output("library-list-status", "children"),
                  Input("library-refresh", "n_clicks"), Input("workspace-mode", "value"))
    def refresh(_, mode):
        if mode != "saved":
            return no_update, no_update
        try:
            records = recent_records(market_store())
            return records, f"{len(records)} saved records available." if records else "No saved records yet. Run research or prepare a study to get started."
        except Exception:
            logging.exception("Saved record list unavailable")
            return [], "Saved history could not be loaded. Existing records have not been changed."

    @app.callback(Output("library-view", "children"), Output("library-export", "disabled"),
                  Input("library-record", "value"), Input("library-refresh", "n_clicks"))
    def inspect(identity, _):
        if not identity:
            return "", True
        try:
            record = load_record(market_store(), identity)
            return render_report(record) if identity.startswith("report:") else render_study(record), False
        except Exception:
            logging.exception("Saved record unavailable")
            return html.P("This saved record is unavailable or failed its integrity check. Refresh the list or contact the operator.", className="freshness-warning"), True

    @app.callback(Output("library-download", "data"), Output("library-export-status", "children"),
                  Input("library-export", "n_clicks"), State("library-record", "value"), prevent_initial_call=True)
    def download(clicks, identity):
        if not clicks:
            return no_update, no_update
        try:
            record = load_record(market_store(), identity)
            kind, key = identity.split(":")
            payload = {"report_id": key, "report": record} if kind == "report" else record
            return dcc.send_string(json.dumps(payload, indent=2, allow_nan=False), f"MarketSignal-{kind}-{key}.json"), "Saved record downloaded."
        except Exception:
            logging.exception("Saved record export failed")
            return no_update, "Download failed. The saved record has not been changed."
