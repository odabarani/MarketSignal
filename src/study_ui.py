"""Dashboard controls for the shared, retained historical study ledger."""
import json
import logging

from dash import Input, Output, State, ctx, dcc, html, no_update

from src.studies import StudyStore


def layout():
    return html.Section([
        html.Div("VALIDATE BEFORE YOU CONCLUDE", className="eyebrow"),
        html.H2("Locked historical study"),
        html.P("Prepare a study before exploring results. Validation selects from nine fixed candidates; "
               "the final period is evaluated once. Studies are shared across this server, not private accounts."),
        html.P("Uses your chosen ticker, fixed costs of 10 bps plus 5 bps slippage, and a 60/20/20 chronological split. "
               "Editable exploratory settings do not apply."),
        dcc.Store(id="study-context", storage_type="session"),
        dcc.Store(id="study-job", storage_type="session"),
        dcc.Interval(id="study-poll", interval=3000),
        html.Div([
            html.Button("Prepare / resume ticker study", id="study-prepare", n_clicks=0),
            html.Button("Run validation & freeze settings", id="study-validate", n_clicks=0, disabled=True),
            html.Button("Evaluate final period once", id="study-finalize", n_clicks=0, disabled=True),
            html.Button("Download study record", id="study-export", n_clicks=0, disabled=True),
        ], className="report-actions"),
        html.P("Starting final evaluation consumes that period even if computation is interrupted. "
               "Previously viewed historical data cannot become unseen."),
        html.Div(id="study-submit-status", role="status"),
        html.Div(id="study-status", role="status"),
        html.Div(id="study-view"),
        dcc.Download(id="study-download"),
    ], className="evidence-panel study-panel")


def render(study):
    protocol = study["protocol"]
    content = [html.H3(f"{protocol['dataset']['symbol']} · {study['state'].replace('_', ' ')}"),
               html.P(f"Study {study['study_id']}", className="snapshot-note"),
               html.P(f"Training through {protocol['train_end']}; validation {protocol['validation_start']} "
                      f"to {protocol['validation_end']}; final starts {protocol['final_start']}."),
               html.P("Selection: highest validation strategy Sharpe; first candidate breaks ties.")]
    if study.get("selection"):
        selection = study["selection"]
        settings = selection["settings"]
        content += [html.P(f"Frozen threshold {settings['threshold']:.0%}; retrain every {settings['retrain_days']} sessions."),
                    html.Table([
                        html.Thead(html.Tr([html.Th(x) for x in ("Threshold", "Retrain sessions", "Validation Sharpe")])),
                        html.Tbody([html.Tr([html.Td(f"{trial['settings']['threshold']:.0%}"),
                                           html.Td(trial['settings']['retrain_days']),
                                           html.Td(f"{trial['evaluation']['metrics']['strategy']['sharpe']:.3f}")])
                                    for trial in selection["trials"]]),
                    ])]
    if study.get("result"):
        evaluation = study["result"]["evaluation"]
        content += [html.H3("Retained final evaluation"),
                    html.P(f"{evaluation['evaluated_predictions']} predictions · {evaluation['prediction_start']} to {evaluation['prediction_end']}"),
                    html.Table([
                        html.Thead(html.Tr([html.Th(x) for x in ("Portfolio", "Total return", "Sharpe", "Max drawdown")])),
                        html.Tbody([html.Tr([
                            html.Td(label), html.Td(f"{evaluation['metrics'][key]['total_return']:.1%}"),
                            html.Td(f"{evaluation['metrics'][key]['sharpe']:.3f}"),
                            html.Td(f"{evaluation['metrics'][key]['max_drawdown']:.1%}"),
                        ]) for key, label in (("strategy", "Strategy (net)"), ("stock_buy_hold", "Stock buy & hold (gross)"),
                                               ("spy_buy_hold", "SPY buy & hold (gross)"))]),
                    ]), html.P("Gross benchmarks exclude trading costs. Download the record for all classification and portfolio metrics.")]
    if study["state"] in ("testing", "validating", "failed_final"):
        content.append(html.P("If the worker was interrupted, preserve this record for operator review. "
                              "Final evaluation cannot be reset from the dashboard."))
    return html.Div(content)


def register(app, market_store, jobs):
    @app.callback(
        Output("study-job", "data"), Output("study-submit-status", "children"),
        Input("study-prepare", "n_clicks"), Input("study-validate", "n_clicks"), Input("study-finalize", "n_clicks"),
        State("ticker", "value"), State("study-context", "data"), prevent_initial_call=True,
    )
    def submit(_prepare, _validate, _finalize, ticker, study_id):
        stage = {"study-prepare": "prepare", "study-validate": "validate", "study-finalize": "finalize"}[ctx.triggered_id]
        try:
            if stage == "prepare":
                from src.snapshots import normalize_symbol
                argument = normalize_symbol(ticker)
            else:
                if not isinstance(study_id, str):
                    raise ValueError("Prepare a study first.")
                study = StudyStore(market_store().path).read(study_id)
                allowed = ("prepared", "failed_validation") if stage == "validate" else ("selected",)
                if study["state"] not in allowed:
                    raise ValueError("This study stage is running or already consumed. Review its retained record.")
                argument = study_id
            job_id = jobs.submit(("study", stage, argument), {
                "kind": "study", "stage": stage, "argument": argument, "store_path": str(market_store().path),
            })
            return job_id, "Study request queued."
        except ValueError as error:
            return no_update, str(error)
        except Exception:
            logging.exception("Study submission failed")
            return no_update, "Study unavailable. Preserve its record and try again later."

    @app.callback(
        Output("study-context", "data"), Output("study-view", "children"), Output("study-status", "children"),
        Output("study-validate", "disabled"), Output("study-finalize", "disabled"), Output("study-export", "disabled"),
        Input("study-poll", "n_intervals"), Input("study-job", "data"), State("study-context", "data"),
    )
    def poll(_, job_id, study_id):
        message, busy = "", False
        if isinstance(job_id, str):
            job = jobs.status(job_id)
            busy = job["state"] in ("queued", "running")
            if job["state"] == "completed":
                study_id = job["result"]["study_id"]
            message = f"Study job: {job['state']}."
            if job["state"] in ("failed", "timed_out", "expired"):
                message += " Retained study state is authoritative; interrupted final stages require operator review."
        if not isinstance(study_id, str):
            return no_update, no_update, message, True, True, True
        try:
            study = StudyStore(market_store().path).read(study_id)
            return (study_id, render(study), message,
                    busy or study["state"] not in ("prepared", "failed_validation"),
                    busy or study["state"] != "selected", False)
        except Exception:
            logging.exception("Study read failed")
            return no_update, no_update, "Saved study is unavailable on this server.", True, True, True

    @app.callback(Output("study-download", "data"), Input("study-export", "n_clicks"),
                  State("study-context", "data"), prevent_initial_call=True)
    def download(clicks, study_id):
        if not clicks or not isinstance(study_id, str):
            return no_update
        try:
            record = StudyStore(market_store().path).read(study_id)
            return dcc.send_string(json.dumps(record, indent=2, allow_nan=False), f"MarketSignal-study-{study_id}.json")
        except Exception:
            logging.exception("Study export failed")
            return no_update
