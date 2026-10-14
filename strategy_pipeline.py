"""Score saved development predictions as portfolios, without fitting new models."""

import argparse
from html import escape
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import log_loss

from src.portfolio import PROBABILITIES, compare_portfolios
from src.research import RESEARCH_TARGET, record_experiment


def load_verified_inputs(source, report_path):
    report = json.loads(report_path.read_text())
    manifest = json.loads(source.with_suffix(".manifest.json").read_text())
    cutoff = report.get("development_end")
    if not cutoff or cutoff != manifest.get("validation_end"):
        raise ValueError("Report and snapshot must share the saved development cutoff.")
    source_hash = sha256(source.read_bytes()).hexdigest()
    if report.get("config", {}).get("source_hash") != source_hash:
        raise ValueError("The prediction report was generated from a different price snapshot.")
    panel = pd.read_csv(source, parse_dates=["Date", "Target_Available_At"])
    # Slice before computing any result. Previously inspected later data is not
    # relabelled as an untouched test, and cannot change these diagnostics.
    panel = panel.loc[panel.Date <= pd.Timestamp(cutoff)].set_index(["Date", "Ticker"]).sort_index()
    predictions_path = report_path.with_suffix(".predictions.csv")
    predictions = pd.read_csv(predictions_path, parse_dates=["Date"])
    reported = {item["candidate"]: item for item in report["summary"]}
    if set(predictions.candidate) != set(reported):
        raise ValueError("Prediction candidates do not match their saved report.")
    for name, group in predictions.groupby("candidate"):
        sample = panel.reindex(pd.MultiIndex.from_frame(group[["Date", "Ticker"]]))
        target = sample[RESEARCH_TARGET].to_numpy()
        if not np.array_equal(target, group[RESEARCH_TARGET].to_numpy()):
            raise ValueError("Prediction outcomes do not match their source snapshot.")
        score = log_loss(target + 1, group[PROBABILITIES].to_numpy(), labels=[0, 1, 2])
        if (len(group) != reported[name]["rows"]
                or not np.isclose(score, reported[name]["log_loss"], atol=1e-10, rtol=1e-8)):
            raise ValueError("Prediction probabilities do not match their saved scores.")
    return panel, predictions, cutoff, source_hash, predictions_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Saved price panel CSV.")
    parser.add_argument("--report", required=True, help="Matching walk-forward JSON report.")
    parser.add_argument("--output", default="data/strategy_comparison.json")
    parser.add_argument("--ledger", default="experiments/research.jsonl")
    args = parser.parse_args(argv)
    source, report_path, output = Path(args.input), Path(args.report), Path(args.output)
    destinations = [output, output.with_suffix(".html"), output.with_suffix(".equity.csv"),
                    output.with_suffix(".trades.csv")]
    inputs = {source.resolve(), source.with_suffix(".manifest.json").resolve(),
                 report_path.resolve(), report_path.with_suffix(".predictions.csv").resolve(),
                 report_path.with_suffix(".html").resolve()}
    ledger_path = Path(args.ledger).resolve()
    protected = inputs | {ledger_path}
    if (output.suffix != ".json" or len({p.resolve() for p in destinations}) != 4
            or ledger_path in inputs or any(p.resolve() in protected for p in destinations)):
        parser.error("Use a separate .json output to preserve inputs and the experiment ledger.")
    panel, predictions, cutoff, source_hash, predictions_path = load_verified_inputs(source, report_path)
    # Fixed before seeing returns: no threshold/holding-period sweep in this CLI.
    settings = {"horizon": 5, "probability_threshold": .55, "top_n": 2, "max_weight": .5,
                "cost_scenarios": [15, 30, 50]}
    root = Path(__file__).resolve().parent
    digest = sha256()
    for path in [root / "strategy_pipeline.py", root / "src/portfolio.py", root / "src/backtest.py"]:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    config = {**settings, "source_hash": source_hash,
              "predictions_hash": sha256(predictions_path.read_bytes()).hexdigest(),
              "report_hash": sha256(report_path.read_bytes()).hexdigest(),
              "code_fingerprint": digest.hexdigest(), "development_end": cutoff,
              "tickers": sorted(predictions.Ticker.unique()),
              "versions": {name: version(name) for name in ["numpy", "pandas", "scikit-learn", "plotly"]}}
    record_experiment(args.ledger, {**config, "stage": "portfolio_started"}, {}, panel)
    try:
        summary, results = compare_portfolios(panel, predictions, cutoff, **settings)
        yearly = []
        for (name, cost), result in results.items():
            returns = result.equity.pct_change(fill_method=None).dropna()
            for year, values in returns.groupby(returns.index.year):
                yearly.append({"strategy": name, "cost_bps_per_side": cost, "year": int(year),
                               "sessions": len(values), "net_return": float((1 + values).prod() - 1)})
        record = record_experiment(args.ledger, {**config, "stage": "portfolio_complete"},
                                   {"summary": summary.to_dict("records")}, panel,
                                   notes="Reused development data; no strategy promotion or untouched-test claim.")
    except Exception as exc:
        record_experiment(args.ledger, {**config, "stage": "portfolio_failed"},
                          {"error": str(exc)}, panel)
        raise
    report = {"config": config, "summary": summary.to_dict("records"), "yearly": yearly,
              "experiment_id": record["experiment_id"],
              "status": "Development diagnostic; no algorithm is approved for live trading.",
              "limitations": ["Five survivor-selected technology stocks; not a historical universe.",
                              "Price returns exclude dividends, taxes, FX, and cash interest.",
                              "Costs are hypothetical per-side scenarios; liquidity and impact are not estimated.",
                              "Fixed positions liquidate at the fifth close; repeated cohorts repay entry/exit costs.",
                              "The probability rule predicts relative performance, not positive absolute returns.",
                              "Previously viewed development data cannot establish future profitability."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    pd.concat([result.equity.rename("equity").to_frame().assign(strategy=name, cost_bps_per_side=cost)
               for (name, cost), result in results.items()]).rename_axis("Date").to_csv(output.with_suffix(".equity.csv"))
    pd.concat([result.trades.assign(strategy=name, cost_bps_per_side=cost)
               for (name, cost), result in results.items() if not result.trades.empty],
              ignore_index=True).to_csv(output.with_suffix(".trades.csv"), index=False)
    figure = go.Figure()
    for (name, cost), result in results.items():
        if cost == 15:
            figure.add_trace(go.Scatter(x=result.equity.index, y=result.equity, name=escape(name)))
    figure.update_layout(title="Actual portfolio outcomes · development data · 15 bps per side",
                         template="plotly_dark", hovermode="x unified", yaxis_title="Growth of 10,000",
                         legend=dict(orientation="h"), height=850)
    page = figure.to_html(include_plotlyjs=True, full_html=True)
    page = page.replace("<body>", "<body><p style='font:16px system-ui;padding:20px'>"
                        "Previously inspected development data. Cash, SPY, momentum, and all model candidates "
                        "are shown; this experiment does not establish a profitable live strategy.</p>")
    output.with_suffix(".html").write_text(page)
    columns = ["strategy", "cost_bps_per_side", "annualized_return", "max_drawdown", "sharpe", "return_vs_spy"]
    print(summary.loc[summary.cost_bps_per_side == 15, columns].to_string(index=False))
    print(f"Saved portfolio report: {output}")


if __name__ == "__main__":
    main()
