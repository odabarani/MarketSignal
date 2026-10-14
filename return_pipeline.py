"""Run fixed expected-return candidates and chronological portfolio selection."""

import argparse
from hashlib import sha256
from html import escape
from importlib.metadata import version
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from research_pipeline import model_configuration
from src.research import record_experiment
from src.return_research import POLICY, RETURN_FEATURES, build_return_models, return_walk_forward


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/return_competition.json")
    parser.add_argument("--ledger", default="experiments/research.jsonl")
    parser.add_argument("--min-train-dates", type=int, default=504)
    parser.add_argument("--selection-dates", type=int, default=126)
    parser.add_argument("--evaluation-dates", type=int, default=126)
    args = parser.parse_args(argv)
    source, output, ledger = Path(args.input), Path(args.output), Path(args.ledger)
    manifest_path = source.with_suffix(".manifest.json")
    destinations = [output, output.with_suffix(".html"), output.with_suffix(".predictions.csv"),
                    output.with_suffix(".equity.csv"), output.with_suffix(".trades.csv")]
    protected = {source.resolve(), manifest_path.resolve(), ledger.resolve()}
    if (output.suffix != ".json" or ledger.resolve() in {source.resolve(), manifest_path.resolve()}
            or len({p.resolve() for p in destinations}) != len(destinations)
            or any(p.resolve() in protected for p in destinations)):
        parser.error("Keep outputs and ledger separate from the source and manifest.")
    manifest = json.loads(manifest_path.read_text())
    cutoff = manifest.get("validation_end")
    if not cutoff:
        parser.error("The snapshot needs its saved validation_end cutoff.")
    panel = pd.read_csv(source, parse_dates=["Date"]).set_index(["Date", "Ticker"]).sort_index()
    panel = panel.loc[panel.index.get_level_values("Date") <= pd.Timestamp(cutoff)]
    root = Path(__file__).resolve().parent
    digest = sha256()
    for path in sorted([root / "return_pipeline.py", *root.glob("src/*.py")]):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    models = build_return_models()
    windows = {"min_train_dates": args.min_train_dates, "selection_dates": args.selection_dates,
               "evaluation_dates": args.evaluation_dates}
    config = {"development_end": cutoff, "source_hash": sha256(source.read_bytes()).hexdigest(),
              "code_fingerprint": digest.hexdigest(), "policy": POLICY,
              "features": RETURN_FEATURES, "models": model_configuration(models), "windows": windows,
              "tickers": sorted(panel.index.get_level_values("Ticker").unique()),
              "versions": {name: version(name) for name in ["numpy", "pandas", "scikit-learn", "xgboost"]}}
    record_experiment(ledger, {**config, "stage": "return_started"}, {}, panel)
    try:
        report, predictions, results = return_walk_forward(panel, cutoff, models=models, **windows)
        yearly = []
        for (name, cost), result in results.items():
            returns = result.equity.pct_change(fill_method=None).dropna()
            for year, values in returns.groupby(returns.index.year):
                yearly.append({"candidate": name, "cost_bps_per_side": cost, "year": int(year),
                               "sessions": len(values), "net_return": float((1 + values).prod() - 1)})
        report["yearly"], report["config"] = yearly, config
        report["limitations"] = ["Five survivor-selected technology stocks; no historical membership or delisted coverage.",
                                  "Price returns exclude dividends, taxes, FX, and cash interest.",
                                  "Trailing volatility sizing is not a guarantee of future volatility or drawdown.",
                                  "Costs are assumptions; liquidity, market impact and execution capacity are not estimated.",
                                  "All historical development periods have been inspected. No candidate is promoted live."]
        report["experiment"] = record_experiment(ledger, {**config, "stage": "return_complete"},
                         {"summary": report["summary"], "forecast_errors": report["forecast_errors"]}, panel,
                         notes="Fixed candidates; selection strictly precedes scoring; reused development data.")
    except Exception as exc:
        record_experiment(ledger, {**config, "stage": "return_failed"}, {"error": str(exc)}, panel)
        raise
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=str, allow_nan=False) + "\n")
    predictions.to_csv(output.with_suffix(".predictions.csv"), index=False)
    pd.concat([result.equity.to_frame().assign(candidate=name, cost_bps_per_side=cost)
               for (name, cost), result in results.items()]).rename_axis("Date").to_csv(output.with_suffix(".equity.csv"))
    trades = [result.trades.assign(candidate=name, cost_bps_per_side=cost)
              for (name, cost), result in results.items() if not result.trades.empty]
    pd.concat(trades, ignore_index=True).to_csv(output.with_suffix(".trades.csv"), index=False)
    figure = go.Figure()
    for (name, cost), result in results.items():
        if cost == 15:
            figure.add_trace(go.Scatter(x=result.equity.index, y=result.equity, name=escape(name),
                                       visible=True if name in ["Chronological selector", "Ridge returns", "SPY buy & hold",
                                                                "SPY volatility target", "Equal-weight volatility target"] else "legendonly"))
    figure.update_layout(title="Expected-return competition · development data · 15 bps per side",
                         template="plotly_dark", hovermode="x unified", yaxis_title="Growth of 10,000",
                         legend=dict(orientation="h"), height=800)
    page = figure.to_html(include_plotlyjs=True, full_html=True)
    page = page.replace("<head>", "<head><title>MarketSignal return competition</title><style>body{background:#111;color:#eee;font:16px system-ui}p{padding:16px}</style>")
    page = page.replace("<body>", "<body><p>Earlier data chooses the model before each evaluation window. "
                        "Reused historical data; no live-profitability claim. Click a legend name to compare all candidates.</p>")
    output.with_suffix(".html").write_text(page)
    scores = pd.DataFrame(report["summary"])
    print(scores.loc[scores.cost_bps_per_side == 15, ["candidate", "annualized_return", "max_drawdown", "sharpe", "orders"]].to_string(index=False))
    print(f"Saved return experiment: {output}")


if __name__ == "__main__":
    main()
