"""Select once on development data, then score fixed candidates on a later period."""

import argparse
from hashlib import sha256
from html import escape
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.base import clone
from threadpoolctl import threadpool_limits

from src.portfolio import BENCHMARK
from src.rebalancing import rebalance_backtest
from src.research import record_experiment
from src.return_research import (
    AVAILABLE, POLICY, RETURN_FEATURES, RETURN_TARGET, _forecast,
    allocation_rules, build_return_models, prepare_return_panel,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--development-report", required=True)
    parser.add_argument("--output", default="data/locked_return_test.json")
    parser.add_argument("--ledger", default="experiments/research.jsonl")
    args = parser.parse_args(argv)

    source, development, output, ledger = map(Path, [
        args.input, args.development_report, args.output, args.ledger,
    ])
    manifest_path = source.with_suffix(".manifest.json")
    destinations = [output, output.with_suffix(".html"), output.with_suffix(".equity.csv"),
                    output.with_suffix(".trades.csv")]
    protected = {source.resolve(), manifest_path.resolve(), development.resolve(), ledger.resolve()}
    if (output.suffix != ".json" or len({p.resolve() for p in destinations}) != len(destinations)
            or any(p.resolve() in protected for p in destinations)):
        parser.error("Use a separate JSON output, chart, equity, and trades path.")
    if any(path.exists() for path in destinations):
        parser.error("This locked evaluation already has output files; choose a new untouched dataset.")

    source_hash = sha256(source.read_bytes()).hexdigest()
    if ledger.exists():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            prior = json.loads(line)
            prior_config = prior.get("config", {})
            if (prior_config.get("stage") == "locked_return_test_complete"
                    and prior_config.get("source_hash") == source_hash):
                parser.error("This frozen source already has a completed locked evaluation in the ledger.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected_report = json.loads(development.read_text(encoding="utf-8"))
    cutoff = pd.Timestamp(manifest["validation_end"])
    if (manifest.get("source_hash") != source_hash
            or selected_report.get("config", {}).get("source_hash") != source_hash
            or pd.Timestamp(selected_report.get("development_end")) != cutoff
            or selected_report.get("policy") != POLICY):
        parser.error("Development report, source snapshot, cutoff, and fixed policy must match.")

    cost = int(POLICY["base_cost_bps"])
    budget = float(POLICY["drawdown_budget"])
    development_rows = [row for row in selected_report["summary"]
                        if row["cost_bps_per_side"] == cost and row["candidate"] != "Chronological selector"
                        and row["max_drawdown"] >= -budget]
    if not development_rows:
        parser.error("No fixed candidate met the development drawdown screen.")
    frozen = sorted(development_rows,
                    key=lambda row: (-row["annualized_return"], row["candidate"]))[0]["candidate"]

    raw = pd.read_csv(source, parse_dates=["Date"]).set_index(["Date", "Ticker"]).sort_index()
    dates = raw.index.get_level_values("Date").unique().sort_values()
    if cutoff not in dates:
        parser.error("The development cutoff must be a session in the frozen panel.")
    cutoff_position = dates.get_loc(cutoff)
    if len(dates) - cutoff_position - 1 < 126:
        parser.error("At least 126 later sessions are required for the locked evaluation.")

    prepared = prepare_return_panel(raw, dates[-1], POLICY["horizon"])
    feature_dates = prepared.index.get_level_values("Date")
    complete_features = np.isfinite(prepared[RETURN_FEATURES].to_numpy(dtype=float)).all(axis=1)
    known_targets = (prepared[RETURN_TARGET].notna() & prepared[AVAILABLE].notna()
                     & (prepared[AVAILABLE] <= cutoff) & (feature_dates <= cutoff))
    training = prepared.loc[complete_features & known_targets]
    if training.empty:
        parser.error("No complete training outcomes were available at the frozen cutoff.")

    # The first allocation is made after the final development close and fills
    # at the first held-out open. The final two sessions are reserved for exit.
    final_date = dates[-1]
    decision_positions = np.arange(cutoff_position, len(dates) - 2, POLICY["horizon"])
    decisions = dates[decision_positions]
    test_features = prepared.loc[complete_features & feature_dates.isin(decisions)]
    if len(test_features) != len(decisions) * len(manifest["tickers"]):
        parser.error("Held-out decision dates do not have a complete feature row for every asset.")

    forecasts = []
    models = build_return_models()
    with threadpool_limits(limits=2):
        for name, model in models.items():
            fitted = clone(model).fit(training[RETURN_FEATURES], training[RETURN_TARGET])
            forecasts.append(_forecast(fitted, test_features, RETURN_FEATURES, name, "locked"))
    forecasts = pd.concat(forecasts, ignore_index=True)
    allocations = allocation_rules(prepared, forecasts, decisions, policy=POLICY)
    if frozen not in allocations:
        parser.error(f"The selected development candidate is unavailable: {frozen}.")

    results, summaries, equity_rows, trade_rows = {}, [], [], []
    yearly_rows, regime_rows = [], []
    market = prepared.groupby(level="Date")["VIX"].first()
    training_vix = market.loc[market.index <= cutoff]
    high_vix_threshold = float(training_vix.median())
    for candidate, weights in allocations.items():
        for cost_bps in [15, 30, 50]:
            result = rebalance_backtest(prepared, weights, final_date, cost_bps)
            results[candidate, cost_bps] = result
            summaries.append({"candidate": candidate, "cost_bps_per_side": cost_bps,
                              **result.metrics, "selected_before_holdout": candidate == frozen})
            if cost_bps == cost:
                equity_rows.append(result.equity.rename(candidate).to_frame())
            daily = result.equity.pct_change(fill_method=None).dropna()
            for year, values in daily.groupby(daily.index.year):
                yearly_rows.append({"candidate": candidate, "cost_bps_per_side": cost_bps,
                                    "year": int(year), "sessions": len(values),
                                    "net_return": float((1 + values).prod() - 1)})
            if cost_bps == cost:
                daily_vix = market.reindex(daily.index)
                for regime, mask in [("VIX at or below development median", daily_vix <= high_vix_threshold),
                                     ("VIX above development median", daily_vix > high_vix_threshold)]:
                    values = daily.loc[mask.fillna(False)]
                    if not values.empty:
                        regime_rows.append({"candidate": candidate, "regime": regime,
                                            "development_vix_median": high_vix_threshold,
                                            "sessions": len(values),
                                            "net_return": float((1 + values).prod() - 1)})
            if not result.trades.empty:
                trades = result.trades.copy()
                trades["candidate"], trades["cost_bps_per_side"] = candidate, cost_bps
                trade_rows.append(trades)

    summary = {"dataset_source_hash": source_hash, "development_end": str(cutoff.date()),
               "holdout_start": str(dates[cutoff_position + 1].date()),
               "holdout_end": str(final_date.date()), "development_candidate": frozen,
               "selection_rule": f"Highest development annualized net return at {cost} bps per side, among fixed candidates with drawdown no worse than {-budget:.0%}; alphabetical tie-break.",
               "policy": POLICY, "holdout_sessions": len(dates) - cutoff_position - 1,
               "summary": summaries, "yearly_returns": yearly_rows,
               "vix_regime_returns": regime_rows,
               "limitations": ["One historical ETF universe; it does not cover historical individual-stock membership.",
                               "The source uses vendor-adjusted OHLC and may be revised; hash identifies this frozen file.",
                               "Costs are fixed percentage assumptions; liquidity, market impact, taxes, cash interest, and FX are excluded.",
                               "The evaluation is one historical holdout and does not establish future profitability."]}
    figure = go.Figure()
    for candidate, frame in ((name, frame) for item in equity_rows for name, frame in item.items()):
        figure.add_trace(go.Scatter(x=frame.index, y=frame.values, name=escape(candidate),
                                    line={"width": 3 if candidate == frozen else 1.5},
                                    visible=True if candidate in [frozen, "SPY buy & hold", "Cash",
                                                                  "Equal-weight volatility target"] else "legendonly"))
    figure.update_layout(title=f"Locked ETF evaluation · selected: {escape(frozen)} · {cost} bps per side",
                         template="plotly_dark", hovermode="x unified", yaxis_title="Growth of 10,000",
                         legend={"orientation": "h"}, height=760)
    page = figure.to_html(include_plotlyjs=True, full_html=True)
    page = page.replace("<head>", "<head><title>MarketSignal locked evaluation</title><style>body{background:#111;color:#eee;font:16px system-ui}p{padding:16px}</style>")
    page = page.replace("<body>", "<body><p>Strategy selected using data through " + str(cutoff.date()) +
                        ". Later sessions are scored once at the frozen settings. This is research, not a live-profitability claim.</p>")

    fingerprint = sha256()
    for path in sorted([Path(__file__).resolve(), Path("src/return_research.py").resolve(),
                        Path("src/rebalancing.py").resolve()]):
        fingerprint.update(path.name.encode())
        fingerprint.update(path.read_bytes())
    ledger_config = {"stage": "locked_return_test_complete", "source_hash": source_hash,
                     "development_end": str(cutoff.date()), "holdout_end": str(final_date.date()),
                     "selected_candidate": frozen, "code_fingerprint": fingerprint.hexdigest()}
    record = record_experiment(ledger, ledger_config, {"summary": summaries}, prepared,
                               notes="Candidate selected before holdout; all later dates evaluated once at frozen settings.")
    summary["experiment_id"] = record["experiment_id"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pd.concat(equity_rows, axis=1).rename_axis("Date").to_csv(output.with_suffix(".equity.csv"))
    pd.concat(trade_rows, ignore_index=True).to_csv(output.with_suffix(".trades.csv"), index=False)
    output.with_suffix(".html").write_text(page, encoding="utf-8")
    scores = pd.DataFrame(summaries)
    print(f"Frozen candidate from development: {frozen}")
    print(f"Locked period: {summary['holdout_start']} to {summary['holdout_end']}")
    print(scores.loc[scores.cost_bps_per_side == cost,
                     ["candidate", "annualized_return", "max_drawdown", "sharpe", "orders"]].to_string(index=False))
    print(f"Saved one-time evaluation: {output}")


if __name__ == "__main__":
    main()
