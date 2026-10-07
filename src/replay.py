"""Verify a completed report offline against its retained dataset and runtime."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
import sqlite3

import pandas as pd
from threadpoolctl import threadpool_limits

from src.backtest import backtest, calculate_metrics, equity_from_returns, trading_metrics
from src.features import TECHNICAL_FEATURES, add_features
from src.model import predict_latest, walk_forward_evaluate
from src.reports import json_safe, runtime_provenance
from src.settings import validate_settings
from src.signals import generate_signal
from src.snapshots import SnapshotStore

ROOT = Path(__file__).resolve().parents[1]


def verify_runtime(report, root):
    if report.get("runtime") != runtime_provenance(root):
        raise ValueError("Runtime or source differs from the report. Restore its recorded source and dependency versions before replay.")


def compare(expected, actual, path="results", tolerance=1e-8):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or expected.keys() != actual.keys():
            raise ValueError(f"Replay fields differ at {path}.")
        for key in expected:
            compare(expected[key], actual[key], f"{path}.{key}", tolerance)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise ValueError(f"Replay sample lengths differ at {path}.")
        for offset, (left, right) in enumerate(zip(expected, actual)):
            compare(left, right, f"{path}[{offset}]", tolerance)
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float)) or isinstance(actual, bool) or not math.isclose(
            expected, actual, rel_tol=tolerance, abs_tol=tolerance
        ):
            raise ValueError(f"Replay numerical mismatch at {path}.")
    elif expected != actual:
        raise ValueError(f"Replay mismatch at {path}.")


def replay(envelope, store_path, root=ROOT):
    if not isinstance(envelope, dict):
        raise ValueError("Expected a downloaded report object.")
    report = envelope.get("report")
    if not isinstance(report, dict) or report.get("schema_version") != 2:
        raise ValueError("Offline replay supports report schema version 2 only.")
    payload = json.dumps(report, sort_keys=True, allow_nan=False)
    if sha256(payload.encode()).hexdigest() != envelope.get("report_id"):
        raise ValueError("Report checksum failed. Use an unchanged downloaded report.")
    verify_runtime(report, root)
    compare({"kind": "exploratory_walk_forward", "selection_locked": False},
            report.get("evaluation_protocol"), "evaluation_protocol")
    raw = SnapshotStore(store_path).load_readonly(report["dataset"]["id"])
    compare(report["dataset"], raw.attrs["snapshot"], "dataset")
    settings = report["settings"]
    threshold, retrain_days, cost, slippage = validate_settings(
        settings["threshold"], settings["retrain_days"],
        settings["transaction_cost_bps"], settings["slippage_bps"],
    )
    if (settings["ticker"] != raw.attrs["snapshot"]["symbol"]
            or settings.get("starting_capital_usd") != 10000
            or settings.get("liquidate_at_end") is not True
            or settings.get("benchmark_transaction_cost_bps") != 0):
        raise ValueError("Report settings are incompatible with this replay workflow.")
    compare(report["model"], {
        "name": "XGBoost", "random_seed": 42,
        "target": "Close five sessions later exceeds decision-date close",
        "initial_train_fraction": .7, "forecast_sessions": 5,
    }, "model")
    features = add_features(raw)
    with threadpool_limits(limits=2):
        _, predictions, classification = walk_forward_evaluate(features, TECHNICAL_FEATURES,
                                                                retrain_every=retrain_days)
        _, latest, probability = predict_latest(features, TECHNICAL_FEATURES)
    equity, detail = backtest(predictions, transaction_cost_bps=cost,
                              slippage_bps=slippage, confidence_threshold=threshold)
    dates = pd.DatetimeIndex(detail["RealizationDate"])
    stock = equity_from_returns(pd.Series(detail["market_return"].to_numpy(), index=dates),
                                10000, equity.index[0])
    spy = equity_from_returns(pd.Series(detail["benchmark_return"].to_numpy(), index=dates),
                              10000, equity.index[0])
    strategy = calculate_metrics(equity)
    strategy.update(trading_metrics(detail))
    actual = json_safe({
        "sample": {"history_start": str(raw.index.min().date()),
                   "history_end": str(raw.index.max().date()), "history_rows": len(raw),
                   "evaluation_start": str(predictions.index.min().date()),
                   "evaluation_end": str(predictions.index.max().date()),
                   "evaluated_predictions": len(predictions)},
        "latest_estimate": {"date": str(latest.name.date()), "close_usd": float(latest["Close"]),
                            "up_probability": probability,
                            "signal": generate_signal(int(probability >= .5), probability, threshold)},
        "metrics": {"classification": classification, "strategy": strategy,
                    "stock_buy_hold": calculate_metrics(stock), "spy_buy_hold": calculate_metrics(spy)},
        "equity_curves": {"dates": [str(date.date()) for date in equity.index],
                          "strategy": equity.tolist(), "stock_buy_hold": stock.tolist(),
                          "spy_buy_hold": spy.tolist()},
    })
    compare({key: report[key] for key in actual}, actual)
    return {"status": "verified", "report_id": envelope["report_id"],
            "dataset_id": report["dataset"]["id"], "tolerance": 1e-8,
            "evaluated_predictions": len(predictions)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = replay(json.loads(args.report.read_text()), args.store)
    except (ValueError, KeyError, OSError, sqlite3.Error) as error:
        print(f"Replay failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
