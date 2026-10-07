"""Retain completed research reports so exports never recompute a run."""

from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
import math
import platform
from pathlib import Path
import re

import numpy as np
from src.backtest import calculate_metrics
from src.market_calendar import session_quality

PACKAGES = ("dash", "yfinance", "pandas", "numpy", "scikit-learn", "xgboost", "plotly", "scipy", "threadpoolctl", "exchange-calendars")


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def runtime_provenance(root):
    root = Path(root)
    paths = [root / "app.py", root / "requirements.txt", root / "requirements.lock",
             *sorted((root / "src").glob("*.py"))]
    hashes = {str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
              for path in paths if path.is_file()}
    return {"python_version": platform.python_version(), "source_files_sha256": hashes,
            "dependencies": {package: version(package) for package in PACKAGES}}


def build_report(raw, predictions, latest, up_probability, signal, settings,
                 classification, strategy, equity, stock, spy, root):
    return json_safe({
        "schema_version": 2,
        "evaluation_protocol": {"kind": "exploratory_walk_forward", "selection_locked": False},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dict(raw.attrs["snapshot"]),
        "data_quality": session_quality(raw.index.max()),
        "settings": settings,
        "model": {"name": "XGBoost", "random_seed": 42,
                  "target": "Close five sessions later exceeds decision-date close",
                  "initial_train_fraction": .7, "forecast_sessions": 5},
        "sample": {
            "history_start": str(raw.index.min().date()),
            "history_end": str(raw.index.max().date()), "history_rows": len(raw),
            "evaluation_start": str(predictions.index.min().date()),
            "evaluation_end": str(predictions.index.max().date()),
            "evaluated_predictions": len(predictions),
        },
        "latest_estimate": {"date": str(latest.name.date()), "close_usd": float(latest["Close"]),
                            "up_probability": up_probability, "signal": signal},
        "metrics": {"classification": classification, "strategy": strategy,
                    "stock_buy_hold": calculate_metrics(stock),
                    "spy_buy_hold": calculate_metrics(spy)},
        "equity_curves": {
            "dates": [str(date.date()) for date in equity.index],
            "strategy": equity.tolist(), "stock_buy_hold": stock.tolist(),
            "spy_buy_hold": spy.tolist(),
        },
        "runtime": runtime_provenance(root),
        "limitations": [
            "Research simulation; no orders placed. Historical performance is not a forecast.",
            "Daily historical prices, not real-time quotes; provider data may be stale or revised.",
            "Five-session outcomes overlap and are not independent observations.",
            "Probabilities are uncalibrated model estimates; the calibration chart is diagnostic.",
            "Majority accuracy is a retrospective evaluation-sample comparator.",
            "Training baseline chooses the majority class from labels available at each fit.",
            "Changing settings after viewing results can overfit this evaluation.",
            "Long/cash, next-open simulation; costs charged on entry, exit and final liquidation.",
            "Cash dividends credited for holding into ex-date; fractional reinvestment is simplified.",
            "Buy-and-hold benchmarks include cash dividends but exclude transaction costs.",
            "252-session annualization; Sharpe uses zero risk-free rate.",
            "Positive interval rate measures open-to-open returns, not profitable trades.",
            "Yahoo split-normalized histories are retrospective, not point-in-time archives.",
            "Public data-display rights require review.",
            "Offline replay requires matching source/runtime and the retained dataset; different-hardware reproducibility is unverified.",
        ],
    })


class ReportStore:
    def __init__(self, snapshots):
        self.snapshots = snapshots

    def save(self, report):
        payload = json.dumps(report, sort_keys=True, allow_nan=False)
        report_id = sha256(payload.encode()).hexdigest()
        with self.snapshots.connect() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS reports (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
            connection.execute("INSERT OR IGNORE INTO reports VALUES (?, ?)", (report_id, payload))
        return report_id

    def load(self, report_id):
        if not isinstance(report_id, str) or not re.fullmatch(r"[a-f0-9]{64}", report_id):
            raise ValueError("Run research before downloading a report.")
        with self.snapshots.connect() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS reports (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
            row = connection.execute("SELECT payload FROM reports WHERE id = ?", (report_id,)).fetchone()
        if row is None:
            raise ValueError("This report is unavailable. Run research again.")
        if sha256(row[0].encode()).hexdigest() != report_id:
            raise ValueError("Report integrity check failed. Run research again.")
        return json.loads(row[0])
