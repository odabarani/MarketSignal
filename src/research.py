"""Leakage-aware building blocks for multi-stock MarketSignal research."""

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from src.features import FORECAST_HORIZON, TECHNICAL_FEATURES, add_features

DOWN = -1
NO_TRADE = 0
UP = 1
RESEARCH_TARGET = "Research_Target"


def add_cost_aware_target(
    frame,
    horizon=FORECAST_HORIZON,
    round_trip_cost_bps=30,
    minimum_edge_bps=0,
):
    """Label excess returns from the next open as DOWN, NO_TRADE, or UP."""
    if horizon < 1:
        raise ValueError("Forecast horizon must be at least one session.")
    if round_trip_cost_bps < 0 or minimum_edge_bps < 0:
        raise ValueError("Costs and minimum edge cannot be negative.")
    required = {"Open", "Close", "SPY_Open", "SPY_Close"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing target columns: {sorted(missing)}")

    result = frame.copy()
    stock_return = result["Close"].shift(-horizon) / result["Open"].shift(-1) - 1
    benchmark_return = (
        result["SPY_Close"].shift(-horizon) / result["SPY_Open"].shift(-1) - 1
    )
    excess_return = stock_return - benchmark_return
    hurdle = (round_trip_cost_bps + minimum_edge_bps) / 10_000

    target = pd.Series(NO_TRADE, index=result.index, dtype="float64")
    target.loc[excess_return > hurdle] = UP
    target.loc[excess_return < -hurdle] = DOWN
    target.loc[excess_return.isna()] = np.nan
    result["Forward_Excess_Return"] = excess_return
    result[RESEARCH_TARGET] = target
    result["Target_Available_At"] = pd.Series(
        result.index, index=result.index
    ).shift(-horizon)
    return result


def build_research_panel(
    market_data,
    sectors=None,
    horizon=FORECAST_HORIZON,
    round_trip_cost_bps=30,
    minimum_edge_bps=0,
):
    """Build one date-by-ticker panel without mixing rolling asset histories."""
    if not market_data:
        raise ValueError("Market data cannot be empty.")
    sectors = sectors or {}
    frames = []
    for ticker, raw in market_data.items():
        symbol = str(ticker).strip().upper()
        featured = add_features(raw)
        featured = add_cost_aware_target(
            featured,
            horizon=horizon,
            round_trip_cost_bps=round_trip_cost_bps,
            minimum_edge_bps=minimum_edge_bps,
        )
        featured["Ticker"] = symbol
        featured["Sector"] = sectors.get(symbol, "Unknown")
        frames.append(featured.reset_index(names="Date"))

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.set_index(["Date", "Ticker"]).sort_index()
    if panel.index.has_duplicates:
        raise ValueError("Research panel contains duplicate date-ticker rows.")
    return panel


@dataclass(frozen=True)
class ChronologicalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_end: pd.Timestamp
    validation_end: pd.Timestamp


def chronological_split(panel, train_fraction=0.60, validation_fraction=0.20):
    """Create locked date splits and embargo labels unavailable at each boundary."""
    if not isinstance(panel.index, pd.MultiIndex) or panel.index.names[:2] != [
        "Date", "Ticker"
    ]:
        raise ValueError("Research panel must use a Date, Ticker MultiIndex.")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Split fractions must be between zero and one.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must leave a test period.")

    dates = pd.DatetimeIndex(panel.index.get_level_values("Date").unique()).sort_values()
    if len(dates) < 10:
        raise ValueError("At least ten distinct dates are required.")
    train_position = max(1, int(len(dates) * train_fraction)) - 1
    validation_position = max(
        train_position + 1, int(len(dates) * (train_fraction + validation_fraction)) - 1
    )
    if validation_position >= len(dates) - 1:
        raise ValueError("Split fractions leave too few test dates.")
    train_end = dates[train_position]
    validation_end = dates[validation_position]

    feature_dates = panel.index.get_level_values("Date")
    available = pd.to_datetime(panel["Target_Available_At"])
    known = panel[RESEARCH_TARGET].notna()
    train = panel.loc[known & (feature_dates <= train_end) & (available <= train_end)].copy()
    validation = panel.loc[
        known & (feature_dates > train_end) & (feature_dates <= validation_end)
        & (available <= validation_end)
    ].copy()
    test = panel.loc[known & (feature_dates > validation_end)].copy()

    if any(part.empty for part in (train, validation, test)):
        raise ValueError("Chronological split produced an empty period.")
    return ChronologicalSplit(train, validation, test, train_end, validation_end)


def dataset_fingerprint(panel):
    """Return a compact identity for the exact panel used by an experiment."""
    hashed = pd.util.hash_pandas_object(
        panel.reindex(sorted(panel.columns), axis=1), index=True
    ).to_numpy()
    return sha256(hashed.tobytes()).hexdigest()


def record_experiment(path, config, metrics, panel, notes=""):
    """Append an immutable JSON-lines record of a research trial."""
    record = {
        "experiment_id": str(uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_fingerprint": dataset_fingerprint(panel),
        "config": config,
        "metrics": metrics,
        "notes": notes,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    return record
